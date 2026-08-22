"""``PUT /api/profiles/{name}/soul`` must not destroy an existing SOUL.md.

The dashboard persona editor replaces the whole document on every Save. A bare
``write_text()`` truncates SOUL.md before the new body lands, and the paired
``GET`` reports an unreadable file as ``{"content": "", "exists": False}`` — so
an interrupted save presents as "your persona was never set" and the editor's
next Save persists that empty document over the original.

Lives in its own module rather than ``test_web_server.py`` to keep the harness
small and focused on this one endpoint pair.
"""

from __future__ import annotations

import contextlib
import os
import stat
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


SOUL = "# Persona\n\nYou are a careful, terse assistant.\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "soul-test-token")
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = "Bearer soul-test-token"
        yield c


@pytest.fixture()
def profile_dir(tmp_path, monkeypatch) -> Path:
    """Create a real profile directory under the test HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import profiles as profiles_mod

    d = profiles_mod.get_profile_dir("demo")
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestSoulWriteDurability:
    def test_put_replaces_soul(self, client, profile_dir: Path):
        """Happy path: the editor's Save still works."""
        (profile_dir / "SOUL.md").write_text(SOUL, encoding="utf-8")

        r = client.put("/api/profiles/demo/soul", json={"content": "# New\n"})

        assert r.status_code == 200, r.text
        assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == "# New\n"

    def test_put_creates_soul_when_absent(self, client, profile_dir: Path):
        """A first save has no prior file to preserve permissions from."""
        assert not (profile_dir / "SOUL.md").exists()

        r = client.put("/api/profiles/demo/soul", json={"content": SOUL})

        assert r.status_code == 200, r.text
        assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == SOUL

    def test_existing_soul_survives_an_interrupted_save(
        self, client, profile_dir: Path
    ):
        soul = profile_dir / "SOUL.md"
        soul.write_text(SOUL, encoding="utf-8")
        original = soul.read_bytes()

        def boom(fd):
            raise OSError("simulated crash mid-write")

        # Scoped context so restoring os.fsync doesn't also undo the
        # HERMES_HOME patch the client/profile_dir fixtures installed.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "fsync", boom)
            r = client.put(
                "/api/profiles/demo/soul", json={"content": "# clobbered\n"}
            )

        assert r.status_code == 500
        # The persona the user already had must survive verbatim...
        assert soul.read_bytes() == original
        # ...and the paired GET must not report it as never-set, which is what
        # would make the next Save persist an empty document.
        g = client.get("/api/profiles/demo/soul")
        assert g.status_code == 200, g.text
        assert g.json()["exists"] is True
        assert g.json()["content"] == SOUL
        # No temp file left behind in the profile directory.
        assert list(profile_dir.glob("*.tmp")) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_existing_file_mode_is_preserved(self, client, profile_dir: Path):
        """Profile SOUL.md is created 0644 and never run through
        ``_secure_file``; saving from the dashboard must not change that."""
        soul = profile_dir / "SOUL.md"
        soul.write_text(SOUL, encoding="utf-8")
        os.chmod(soul, 0o644)

        r = client.put("/api/profiles/demo/soul", json={"content": "# New\n"})

        assert r.status_code == 200, r.text
        mode = stat.S_IMODE(soul.stat().st_mode)
        assert mode == 0o644, f"mode changed to {oct(mode)}"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_created_file_mode_is_not_tightened(self, client, profile_dir: Path):
        """The first-ever Save must not leave SOUL.md owner-only.

        There is no prior file to copy permissions from, and
        ``atomic_write_text`` swaps in a ``tempfile.mkstemp`` file (0600).
        Profile creation seeds SOUL.md with a plain ``write_text()`` and
        chmods only ``.env`` to 0600, so routing this endpoint through the
        atomic writer must not tighten the persona document as a side effect.
        """
        soul = profile_dir / "SOUL.md"
        assert not soul.exists()

        r = client.put("/api/profiles/demo/soul", json={"content": SOUL})

        assert r.status_code == 200, r.text
        mode = stat.S_IMODE(soul.stat().st_mode)
        assert mode == 0o644, f"first save created SOUL.md as {oct(mode)}"


class TestIsolatedProfileSoulScope:
    """An isolated (``--isolated``) dashboard scoped to one named profile must
    not read or write another profile's SOUL.md (#91330).

    The unified machine dashboard is intentionally a machine-wide management
    surface (cross-profile access is by design). But a server launched with
    ``--isolated`` from a named profile runs scoped to that profile, and
    letting it rewrite another profile's persona is a prompt-injection vector.
    """

    @pytest.fixture()
    def isolated_home(self, tmp_path, monkeypatch):
        """A hermes root with two real profiles; server scoped to ``alice``."""
        monkeypatch.setenv("HOME", str(tmp_path))
        profiles_root = tmp_path / ".hermes" / "profiles"
        for p in ("alice", "bob"):
            (profiles_root / p).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(profiles_root / "alice"))
        return profiles_root

    @contextlib.contextmanager
    def _client(self, monkeypatch, *, isolated: bool):
        """A TestClient with the shared app scoped as isolated or not.

        ``app.state.isolated`` is a process-global; snapshot it and restore it
        on exit so an isolated test can't leak into a later non-isolated test
        in the same process (its only default is when the attribute is never
        set at all).
        """
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "soul-test-token")
        from hermes_cli import web_server

        prev = getattr(web_server.app.state, "isolated", None)
        had = hasattr(web_server.app.state, "isolated")
        web_server.app.state.isolated = isolated
        c = TestClient(web_server.app, raise_server_exceptions=False)
        c.headers["Authorization"] = "Bearer soul-test-token"
        try:
            with c:
                yield c
        finally:
            if had:
                web_server.app.state.isolated = prev
            else:
                delattr(web_server.app.state, "isolated")

    def test_cross_profile_soul_write_refused(self, isolated_home, monkeypatch):
        bob = isolated_home / "bob"
        (bob / "SOUL.md").write_text("# Bob's persona\n", encoding="utf-8")
        before = (bob / "SOUL.md").read_text(encoding="utf-8")

        with self._client(monkeypatch, isolated=True) as c:
            r = c.put("/api/profiles/bob/soul", json={"content": "# Pwned\n"})

        assert r.status_code == 403, r.text
        # Bob's persona must be untouched.
        assert (bob / "SOUL.md").read_text(encoding="utf-8") == before

    def test_cross_profile_soul_read_refused(self, isolated_home, monkeypatch):
        (isolated_home / "bob" / "SOUL.md").write_text("# Bob\n", encoding="utf-8")

        with self._client(monkeypatch, isolated=True) as c:
            r = c.get("/api/profiles/bob/soul")

        assert r.status_code == 403, r.text

    def test_isolated_default_profile_blocks_named_profiles(
        self, tmp_path, monkeypatch
    ):
        """An isolated server scoped to the DEFAULT profile must still refuse
        cross-profile persona access (#91330 P1)."""
        monkeypatch.setenv("HOME", str(tmp_path))
        profiles_root = tmp_path / ".hermes" / "profiles"
        bob = profiles_root / "bob"
        bob.mkdir(parents=True, exist_ok=True)
        (bob / "SOUL.md").write_text("# Bob\n", encoding="utf-8")
        # Isolated server running as the default (root) profile.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        with self._client(monkeypatch, isolated=True) as c:
            r = c.put("/api/profiles/bob/soul", json={"content": "# nope\n"})

        assert r.status_code == 403, r.text
        assert (bob / "SOUL.md").read_text(encoding="utf-8") == "# Bob\n"

    def test_same_profile_soul_still_works(self, isolated_home, monkeypatch):
        with self._client(monkeypatch, isolated=True) as c:
            put = c.put("/api/profiles/alice/soul", json={"content": SOUL})
            get = c.get("/api/profiles/alice/soul")

        assert put.status_code == 200, put.text
        assert get.status_code == 200, get.text
        assert get.json()["content"] == SOUL

    def test_machine_dashboard_keeps_cross_profile_access(self, tmp_path, monkeypatch):
        """Control: the default machine dashboard is intentionally machine-wide
        and must not start refusing cross-profile SOUL edits."""
        monkeypatch.setenv("HOME", str(tmp_path))
        profiles_root = tmp_path / ".hermes" / "profiles"
        bob = profiles_root / "bob"
        bob.mkdir(parents=True, exist_ok=True)
        # Machine dashboard runs from the root home, NOT isolated.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        with self._client(monkeypatch, isolated=False) as c:
            r = c.put("/api/profiles/bob/soul", json={"content": "# ok\n"})

        assert r.status_code == 200, r.text
        assert (bob / "SOUL.md").read_text(encoding="utf-8") == "# ok\n"

    def test_cross_profile_endpoints_gated_in_isolated(self, isolated_home, monkeypatch):
        """DELETE/rename/model/export for another profile are refused from an
        isolated server — the full isolation boundary, not just SOUL.md."""
        (isolated_home / "bob" / "SOUL.md").write_text("# Bob\n", encoding="utf-8")

        with self._client(monkeypatch, isolated=True) as c:
            d = c.delete("/api/profiles/bob")
            x = c.post("/api/profiles/bob/export", json={})

        assert d.status_code == 403, d.text
        assert x.status_code == 403, x.text

    def test_create_with_clone_from_sibling_refused(self, isolated_home, monkeypatch):
        """Adversarial witness (#91330 review, stop-the-line bypass): an
        isolated server must not read a sibling profile through
        ``POST /api/profiles`` ``clone_from`` — cloning copies the source's
        config/.env/SOUL/skills into a new profile the client controls."""
        bob = isolated_home / "bob"
        bob.mkdir(parents=True, exist_ok=True)
        sentinel = "SECRET_SENTINEL_NEVER_COPY_ME=1\n"
        (bob / ".env").write_text(sentinel, encoding="utf-8")
        (bob / "SOUL.md").write_text("# Bob\n", encoding="utf-8")

        with self._client(monkeypatch, isolated=True) as c:
            r = c.post(
                "/api/profiles",
                json={"name": "evil", "clone_from": "bob"},
            )

        assert r.status_code == 403, r.text
        # No destination profile was created...
        assert not (isolated_home / "evil").exists()
        # ...and the sentinel never left Bob.
        found = [str(p) for p in isolated_home.rglob("*") if p.is_file() and sentinel in p.read_text(encoding="utf-8", errors="ignore")]
        assert found == [str(bob / ".env")], f"sentinel leaked to: {found}"

    def test_create_clone_all_implicit_default_refused(self, tmp_path, monkeypatch):
        """The implicit clone-all source ('default') is also authority-bearing:
        an isolated server scoped to a named profile must not full-copy the
        machine default profile without ever naming it."""
        monkeypatch.setenv("HOME", str(tmp_path))
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        profiles_root = hermes_root / "profiles"
        default_profile = hermes_root  # root HERMES_HOME *is* the default profile
        sentinel = "DEFAULT_SECRET_SENTINEL=1\n"
        (default_profile / ".env").write_text(sentinel, encoding="utf-8")
        (profiles_root / "alice").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(profiles_root / "alice"))

        with self._client(monkeypatch, isolated=True) as c:
            r = c.post("/api/profiles", json={"name": "evil", "clone_all": True})

        assert r.status_code == 403, r.text
        assert not (profiles_root / "evil").exists()

    def test_import_refused_when_isolated(self, isolated_home, monkeypatch):
        """Archive import creates a full profile directory — machine-global
        control-plane mutation, refused on an isolated server."""
        with self._client(monkeypatch, isolated=True) as c:
            r = c.post("/api/profiles/import", json={"archive": "/tmp/nope.tar.gz"})

        assert r.status_code == 403, r.text

    def test_active_switch_refused_when_isolated(self, isolated_home, monkeypatch):
        """Switching the machine-wide active profile regains authority over
        other profiles' CLI/gateway routing — refused when isolated."""
        with self._client(monkeypatch, isolated=True) as c:
            r = c.post("/api/profiles/active", json={"name": "bob"})

        assert r.status_code == 403, r.text

    def test_control_plane_works_on_machine_dashboard(self, tmp_path, monkeypatch):
        """Control: the unified machine dashboard keeps profile creation and
        active-profile switching (intentional machine-wide management)."""
        monkeypatch.setenv("HOME", str(tmp_path))
        profiles_root = tmp_path / ".hermes" / "profiles"
        profiles_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        with self._client(monkeypatch, isolated=False) as c:
            r = c.post("/api/profiles", json={"name": "fresh"})

        assert r.status_code == 200, r.text

    def test_cross_profile_endpoints_work_on_machine_dashboard(
        self, tmp_path, monkeypatch
    ):
        """Control: the unified machine dashboard keeps cross-profile
        management (delete/export) — the boundary only bites when isolated."""
        monkeypatch.setenv("HOME", str(tmp_path))
        profiles_root = tmp_path / ".hermes" / "profiles"
        (profiles_root / "bob").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        with self._client(monkeypatch, isolated=False) as c:
            d = c.delete("/api/profiles/bob")

        # Machine dashboard may delete another profile.
        assert d.status_code == 200, d.text
        assert not (profiles_root / "bob").exists()
