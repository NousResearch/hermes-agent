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

import asyncio
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
        # web_server resolves _SESSION_TOKEN once, at import. Read it back from
        # the module instead of assuming the env var above won the race — any
        # test file that imports web_server earlier in the session fixes the
        # token before this fixture runs.
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
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


class TestSoulIoIsOffTheEventLoop:
    """Neither half of the persona editor may run its I/O on the ASGI loop.

    The durability the tests above buy comes from ``atomic_write_text``, which
    fsyncs before replacing — so the save blocks for as long as the filesystem
    takes to commit. These handlers sit in the same router as the profile
    delete and describe-auto paths; the rest of that sweep is covered by
    ``tests/hermes_cli/test_web_profiles_off_loop.py``.
    """

    @staticmethod
    def _probe(seen, tag):
        """Record whether the caller's thread is running an event loop."""
        try:
            asyncio.get_running_loop()
            seen.append((tag, True))
        except RuntimeError:
            seen.append((tag, False))

    def test_get_soul_reads_off_loop(self, client, profile_dir: Path, monkeypatch):
        (profile_dir / "SOUL.md").write_text(SOUL, encoding="utf-8")
        seen: list[tuple[str, bool]] = []
        real_read_text = Path.read_text

        def probing_read_text(self, *args, **kwargs):
            if self.name == "SOUL.md":
                TestSoulIoIsOffTheEventLoop._probe(seen, "read")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", probing_read_text)

        r = client.get("/api/profiles/demo/soul")

        assert r.status_code == 200, r.text
        assert r.json()["content"] == SOUL
        assert ("read", False) in seen, (
            f"SOUL.md must be read off the event loop; proof: {seen}"
        )

    def test_put_soul_writes_off_loop(self, client, profile_dir: Path, monkeypatch):
        seen: list[tuple[str, bool]] = []
        import utils

        real_write = utils.atomic_write_text

        def probing_write(*args, **kwargs):
            TestSoulIoIsOffTheEventLoop._probe(seen, "write")
            return real_write(*args, **kwargs)

        monkeypatch.setattr(utils, "atomic_write_text", probing_write)

        r = client.put("/api/profiles/demo/soul", json={"content": SOUL})

        assert r.status_code == 200, r.text
        assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == SOUL
        assert ("write", False) in seen, (
            f"SOUL.md must be written off the event loop; proof: {seen}"
        )

    def test_missing_soul_is_still_reported_absent(self, client, profile_dir: Path):
        """The offloaded reader must keep distinguishing "no file" from
        "empty file" — the whole point of the durability tests above."""
        assert not (profile_dir / "SOUL.md").exists()

        r = client.get("/api/profiles/demo/soul")

        assert r.status_code == 200, r.text
        assert r.json() == {"content": "", "exists": False}

    def test_empty_soul_is_still_reported_present(self, client, profile_dir: Path):
        (profile_dir / "SOUL.md").write_text("", encoding="utf-8")

        r = client.get("/api/profiles/demo/soul")

        assert r.status_code == 200, r.text
        assert r.json() == {"content": "", "exists": True}
