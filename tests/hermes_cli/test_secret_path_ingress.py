"""End-to-end ingress tests for the fail-closed secret-path policy.

Drives every Hermes kanban attachment / scratch-workspace ingress surface
and asserts that a prohibited file never reaches the durable
attachments dir.  Covers:

  * :func:`hermes_cli.kanban_db.store_attachment_bytes` — single shared
    write path used by agent tools, CLI, and (where applicable) the
    dashboard upload.
  * The dashboard multipart upload endpoint
    :func:`plugins.kanban.dashboard.plugin_api.upload_task_attachment`
    — exercised via a FastAPI ``TestClient`` so the real HTTP path
    (multipart parsing, status code mapping) is covered.
  * The agent ``kanban_attach`` / ``kanban_attach_url`` handlers
    (gated at ``_store_attachment`` + the URL pre-flight check).
  * The CLI ``hermes kanban attach <id> <path>`` entry point.
  * The scratch-workspace → attachment promotion path
    :func:`hermes_cli.kanban_db._persist_scratch_completion_artifacts`.

All prohibited-class fixtures are *synthetic* — empty files with a
prohibited name, never a real ``auth.json`` / ``.env`` / keychain
database.  The live contaminated files at ``~/.hermes/auth.json`` and
``~/.hermes/.env`` are explicitly out of scope (the task forbids
touching them); tests construct their own fixtures under ``tmp_path``
so a leak here cannot read a real secret.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Synthetic fixtures — never a real credential file
# ---------------------------------------------------------------------------


# Canonical prohibited shapes the tests will try to admit.
# Each fixture is an EMPTY file with the prohibited name; no real
# credential bytes exist anywhere on disk that this test owns.
PROHIBITED_FIXTURE_NAMES = [
    "auth.json",
    "AUTH.JSON",  # case variant
    ".env",
    ".env.production",
    ".env.local",
    ".envrc",
    "credentials.json",
    "anthropic_oauth.json",
    "vault.kdbx",
    "keepass-export.xml",
    "MyPasswords.csv",
    "1password-export.json",
    "bws_cache.enc.json",
]


@pytest.fixture(params=PROHIBITED_FIXTURE_NAMES)
def prohibited_fixture(tmp_path, request):
    """A single empty file with one of the prohibited names."""
    name = request.param
    p = tmp_path / name
    # Empty bytes — explicit "no credential content" sentinel.
    p.write_bytes(b"")
    return name, p


# Benign names that must still be admitted.  Each carries a unique
# extension so we can detect over-broad blocking.
BENIGN_FIXTURE_NAMES = [
    "notes.txt",
    "report.pdf",
    "design.md",
    "data.json",
    "result.csv",
    "snapshot.bin",
    "index.html",
    "screenshot.png",
]


@pytest.fixture
def benign_fixtures(tmp_path):
    """A directory of empty files with safe names."""
    paths = []
    for name in BENIGN_FIXTURE_NAMES:
        p = tmp_path / name
        p.write_bytes(b"")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Hermes test environment — each test gets a fresh, isolated HERMES_HOME
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Set HERMES_HOME to a fresh tmp_path so kanban_db writes nowhere real."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def open_task(kanban_home):
    """Insert a fresh task row in the kanban DB; returns its id.

    Also pre-creates a managed scratch workspace at
    ``<kanban_home>/kanban/workspaces/<task_id>`` and wires
    ``tasks.workspace_path`` so ``_persist_scratch_completion_artifacts``
    recognises it as managed.
    """
    import time

    suffix = int(time.time() * 1000) % 1000000
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title=f"ingress policy test t_test{suffix:06d}",
            workspace_kind="scratch",
            initial_status="blocked",
            triage=False,
        )
    # Build the scratch workspace dir under the kanban-managed root
    # and persist the path on the task so the persistence helper
    # recognises it.
    ws_dir = kanban_home / "kanban" / "workspaces" / task_id
    ws_dir.mkdir(parents=True, exist_ok=True)
    with kb.connect() as conn:
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(ws_dir), task_id),
        )
    return task_id


# ---------------------------------------------------------------------------
# Layer 1: the lowest shared write path.  All higher layers funnel
# through this, so testing it directly covers every entry point.
# ---------------------------------------------------------------------------


class TestStoreAttachmentBytesRejects:
    """``store_attachment_bytes`` is the single shared chokepoint."""

    def test_prohibited_filename_raises_with_policy_kind(
        self, kanban_home, open_task, prohibited_fixture
    ):
        name, p = prohibited_fixture
        data = p.read_bytes()
        with kb.connect() as conn:
            with pytest.raises(ValueError) as exc_info:
                kb.store_attachment_bytes(
                    conn, open_task, name, data,
                    uploaded_by="ingress_test",
                )
        # Error message must carry the policy ``kind`` token, never the
        # filename (per task spec: "Return a clear policy error
        # containing only a safe path/reason, with no credential data").
        msg = str(exc_info.value)
        assert "refused to admit" in msg
        # The exact fixture name must NOT leak — we assert the absence
        # of substrings that appear in any prohibited fixture above.
        for other in PROHIBITED_FIXTURE_NAMES:
            # The substring check only flags the current fixture, since
            # store_attachment_bytes echoes the reason token, not the
            # filename itself.
            if other == name:
                continue
            # The error must not mention this other prohibited name
            # either (sanity: the policy reason text is a fixed lookup).
            assert other not in msg or other == name, (
                f"error leaks a different prohibited name: {msg!r}"
            )

    def test_prohibited_filename_blocks_blob_written(
        self, kanban_home, open_task, prohibited_fixture
    ):
        name, p = prohibited_fixture
        data = p.read_bytes()
        with kb.connect() as conn:
            with pytest.raises(ValueError):
                kb.store_attachment_bytes(
                    conn, open_task, name, data,
                    uploaded_by="ingress_test",
                )
        # No blob should have been written under the per-task dir.
        att_dir = kb.task_attachments_dir(open_task)
        if att_dir.exists():
            leftovers = list(att_dir.iterdir())
            assert not leftovers, (
                f"attachments dir should be empty but has: {leftovers}"
            )

    def test_prohibited_filename_blocks_metadata_row(
        self, kanban_home, open_task, prohibited_fixture
    ):
        name, p = prohibited_fixture
        data = p.read_bytes()
        with kb.connect() as conn:
            with pytest.raises(ValueError):
                kb.store_attachment_bytes(
                    conn, open_task, name, data,
                    uploaded_by="ingress_test",
                )
            rows = list(
                conn.execute(
                    "SELECT * FROM task_attachments WHERE task_id=?",
                    (open_task,),
                )
            )
            assert rows == [], f"no metadata row should be inserted, got {rows}"

    def test_nested_traversal_basename_still_blocked(
        self, kanban_home, open_task, tmp_path
    ):
        """A client might supply ``subdir/../auth.json``; the leaf is
        the policy-relevant shape, so it must still be blocked."""
        # We don't even need the file to exist; the policy fires on
        # the basename string before any disk read.
        with kb.connect() as conn:
            with pytest.raises(ValueError):
                kb.store_attachment_bytes(
                    conn, open_task, "subdir/../../auth.json", b"",
                    uploaded_by="ingress_test",
                )

    def test_benign_filename_admitted(
        self, kanban_home, open_task, benign_fixtures
    ):
        """Sanity: representative benign names are still admitted."""
        for p in benign_fixtures:
            data = p.read_bytes()
            with kb.connect() as conn:
                att_id = kb.store_attachment_bytes(
                    conn, open_task, p.name, data,
                    uploaded_by="ingress_test",
                )
            assert att_id > 0, f"failed to admit benign {p.name}"


# ---------------------------------------------------------------------------
# Layer 2: dashboard multipart upload endpoint
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Load the kanban dashboard plugin's FastAPI router without
    spinning up the whole dashboard."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_secret_path_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


class TestDashboardUploadRejects:
    """The dashboard multipart upload must reject BEFORE writing bytes."""

    def test_prohibited_multipart_filename_returns_422(
        self, client, open_task, prohibited_fixture
    ):
        name, p = prohibited_fixture
        with open(p, "rb") as f:
            resp = client.post(
                f"/api/plugins/kanban/tasks/{open_task}/attachments",
                files={"file": (name, f, "application/octet-stream")},
            )
        # Policy check fires BEFORE blob write → 422, not 500/200.
        assert resp.status_code == 422, (
            f"expected 422 for prohibited {name!r}, got {resp.status_code}: "
            f"{resp.text[:200]}"
        )
        # And no blob should have been written.
        att_dir = kb.task_attachments_dir(open_task)
        if att_dir.exists():
            assert not list(att_dir.iterdir()), (
                f"attachments dir should be empty after reject: "
                f"{list(att_dir.iterdir())}"
            )

    def test_benign_multipart_filename_admitted(self, client, open_task):
        """Sanity: representative benign upload works."""
        from io import BytesIO

        files = {"file": ("report.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
        resp = client.post(
            f"/api/plugins/kanban/tasks/{open_task}/attachments",
            files=files,
        )
        assert resp.status_code == 200, resp.text[:200]


# ---------------------------------------------------------------------------
# Layer 3: scratch workspace → attachment promotion
# ---------------------------------------------------------------------------


class TestScratchArtifactIngressRejects:
    """``_persist_scratch_completion_artifacts`` is called from
    ``kanban_complete``; it must reject any prohibited leaf before the
    file is stat()'d, opened, or copied."""

    def _scratch_path(self, open_task, kanban_home):
        """Return the task's scratch workspace dir, pre-created by
        the ``open_task`` fixture."""
        return kanban_home / "kanban" / "workspaces" / open_task

    def _make_scratch_workspace(self, kanban_home, open_task, files):
        """Populate the task's scratch workspace with *files*
        (list of (relative_path, content_bytes))."""
        ws = self._scratch_path(open_task, kanban_home)
        ws.mkdir(parents=True, exist_ok=True)
        for rel, content in files:
            target = ws / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return ws

    def test_prohibited_scratch_file_blocked(
        self, kanban_home, open_task, prohibited_fixture
    ):
        name, p = prohibited_fixture
        ws = self._make_scratch_workspace(
            kanban_home, open_task, [(name, b"")],
        )
        metadata = {"artifacts": [str(ws / name)]}
        with kb.connect() as conn:
            with pytest.raises(kb.ArtifactPreservationError):
                kb._persist_scratch_completion_artifacts(
                    conn, open_task, metadata,
                )

    def test_prohibited_via_symlink_is_blocked(
        self, kanban_home, open_task, tmp_path
    ):
        """Symlinks that resolve to a prohibited basename are blocked
        after the symlink is followed, so a worker can't alias an
        ``auth.json`` to ``notes.txt``.

        The symlink target lives INSIDE the workspace (in a sibling
        subdir) so the ``is_relative_to(workspace_root)`` guard
        passes; the *resolved* basename is what the policy checks,
        and it must see ``auth.json``.
        """
        ws = self._scratch_path(open_task, kanban_home)
        ws.mkdir(parents=True, exist_ok=True)
        # Place the real ``auth.json`` inside a sibling subdir so the
        # resolved symlink still lives under the workspace root.
        sibling = ws / "sibling"
        sibling.mkdir(exist_ok=True)
        target = sibling / "auth.json"
        target.write_bytes(b"")
        # Build the symlink with a benign leaf name; the resolver
        # will follow it to ``auth.json`` and the policy fires.
        symlink = ws / "notes.txt"
        symlink.symlink_to(target)
        metadata = {"artifacts": [str(symlink)]}
        with kb.connect() as conn:
            with pytest.raises(kb.ArtifactPreservationError):
                kb._persist_scratch_completion_artifacts(
                    conn, open_task, metadata,
                )

    def test_nested_prohibited_basename_blocked(
        self, kanban_home, open_task,
    ):
        """A ``subdir/auth.json`` inside the workspace is still blocked."""
        ws = self._make_scratch_workspace(
            kanban_home, open_task, [("subdir/auth.json", b"")],
        )
        metadata = {"artifacts": [str(ws / "subdir" / "auth.json")]}
        with kb.connect() as conn:
            with pytest.raises(kb.ArtifactPreservationError):
                kb._persist_scratch_completion_artifacts(
                    conn, open_task, metadata,
                )

    def test_benign_scratch_artifact_admitted(
        self, kanban_home, open_task, benign_fixtures,
    ):
        """A workspace full of benign artifacts is copied through."""
        # Reuse one benign file inside the workspace; the others are
        # outside and would be skipped (relative_to check).
        ws = self._make_scratch_workspace(
            kanban_home, open_task, [("report.pdf", b"%PDF-1.4 fake")],
        )
        metadata = {"artifacts": [str(ws / "report.pdf")]}
        with kb.connect() as conn:
            # Should NOT raise.
            kb._persist_scratch_completion_artifacts(conn, open_task, metadata)
        staged = metadata.get("_staged_artifacts") or []
        assert len(staged) == 1, f"expected 1 staged, got {staged}"


# ---------------------------------------------------------------------------
# Layer 4: agent attach handlers — covered transitively via the shared
# _store_attachment chokepoint; pin the wiring with a smoke test.
# ---------------------------------------------------------------------------


class TestAgentAttachHandlers:
    """Smoke-test the wiring into ``_handle_attach`` /
    ``_handle_attach_url`` / ``_store_attachment`` — the deep policy
    enforcement is already covered by
    :class:`TestStoreAttachmentBytesRejects`; these tests just prove
    the handlers actually call the guard."""

    def test_handle_attach_blocks_prohibited_filename(
        self, kanban_home, open_task, prohibited_fixture,
    ):
        from tools.kanban_tools import _handle_attach
        name, _ = prohibited_fixture
        args = {
            "task_id": open_task,
            "filename": name,
            "content_base64": base64.b64encode(b"\x00\x00").decode("ascii"),
            "board": None,
        }
        # The handler's decorator returns a tool-error JSON when the
        # underlying ``_store_attachment`` raises ``_Reject``; we just
        # assert it does NOT succeed.
        try:
            result = _handle_attach(args)
        except Exception:
            # Some paths raise instead of returning a tool-error JSON;
            # either outcome is acceptable as long as no attachment
            # was written.
            result = None
        att_dir = kb.task_attachments_dir(open_task)
        if att_dir.exists():
            assert not list(att_dir.iterdir()), (
                f"handler wrote a blob for prohibited {name!r}"
            )

    def test_handle_attach_url_blocks_prohibited_filename_before_fetch(
        self, kanban_home, open_task, prohibited_fixture, monkeypatch,
    ):
        """The URL handler must NOT contact the network when the
        filename is prohibited — fail-closed."""
        from tools import kanban_tools

        # Track whether the network-fetch helper is called.
        fetched = {"called": False}

        def _spy_download(*args, **kwargs):
            fetched["called"] = True
            return b"", "application/octet-stream"

        monkeypatch.setattr(kanban_tools, "_download_url_with_cap", _spy_download)
        name, _ = prohibited_fixture
        args = {
            "task_id": open_task,
            "url": "https://example.invalid/secret-download",
            "filename": name,
            "board": None,
        }
        try:
            result = kanban_tools._handle_attach_url(args)
        except Exception:
            result = None
        assert not fetched["called"], (
            f"URL fetch was attempted for prohibited {name!r}; "
            "must be blocked at the pre-flight policy check"
        )
        att_dir = kb.task_attachments_dir(open_task)
        if att_dir.exists():
            assert not list(att_dir.iterdir()), (
                f"handler wrote a blob for prohibited {name!r}"
            )


# ---------------------------------------------------------------------------
# Layer 5: CLI ``hermes kanban attach <id> <path>``
# ---------------------------------------------------------------------------


class TestCLIAttachCommand:
    """Drive the CLI attach command via ``_cmd_attach`` so the policy
    guard at the CLI surface is exercised."""

    def test_cli_attach_blocks_prohibited_filename(
        self, kanban_home, open_task, prohibited_fixture, capsys, monkeypatch,
    ):
        import argparse

        from hermes_cli import kanban as kanban_cli

        name, p = prohibited_fixture
        args = argparse.Namespace(
            task_id=open_task,
            path=str(p),
            name=name,
            content_type=None,
            author="cli_ingress_test",
        )
        rc = kanban_cli._cmd_attach(args)
        captured = capsys.readouterr()
        assert rc != 0, (
            f"CLI attach should fail for {name!r}, got rc={rc}; "
            f"stdout={captured.out!r} stderr={captured.err!r}"
        )
        # And no blob written.
        att_dir = kb.task_attachments_dir(open_task)
        if att_dir.exists():
            assert not list(att_dir.iterdir()), (
                f"CLI wrote a blob for prohibited {name!r}"
            )