"""Behaviour contracts for the ``pre_attachment_store`` policy hook — the single
pre-store seam in front of every durable Kanban attachment ingress (in-memory
store lane, dashboard file adoption, completion/review artifact staging).

Regression class: a worker's ``kanban_complete(artifacts=[...])`` copied a file
holding a live credential into the durable attachments dir with no programmable
check in front of it (the tool-layer ``pre_tool_call`` guard never sees the
engine-side staging copy). The seam must refuse before the row exists, keep no
blob, surface the subscriber's value-free message, audit an event, and fail
closed when the policy itself breaks.

Hooks are registered by patching the plugin manager's ``_hooks`` registry
directly — the same registry ``invoke_hook``/``has_hook`` read (pattern:
``test_kanban_lifecycle_hooks.py``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_attachments as kba
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def policy():
    """Install subscriber callbacks for ``pre_attachment_store`` for one test."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def install(*callbacks):
        mgr._hooks["pre_attachment_store"] = list(callbacks)

    try:
        yield install
    finally:
        mgr._hooks = saved


def _task(conn, title: str = "attachment policy", **kw) -> str:
    return kb.create_task(conn, title=title, **kw)


def test_base_defect_no_gate_existed_in_front_of_the_staging_copy():
    """Red on base: before this seam, completion staging copied a declared
    scratch artifact into the durable attachments dir with no programmable
    check in front of it — the incident blob. Proven by the seam being absent,
    not by source text: the store module and its hook name are the contract."""
    from hermes_cli.plugins import VALID_HOOKS

    assert "pre_attachment_store" in VALID_HOOKS
    import hermes_cli.kanban_db_attachments  # the seam's existence is the contract


def _event_kinds(conn, task_id) -> list[str]:
    return [e.kind for e in kb.list_events(conn, task_id)]


def _blobs(task_dir: Path) -> list[Path]:
    return sorted(p.name for p in task_dir.iterdir()) if task_dir.exists() else []


# ---------------------------------------------------------------------------
# In-memory store lane (agent tools + CLI ride this)
# ---------------------------------------------------------------------------


def test_block_refuses_store_leaves_no_blob_no_row_but_an_event(kanban_home, policy):
    def refuse(**kw):
        return {"action": "block", "message": "needle store:key fp12 aaaaaaaaaaaa"}

    policy(refuse)

    conn = kbc.connect()
    try:
        tid = _task(conn)
        with pytest.raises(kba.AttachmentPolicyBlocked) as exc_info:
            kba.store_attachment_bytes(conn, tid, "dump.json", b"VLLM_API_KEY=secret", uploaded_by="agent")
        assert "fp12" in str(exc_info.value)
        assert "secret" not in str(exc_info.value)  # core never echoes the content
        assert kb.list_attachments(conn, tid) == []
        assert _event_kinds(conn, tid).count("attachment_policy_blocked") == 1
        assert _blobs(kb.task_attachments_dir(tid)) == []
    finally:
        conn.close()


def test_payload_shape_and_private_mode_during_callback(kanban_home, policy):
    seen: dict = {}

    def capture(**kw):
        seen.update(kw)
        seen["mode_during"] = Path(kw["path"]).stat().st_mode & 0o777
        seen["bytes_during"] = Path(kw["path"]).read_bytes()
        # implicit None == allow

    policy(capture)

    conn = kbc.connect()
    try:
        tid = _task(conn)
        att_id = kba.store_attachment_bytes(
            conn, tid, "notes.txt", b"plain bytes", content_type="text/plain",
            uploaded_by="agent", source="attach_url")
        assert att_id > 0
        assert seen["task_id"] == tid
        assert seen["source"] == "attach_url"
        assert seen["filename"] == "notes.txt"
        assert seen["content_type"] == "text/plain"
        assert seen["size"] == len(b"plain bytes")
        assert seen["uploaded_by"] == "agent"
        assert seen["path"] == str(Path(seen["path"]).resolve(strict=False)) or Path(seen["path"]).is_file()
        assert seen["mode_during"] == 0o600  # not-yet-approved bytes never sit at the default mode
        assert seen["bytes_during"] == b"plain bytes"  # the gate scans the real staged bytes
    finally:
        conn.close()


def test_in_place_redaction_lands_and_row_records_post_hook_size(kanban_home, policy):
    """A subscriber may scrub the staged file and allow — the row must carry the
    rewritten size, never a lying pre-hook byte count."""

    def redact(**kw):
        path = Path(kw["path"])
        data = path.read_bytes().replace(b"secret", b"[redacted]")
        path.write_bytes(data[: len(data) // 2])  # also shrink
        return {"action": "allow"}

    policy(redact)

    conn = kbc.connect()
    try:
        tid = _task(conn)
        blob = b"secret secret secret secret"
        att_id = kba.store_attachment_bytes(conn, tid, "log.txt", blob)
        att = kb.get_attachment(conn, att_id)
        stored = Path(att.stored_path)
        content = stored.read_bytes()
        assert b"secret" not in content
        assert att.size == len(content)
    finally:
        conn.close()


def test_raising_subscriber_fails_closed(kanban_home, policy):
    def broken(**kw):
        raise RuntimeError("oracle offline")

    policy(broken)

    conn = kbc.connect()
    try:
        tid = _task(conn)
        with pytest.raises(kba.AttachmentPolicyBlocked) as exc_info:
            kba.store_attachment_bytes(conn, tid, "x.bin", b"data")
        assert "raised RuntimeError" in str(exc_info.value)  # names the breakage, not the bytes
        assert kb.list_attachments(conn, tid) == []
        assert _blobs(kb.task_attachments_dir(tid)) == []
    finally:
        conn.close()


def test_any_block_wins_and_allow_path_restores_default_mode(kanban_home, policy):
    def allow(**kw):
        return None

    def refuse(**kw):
        return {"action": "block", "message": "second subscriber says no"}

    policy(allow, refuse)

    conn = kbc.connect()
    try:
        tid = _task(conn)
        with pytest.raises(kba.AttachmentPolicyBlocked):
            kba.store_attachment_bytes(conn, tid, "x.bin", b"data")

        policy(allow)  # only the blocking subscriber refused; alone, the store proceeds
        tid2 = _task(conn)
        att_id = kba.store_attachment_bytes(conn, tid2, "ok.bin", b"data")
        att = kb.get_attachment(conn, att_id)
        # The allowed blob ends at the same mode an un-scanned store would give it.
        umask_probe = Path(kb.task_attachments_dir(tid2)) / "umask-probe"
        umask_probe.write_bytes(b"x")
        assert Path(att.stored_path).stat().st_mode & 0o777 == umask_probe.stat().st_mode & 0o777
        umask_probe.unlink()
    finally:
        conn.close()


def test_policy_inactive_stores_exactly_as_before(kanban_home):
    """No subscriber: byte-identical legacy fast path — no event, blob written,
    row recorded with the caller's bytes."""
    conn = kbc.connect()
    try:
        tid = _task(conn)
        att_id = kba.store_attachment_bytes(conn, tid, "doc.txt", b"some bytes", uploaded_by="tester")
        att = kb.get_attachment(conn, att_id)
        assert Path(att.stored_path).read_bytes() == b"some bytes"
        assert att.size == len(b"some bytes")
        assert "attachment_policy_blocked" not in _event_kinds(conn, tid)
    finally:
        conn.close()


def test_unknown_task_refused_before_any_blob_write(kanban_home):
    conn = kbc.connect()
    try:
        with pytest.raises(ValueError, match="unknown task"):
            kba.store_attachment_bytes(conn, "t_nosuch", "x.bin", b"data")
        assert not kb.task_attachments_dir("t_nosuch").exists()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Completion/review artifact staging (kanban_complete side — the incident door)
# ---------------------------------------------------------------------------


def _scratch_task(conn, home: Path) -> tuple[str, Path]:
    workspace = home / "kanban" / "workspaces" / "w1"
    workspace.mkdir(parents=True)
    tid = kb.create_task(conn, title="scratch", assignee="worker",
                         workspace_kind="scratch", workspace_path=str(workspace))
    return tid, workspace


def test_completion_artifact_block_rolls_back_completion(kanban_home, policy):
    def refuse(**kw):
        return {"action": "block", "message": "needle store:key fp12 bbbbbbbbbbbb"}

    policy(refuse)

    conn = kbc.connect()
    try:
        tid, workspace = _scratch_task(conn, kanban_home)
        artifact = workspace / "inspect-pre.json"
        artifact.write_bytes(b'{"Env": ["VLLM_API_KEY=secret"]}')
        kb.claim_task(conn, tid)
        with pytest.raises(kb.ArtifactPreservationError, match="refused by attachment policy"):
            kb.complete_task(conn, tid, summary="done", force=True,
                             metadata={"artifacts": [str(artifact)]})
        task = kb.get_task(conn, tid)
        assert task.status == "running"  # retryable, nothing half-committed
        assert kb.list_attachments(conn, tid) == []
        assert _blobs(kb.task_attachments_dir(tid)) == []  # staged copy unlinked
    finally:
        conn.close()


def test_completion_artifact_allow_stages_normally(kanban_home, policy):
    calls: list[str] = []

    def allow(**kw):
        calls.append(kw["source"])  # implicit None == allow

    policy(allow)

    conn = kbc.connect()
    try:
        tid, workspace = _scratch_task(conn, kanban_home)
        artifact = workspace / "report.md"
        artifact.write_bytes(b"# fine")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="done", force=True,
                                metadata={"artifacts": [str(artifact)]}) is True
        assert calls == ["completion_artifact"]
        atts = kb.list_attachments(conn, tid)
        assert [a.filename for a in atts] == ["report.md"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard upload door (streamed file adoption)
# ---------------------------------------------------------------------------


def _load_plugin_router():
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_attachment_policy_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_dashboard_upload_blocked_400_no_row_no_staged_file(client, kanban_home, policy):
    def refuse(**kw):
        return {"action": "block", "message": "needle harbor:key fp12 cccccccccccc"}

    policy(refuse)

    conn = kbc.connect()
    try:
        tid = _task(conn, title="dashboard")
    finally:
        conn.close()
    resp = client.post(
        f"/api/plugins/kanban/tasks/{tid}/attachments",
        files={"file": ("dump.json", b"API_KEY=leaked", "application/json")},
    )
    assert resp.status_code == 400
    assert "fp12" in resp.json()["detail"]
    assert _blobs(kb.task_attachments_dir(tid)) == []  # .part staged, then unlinked
    conn = kbc.connect()
    try:
        assert kb.list_attachments(conn, tid) == []
        assert _event_kinds(conn, tid).count("attachment_policy_blocked") == 1
    finally:
        conn.close()


def test_dashboard_upload_allow_roundtrip_and_size_after_redaction(client, kanban_home, policy):
    def redact_shrink(**kw):
        Path(kw["path"]).write_bytes(b"redacted")  # implicit None == allow

    policy(redact_shrink)

    conn = kbc.connect()
    try:
        tid = _task(conn, title="dashboard-ok")
    finally:
        conn.close()
    resp = client.post(
        f"/api/plugins/kanban/tasks/{tid}/attachments",
        files={"file": ("a.txt", b"some original bytes", "text/plain")},
    )
    assert resp.status_code == 200
    att = resp.json()["attachment"]
    assert att["filename"] == "a.txt"
    conn = kbc.connect()
    try:
        row = kb.get_attachment(conn, att["id"])
        assert row.size == len(Path(row.stored_path).read_bytes())
    finally:
        conn.close()


def test_dashboard_upload_over_cap_413_leaves_no_staged_file(client, kanban_home, monkeypatch):
    import sys

    mod = sys.modules["hermes_dashboard_plugin_attachment_policy_test"]
    monkeypatch.setattr(mod, "KANBAN_ATTACHMENT_MAX_BYTES", 16)
    conn = kbc.connect()
    try:
        tid = _task(conn, title="dashboard-413")
    finally:
        conn.close()
    resp = client.post(
        f"/api/plugins/kanban/tasks/{tid}/attachments",
        files={"file": ("big.bin", b"x" * 64, "application/octet-stream")},
    )
    assert resp.status_code == 413
    assert _blobs(kb.task_attachments_dir(tid)) == []
