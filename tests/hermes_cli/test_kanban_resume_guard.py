"""Kanban worker transcripts must not resume as ordinary write-capable sessions (#68779).

Behaviour tests for the centralized guard (`hermes_cli/kanban_resume_guard.py`) and the three
CLI resume surfaces it protects: startup `--resume` (`_preload_resumed_session`), mid-chat
`/resume`, and the quiet one-shot resume (`oneshot._load_resume_target`). The gateway surface
is covered in `tests/tui_gateway/test_kanban_resume_guard.py`.
"""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _db(tmp_path, *, source="cli", flip_source_to=None, parent=None, end_reason=None,
        session_id="sess-plain", with_message=True):
    """The classic CLI's session store, reading the store seeded at ``tmp_path`` (its gateway
    connection is a stand-in no test reaches); ``.persistence`` is the storage interface handle."""
    from hermes_cli.gateway_session_store import GatewaySessionStore, SurfaceLabelReader
    store = storage_handle(tmp_path)
    seed_session(store, session_id, source, parent_session_id=parent)
    if flip_source_to:  # simulate a later surface flipping live ``source`` (created_source stays)
        raw_holder(store)._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET source = ? WHERE id = ?", (flip_source_to, session_id)))
    if with_message:
        seed_messages(store, session_id, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"})
    if end_reason:
        store.sessions.end(session_id, end_reason)
    db = GatewaySessionStore(SimpleNamespace(final=False))
    db._reader, db.persistence = SurfaceLabelReader(store), store
    return db


# ── is_kanban_owned_session ──────────────────────────────────────────

def test_kanban_source_row_is_kanban_owned(tmp_path):
    db = _db(tmp_path, source="kanban", session_id="worker-1")
    assert db.is_kanban_owned_session("worker-1") is True


def test_cli_source_row_is_not_kanban_owned(tmp_path):
    db = _db(tmp_path, source="cli", session_id="chat-1")
    assert db.is_kanban_owned_session("chat-1") is False


def test_created_source_survives_surface_flip(tmp_path):
    """A worker row whose live ``source`` was flipped (e.g. by a later surface) stays
    kanban-owned via immutable ``created_source``."""
    db = _db(tmp_path, source="kanban", flip_source_to="cli", session_id="flipped-1")
    assert db.is_kanban_owned_session("flipped-1") is True


def test_compression_lineage_of_kanban_worker_is_owned(tmp_path):
    """Resuming the compression child of a worker transcript is resuming the worker's
    conversation — it must classify as kanban-owned."""
    db = _db(tmp_path, source="kanban", session_id="worker-root", end_reason="compression")
    seed_session(db.persistence, "worker-tip", "cli", parent_session_id="worker-root")
    seed_messages(db.persistence, "worker-tip", {"role": "user", "content": "continued"})
    assert db.is_kanban_owned_session("worker-tip") is True


def test_branch_child_of_kanban_worker_is_not_owned(tmp_path):
    """A delegate/branch child is a DIFFERENT conversation — not swept into the guard."""
    db = _db(tmp_path, source="kanban", session_id="worker-root", end_reason="blocked")
    seed_session(db.persistence, "branch-child", "cli", parent_session_id="worker-root",
                 model_config={"_branched_from": "worker-root"})
    assert db.is_kanban_owned_session("branch-child") is False


# ── the centralized guard ────────────────────────────────────────────

def _refusal(db, sid):
    from hermes_cli.kanban_resume_guard import kanban_resume_refusal
    return kanban_resume_refusal(db, sid)


def test_guard_refuses_kanban_session_with_reason(tmp_path):
    db = _db(tmp_path, source="kanban", session_id="worker-1")
    refusal = _refusal(db, "worker-1")
    assert refusal is not None
    assert "worker-1" in refusal
    assert "Kanban" in refusal


def test_guard_allows_ordinary_session(tmp_path):
    db = _db(tmp_path, source="cli", session_id="chat-1")
    assert _refusal(db, "chat-1") is None


def test_guard_exempts_dispatcher_owned_run(tmp_path, monkeypatch):
    """A process holding HERMES_KANBAN_TASK is a dispatcher-owned run — the supervised
    continuation — and may resume (env beats no other signal)."""
    db = _db(tmp_path, source="kanban", session_id="worker-1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-42")
    assert _refusal(db, "worker-1") is None


def test_guard_fails_open_on_probe_error(tmp_path):
    db = _db(tmp_path, source="cli", session_id="chat-1")

    def boom(session_id):
        raise RuntimeError("database is locked")

    db.is_kanban_owned_session = boom
    assert _refusal(db, "chat-1") is None


# ── CLI startup --resume / -c ────────────────────────────────────────

def _make_cli(resume=None):
    from cli import HermesCLI
    import cli as _cli_mod

    _clean_config = {
        "model": {"default": "anthropic/claude-opus-4.6", "base_url": "https://openrouter.ai/api/v1",
                  "provider": "auto"},
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        return HermesCLI(resume=resume)


def test_startup_resume_refuses_kanban_session_and_starts_fresh(tmp_path):
    db = _db(tmp_path, source="kanban", session_id="worker-1")
    reopened = []
    orig_reopen = db.reopen_session
    db.reopen_session = lambda session_id: reopened.append(session_id) or orig_reopen(session_id)

    cli = _make_cli(resume="worker-1")
    cli._session_db = db
    buf = StringIO()
    cli.console.file = buf

    assert cli._preload_resumed_session() is False
    assert "worker-1" in buf.getvalue()
    assert "Kanban" in buf.getvalue()
    # The worker row must never receive this process's writes: fresh id, no resume state.
    assert cli._resumed is False
    assert cli.session_id != "worker-1"
    assert reopened == [], "the kanban worker row must not be reopened by an ordinary resume"


def test_startup_resume_ordinary_session_still_works(tmp_path):
    db = _db(tmp_path, source="cli", session_id="chat-1")
    cli = _make_cli(resume="chat-1")
    cli._session_db = db
    buf = StringIO()
    cli.console.file = buf

    assert cli._preload_resumed_session() is True
    assert len(cli.conversation_history) == 2
    assert cli.session_id == "chat-1"


# ── mid-chat /resume ─────────────────────────────────────────────────

def test_midchat_resume_refuses_kanban_target(tmp_path):
    db = _db(tmp_path, source="cli", session_id="chat-1")
    seed_session(db.persistence, "worker-1", "kanban")
    seed_messages(db.persistence, "worker-1", {"role": "user", "content": "task work"})
    cli = _make_cli(resume=None)
    cli._session_db = db
    cli.session_id = "chat-1"
    cli.conversation_history = [{"role": "user", "content": "hi"}]
    printed = []
    with patch("cli._cprint", lambda line: printed.append(line)):
        cli._handle_resume_command("/resume worker-1")

    joined = "\n".join(str(p) for p in printed)
    assert "Kanban" in joined
    # The chat stays on its own session — no switch, no history swap.
    assert cli.session_id == "chat-1"
    assert cli.conversation_history == [{"role": "user", "content": "hi"}]


# ── quiet one-shot resume ────────────────────────────────────────────

@pytest.fixture
def handle(tmp_path):
    """The storage interface handle on the store ``_db`` seeds at ``tmp_path``."""
    from persistence_assembly import open_persistence, profile_scope
    persistence = open_persistence(profile_scope(home=tmp_path))
    yield persistence
    persistence.close()


def test_oneshot_resume_refuses_kanban_session(tmp_path, handle):
    from hermes_cli.oneshot import _load_resume_target
    _db(tmp_path, source="kanban", session_id="worker-1")
    with pytest.raises(RuntimeError, match="cannot resume session worker-1"):
        _load_resume_target(handle, "worker-1")


def test_oneshot_resume_ordinary_session_still_works(tmp_path, handle):
    from hermes_cli.oneshot import _load_resume_target
    _db(tmp_path, source="cli", session_id="chat-1")
    sid, history, _meta = _load_resume_target(handle, "chat-1")
    assert sid == "chat-1"
    assert len(history) == 2
