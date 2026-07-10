"""Phase 3 — close-path emit contract tests.

Per the approved directive, verify:
- every session close path emits the ``on_session_end`` plugin hook where
  applicable (TUI, ACP, and the unified agent teardown used by CLI/gateway/
  oneshot);
- a failing hook listener must NEVER block the close path;
- a closed session, once the hook fires, becomes searchable via the memory
  archive listener (consumer side);
- raw transcripts stay byte-identical (ownership rule);
- a failed refresh is retried and eventually succeeds.

These exercise the REAL close-path functions (not mocks of the emit), plus the
real archive-lifecycle listener wired to ``on_session_end``.
"""

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli.memory_index.indexer import MemoryIndex
from hermes_cli.plugins import get_plugin_manager, invoke_hook


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _spy(state):
    def _cb(**kwargs):
        state.append(kwargs)
    return _cb


def _make_session(path: Path, session_id: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"role": "session_meta", "session_id": session_id}),
        json.dumps({"role": "user", "content": text, "ts": "2026-07-09T08:00:00+00:00"}),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def hook_registry():
    """Capture/replace the plugin manager's on_session_end callbacks safely."""
    pm = get_plugin_manager()
    before = list(pm._hooks.get("on_session_end", []))
    pm._hooks["on_session_end"] = []
    yield pm
    # restore
    pm._hooks["on_session_end"] = before


# --------------------------------------------------------------------------- #
# 1. every close path emits the hook
# --------------------------------------------------------------------------- #

def test_tui_finalize_emits_hook(hook_registry):
    from tui_gateway.server import _finalize_session

    fired = []
    hook_registry._hooks["on_session_end"] = [_spy(fired)]

    class _Agent:
        session_id = "tui-s1"
        model = "tencent/hy3:free"
        platform = "tui"

        def _persist_session(self, *a, **k):
            return None

    session = {"_finalized": False, "agent": _Agent(), "history": []}
    _finalize_session(session, end_reason="tui_close")
    assert fired, "TUI _finalize_session must emit on_session_end"
    assert fired[0]["session_id"] == "tui-s1"
    assert fired[0]["interrupted"] is True


def test_acp_cleanup_emits_hook(hook_registry):
    from acp_adapter.session import SessionManager

    fired = []
    hook_registry._hooks["on_session_end"] = [_spy(fired)]

    mgr = SessionManager()
    mgr._sessions["acp-s1"] = object()  # minimal entry; cleanup only needs keys

    def _noop_delete(sid):
        return True

    mgr._delete_persisted = _noop_delete
    mgr._get_db = lambda: None
    mgr.cleanup()

    assert any(k.get("session_id") == "acp-s1" for k in fired), \
        "ACP SessionManager.cleanup must emit on_session_end"


def test_agent_shutdown_memory_provider_emits_hook(hook_registry):
    import run_agent

    fired = []
    hook_registry._hooks["on_session_end"] = [_spy(fired)]

    # Bind the unbound method to a minimal stand-in (no full AIAgent needed):
    # shutdown_memory_provider skips the provider block when _memory_manager is
    # None and still runs the plugin-hook emit.
    class _Stub:
        session_id = "agent-s1"
        model = "tencent/hy3:free"
        platform = "cli"
        _memory_manager = None
        context_compressor = None

    stub = _Stub()
    run_agent.AIAgent.shutdown_memory_provider.__get__(stub)()
    assert any(k.get("session_id") == "agent-s1" for k in fired), \
        "AIAgent.shutdown_memory_provider must emit on_session_end (covers CLI/gateway/oneshot)"
    assert fired[0]["completed"] is True and fired[0]["interrupted"] is False


# --------------------------------------------------------------------------- #
# 2. hook failure never blocks close
# --------------------------------------------------------------------------- #

def test_hook_failure_never_blocks_close(hook_registry):
    from tui_gateway.server import _finalize_session
    from acp_adapter.session import SessionManager
    import run_agent

    def _boom(**kwargs):
        raise RuntimeError("listener exploded")

    hook_registry._hooks["on_session_end"] = [_boom]

    class _Agent:
        session_id = "x"
        model = "m"
        platform = "tui"

        def _persist_session(self, *a, **k):
            return None

    # TUI
    _finalize_session({"_finalized": False, "agent": _Agent(), "history": []}, "tui_close")
    # ACP
    mgr = SessionManager()
    mgr._sessions["a1"] = object()
    mgr._delete_persisted = lambda s: True
    mgr._get_db = lambda: None
    mgr.cleanup()
    # agent teardown
    class _Stub:
        session_id = "a1"
        model = "m"
        platform = "cli"
        _memory_manager = None
        context_compressor = None

    run_agent.AIAgent.shutdown_memory_provider.__get__(_Stub())()
    # If we reached here without raising, the contract holds.


# --------------------------------------------------------------------------- #
# 3. closed session becomes searchable (consumer side) + raw untouched
# --------------------------------------------------------------------------- #

@pytest.fixture
def idx(tmp_path: Path):
    home = tmp_path / "hermes"
    (home / "sessions").mkdir(parents=True)
    return MemoryIndex(db_path=home / "memory" / "index.db", hermes_home=home)


def test_closed_session_becomes_searchable(idx, tmp_path, monkeypatch):
    from hermes_cli.memory_index import archive_lifecycle

    sp = tmp_path / "hermes" / "sessions" / "live1.jsonl"
    _make_session(sp, "live1", "the lighthouse keeps the archive")
    before = sp.read_bytes()

    # The archive-lifecycle listener builds its own MemoryIndex() from
    # HERMES_HOME at call time, so point it at the test home so its enqueue
    # lands in the same DB we assert against. Re-register explicitly so the
    # test is self-contained regardless of sibling-fixture hook state.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    archive_lifecycle.register_listener()

    # Real close path equivalent: fire the hook the way any boundary would.
    invoke_hook("on_session_end", session_id="live1", completed=True,
                interrupted=False, model="tencent/hy3:free", platform="tui")
    # Background flush is fire-and-forget; drain explicitly to assert outcome.
    idx.refresh_pending()

    hits = idx.search("lighthouse")
    assert any(r.source_file.endswith("live1.jsonl") for r in hits)
    # ownership rule: raw transcript byte-identical
    assert sp.read_bytes() == before


# --------------------------------------------------------------------------- #
# 4. failed refresh retries successfully
# --------------------------------------------------------------------------- #

def test_failed_refresh_retries(idx, tmp_path):
    good = tmp_path / "hermes" / "sessions" / "retry.jsonl"
    _make_session(good, "retry", "eventually indexed content")

    real_index = idx.index_session

    calls = {"n": 0}

    def _flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient index failure")
        return real_index(path)

    idx.index_session = _flaky  # simulate a transient failure on first attempt
    idx.enqueue(str(good))

    stats1 = idx.refresh_pending()
    # first attempt failed -> row marked failed, attempts=1
    with sqlite3.connect(str(idx.db_path)) as c:
        row = c.execute(
            "SELECT attempts, status FROM index_pending WHERE source_file=?",
            ("sessions/retry.jsonl",),
        ).fetchone()
    assert row is not None and row[0] == 1 and row[1] == "failed"
    assert stats1["failed"] == 1

    # restore healthy indexer and retry
    idx.index_session = real_index
    stats2 = idx.refresh_pending()
    assert stats2["ok"] == 1
    hits = idx.search("eventually")
    assert any(r.source_file.endswith("retry.jsonl") for r in hits)
