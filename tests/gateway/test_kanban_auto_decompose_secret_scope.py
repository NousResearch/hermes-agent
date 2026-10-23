"""Auto-decompose tick under multiplex.

Regression for #107955 / #57837: the tick runs off-turn in a fresh Context, so
``get_secret`` fails closed unless the tick installs the launch profile's scope.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from agent import secret_scope as ss
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(
        SimpleNamespace(DEFAULT_BOARD="default", auto_decompose_retry_due=lambda conn, tid: True),
        settings,
    )


def test_auto_decompose_tick_reads_launch_profile_secrets_under_multiplex(monkeypatch, tmp_path):
    import hermes_cli

    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=launch-profile-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])

    seen = {}

    def fake_decompose(task_id, author=None):
        seen["value"] = ss.get_secret("ANTHROPIC_API_KEY")
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    ss.set_multiplex_active(True)
    try:
        # Same hop the gateway uses: fresh Context, no inherited per-turn scope.
        decomposed = asyncio.run(_to_thread_process_service(_dispatcher().auto_decompose_tick, 5))
    finally:
        ss.set_multiplex_active(False)

    assert decomposed == 1
    assert seen["value"] == "launch-profile-key"
    assert ss.current_secret_scope() is None


def test_auto_decompose_tick_persists_failures_and_skips_backoff(monkeypatch):
    import hermes_cli

    state = {"due": True, "decompose_calls": 0, "failures": []}

    class _Connection:
        def close(self):
            pass

    def record_failure(conn, task_id, *, reason):
        state["failures"].append((task_id, reason))
        return {"attempt": 1, "limit": 3, "reason": reason, "blocked": False, "retry_at": 400}

    kb = SimpleNamespace(
        DEFAULT_BOARD="default",
        auto_decompose_retry_due=lambda conn, task_id: state["due"],
        record_auto_decompose_failure=record_failure,
    )
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    dispatcher = kwd._KanbanDispatcher(kb, settings)
    monkeypatch.setattr(kwd, "_board_slugs", lambda unused: ["default"])
    monkeypatch.setattr(kwd, "_kbc", lambda: SimpleNamespace(connect=lambda board: _Connection()))

    def decompose(task_id, author=None):
        state["decompose_calls"] += 1
        return SimpleNamespace(ok=False, reason="provider unavailable")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    assert dispatcher.auto_decompose_tick(1) == 0
    assert state["failures"] == [("t1", "provider unavailable")]

    state["due"] = False
    assert dispatcher.auto_decompose_tick(1) == 0
    assert state["decompose_calls"] == 1
