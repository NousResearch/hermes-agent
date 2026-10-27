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
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


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


def test_auto_decompose_suppresses_repeated_failures_and_resets_after_success(monkeypatch):
    """A permanently failing card gets a bounded number of paid attempts per cooldown."""
    import hermes_cli

    dispatcher = _dispatcher()
    attempts = []
    clock = {"now": 100.0}

    def fake_decompose(task_id, author=None):
        attempts.append(task_id)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None, reason="malformed reply")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=fake_decompose)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)
    monkeypatch.setattr(kwd.time, "monotonic", lambda: clock["now"])

    for _ in range(kwd._AUTO_DECOMPOSE_FAILURE_LIMIT + 2):
        dispatcher.auto_decompose_tick(1)

    assert attempts == ["t1"] * kwd._AUTO_DECOMPOSE_FAILURE_LIMIT

    # A successful retry after the cooldown clears the streak, so a later
    # failure is eligible for the full bounded retry budget again.
    clock["now"] += kwd._AUTO_DECOMPOSE_FAILURE_COOLDOWN_SECONDS
    fake.decompose_task = lambda task_id, author=None: SimpleNamespace(
        ok=True, fanout=False, child_ids=None, reason="",
    )
    assert dispatcher.auto_decompose_tick(1) == 1
    assert dispatcher._auto_decompose_failures == {}
