"""Attempt cap for auto-decompose on a failing triage card (issue #118603).

Without a cap, ``auto_decompose_tick`` retries a card whose decomposition
fails on *every* dispatcher tick, forever — burning dispatcher budget and
(reported) re-billing one aux LLM call per tick with no cost attribution.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_failing_triage_card_is_not_retried_every_tick_forever(monkeypatch):
    """A card that fails 8 straight ticks must be billed at most 3 times."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    bills: list[str] = []

    def fake_decompose(task_id, author=None):
        bills.append(task_id)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None,
                               reason="LLM error: InternalServerError")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t_doomed"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(8):
        dispatcher.auto_decompose_tick(10)

    assert len(bills) <= 3, (
        f"doomed card billed {len(bills)} times over 8 dispatcher ticks — "
        "no attempt cap (issue #118603)"
    )
