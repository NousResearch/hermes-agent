"""F5: enforce board.json's ``auto_decompose_enabled`` in the gateway's
auto-decompose tick loop (``_KanbanDispatcher.auto_decompose_tick``).

Narrowing-only, default True: a board with no ``auto_decompose_enabled`` key
in board.json (old boards, boards created before this field existed) must
keep auto-decomposing exactly as before. Only an explicit ``False`` skips
that board's triage sweep for the tick it's read on.
"""

from __future__ import annotations

from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher(monkeypatch, board_meta: dict):
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    fake_kb = SimpleNamespace(
        DEFAULT_BOARD="default",
        read_board_metadata=lambda board=None: board_meta.get(board, {}),
    )
    return kwd._KanbanDispatcher(fake_kb, settings)


def _fake_decomp_module(seen_boards: list):
    def fake_decompose_task(task_id, author=None):
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    def make_list_triage_ids(board_slug):
        def _list():
            seen_boards.append(board_slug)
            return [f"{board_slug}-t1"]
        return _list

    return fake_decompose_task, make_list_triage_ids


def test_disabled_board_skips_auto_decompose(monkeypatch):
    """A board with auto_decompose_enabled=False must not be auto-decomposed;
    a board without the key (default True) must still be swept."""
    import sys
    import hermes_cli

    board_meta = {
        "blocked": {"auto_decompose_enabled": False},
        "open": {},  # no key at all -> default True
    }
    dispatcher = _dispatcher(monkeypatch, board_meta)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["blocked", "open"])

    seen_boards: list = []
    fake_decompose_task, make_list_triage_ids = _fake_decomp_module(seen_boards)

    def fake_list_triage_ids():
        import os
        slug = os.environ.get("HERMES_KANBAN_BOARD")
        return make_list_triage_ids(slug)()

    fake = SimpleNamespace(list_triage_ids=fake_list_triage_ids, decompose_task=fake_decompose_task)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    decomposed = dispatcher.auto_decompose_tick(5)

    assert "blocked" not in seen_boards
    assert "open" in seen_boards
    assert decomposed == 1


def test_missing_field_keeps_auto_decompose_enabled_by_default(monkeypatch):
    import sys
    import hermes_cli

    board_meta = {"legacy": {}}  # old board.json, no auto_decompose_enabled key
    dispatcher = _dispatcher(monkeypatch, board_meta)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["legacy"])

    seen_boards: list = []
    fake_decompose_task, make_list_triage_ids = _fake_decomp_module(seen_boards)

    def fake_list_triage_ids():
        import os
        slug = os.environ.get("HERMES_KANBAN_BOARD")
        return make_list_triage_ids(slug)()

    fake = SimpleNamespace(list_triage_ids=fake_list_triage_ids, decompose_task=fake_decompose_task)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    decomposed = dispatcher.auto_decompose_tick(5)

    assert seen_boards == ["legacy"]
    assert decomposed == 1


def test_flag_flipped_on_reenables_on_next_tick(monkeypatch):
    """The flag is read fresh from board.json every tick, never cached."""
    import sys
    import hermes_cli

    board_meta = {"toggle": {"auto_decompose_enabled": False}}
    dispatcher = _dispatcher(monkeypatch, board_meta)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["toggle"])

    seen_boards: list = []
    fake_decompose_task, make_list_triage_ids = _fake_decomp_module(seen_boards)

    def fake_list_triage_ids():
        import os
        slug = os.environ.get("HERMES_KANBAN_BOARD")
        return make_list_triage_ids(slug)()

    fake = SimpleNamespace(list_triage_ids=fake_list_triage_ids, decompose_task=fake_decompose_task)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    decomposed_tick1 = dispatcher.auto_decompose_tick(5)
    assert decomposed_tick1 == 0
    assert seen_boards == []

    # Flip on: next tick must sweep it, reading board.json fresh (not cached).
    board_meta["toggle"]["auto_decompose_enabled"] = True
    decomposed_tick2 = dispatcher.auto_decompose_tick(5)
    assert decomposed_tick2 == 1
    assert seen_boards == ["toggle"]
