from __future__ import annotations


def test_dashboard_dispatch_passes_selected_task_ids_to_dispatch_once(monkeypatch):
    from plugins.kanban.dashboard import plugin_api
    from hermes_cli import kanban_db

    captured = {}

    class FakeConn:
        def close(self):
            captured['closed'] = True

    monkeypatch.setattr(plugin_api, '_resolve_board', lambda board: board)
    monkeypatch.setattr(plugin_api, '_conn', lambda board=None: FakeConn())

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult(spawned=[('t_selected_1', 'alpha', '')])

    monkeypatch.setattr(plugin_api.kanban_db, 'dispatch_once', fake_dispatch_once)

    result = plugin_api.dispatch(
        payload=plugin_api.DispatchBody(taskIds=['t_selected_1', 't_selected_2']),
        dry_run=True,
        max_n=8,
        board='north-star',
    )

    assert captured['selected_task_ids'] == ['t_selected_1', 't_selected_2']
    assert captured['dry_run'] is True
    assert captured['max_spawn'] == 8
    assert captured['board'] == 'north-star'
    assert captured['closed'] is True
    assert result['selectedOnly'] is True
    assert result['selectedTaskIds'] == ['t_selected_1', 't_selected_2']


def test_dashboard_dispatch_without_selection_preserves_generic_nudge(monkeypatch):
    from plugins.kanban.dashboard import plugin_api
    from hermes_cli import kanban_db

    captured = {}

    class FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(plugin_api, '_resolve_board', lambda board: board)
    monkeypatch.setattr(plugin_api, '_conn', lambda board=None: FakeConn())

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(plugin_api.kanban_db, 'dispatch_once', fake_dispatch_once)

    result = plugin_api.dispatch(payload=None, dry_run=False, max_n=8, board='north-star')

    assert captured['selected_task_ids'] is None
    assert result['selectedOnly'] is False
    assert result['selectedTaskIds'] == []
