from unittest.mock import MagicMock


def test_dashboard_dispatch_uses_health_conditioned_capacity(monkeypatch):
    from hermes_cli import kanban_db, kanban_health
    from plugins.kanban.dashboard import plugin_api

    cfg = {"kanban": {
        "max_in_progress": 6,
        "max_in_progress_per_profile": 2,
        "external_worker_health": {"capacity_multiplier": 2},
    }}
    dispatch_config = kanban_db.DispatchConfig(
        max_in_progress=6,
        max_in_progress_per_profile=2,
    )
    conn = MagicMock()
    captured = {}
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: "default")
    monkeypatch.setattr(plugin_api, "_conn", lambda board=None: conn)
    monkeypatch.setattr(plugin_api, "load_config", lambda: cfg)
    monkeypatch.setattr(
        kanban_db, "load_dispatch_config", lambda config=None: dispatch_config
    )
    monkeypatch.setattr(
        kanban_health, "resolve_capacity_limits", lambda *a, **k: (12, 4)
    )
    monkeypatch.setattr(
        kanban_db,
        "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    plugin_api.dispatch(dry_run=True, max_n=8, board="default")
    assert captured["max_in_progress"] == 12
    assert captured["max_in_progress_per_profile"] == 4


def test_dashboard_dispatch_resolver_exception_retains_configured_caps(monkeypatch):
    from hermes_cli import kanban_db, kanban_health
    from plugins.kanban.dashboard import plugin_api

    cfg = {"kanban": {
        "max_in_progress": 6,
        "max_in_progress_per_profile": 2,
    }}
    dispatch_config = kanban_db.DispatchConfig(
        max_in_progress=6,
        max_in_progress_per_profile=2,
    )
    conn = MagicMock()
    captured = {}
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: "default")
    monkeypatch.setattr(plugin_api, "_conn", lambda board=None: conn)
    monkeypatch.setattr(plugin_api, "load_config", lambda: cfg)
    monkeypatch.setattr(
        kanban_db, "load_dispatch_config", lambda config=None: dispatch_config
    )
    monkeypatch.setattr(
        kanban_health,
        "resolve_capacity_limits",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(
        kanban_db,
        "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    plugin_api.dispatch(dry_run=True, max_n=8, board="default")

    assert captured["max_in_progress"] == 6
    assert captured["max_in_progress_per_profile"] == 2
    conn.close.assert_called_once_with()