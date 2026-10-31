"""Both maintenance entrypoints consume the same opt-in automation retention settings."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("cfg", [{}, {"auto_prune": False}, {"source_retention_days": {}},
                                {"source_retention_days": {"telegram": 1}},
                                {"source_retention_days": {"cron": 0}}])
def test_disabled_settings_do_not_open_gateway_maintenance(cfg, monkeypatch):
    from gateway.run import _housekeeping_state_db_maintenance
    from hermes_cli.session_retention_config import auto_prune_kwargs

    acquire = MagicMock()
    monkeypatch.setattr("hermes_state_registry.acquire", acquire)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"sessions": cfg})
    assert auto_prune_kwargs(cfg) is None
    _housekeeping_state_db_maintenance()
    acquire.assert_not_called()


@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_real_config_loader_forwards_source_only_kwargs(tmp_path, monkeypatch, surface):
    from hermes_cli.config import load_config

    (tmp_path / "config.yaml").write_text(
        "sessions:\n  auto_prune: false\n  source_retention_days: {cron: 30, subagent: 14}\n"
        "  min_vacuum_interval_days: 17\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    assert load_config()["sessions"]["source_retention_days"] == {"cron": 30, "subagent": 14}
    db = MagicMock()
    db.get_meta.return_value = "already-done"
    if surface == "cli":
        from hermes_cli.cli_auto_maintenance import _run_state_db_auto_maintenance
        _run_state_db_auto_maintenance(db)
    else:
        from gateway.run import _housekeeping_state_db_maintenance
        monkeypatch.setattr("gateway.run.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("hermes_state_registry.acquire", lambda: db)
        release = MagicMock()
        monkeypatch.setattr("hermes_state_registry.release_or_close", release)
        _housekeeping_state_db_maintenance((tmp_path, tmp_path / "transcripts"))
        release.assert_called_once_with(db)
    db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=90, min_interval_hours=24, min_vacuum_interval_days=17, vacuum=True,
        global_prune=False, source_retention_days={"cron": 30, "subagent": 14},
        sessions_dir=tmp_path / ("sessions" if surface == "cli" else "transcripts"))


@pytest.mark.parametrize("raw", [None, [], "banana", {"cron": -1}, {"cron": "bad"},
                                {"cron": float("nan")}, {"cron": float("inf")}, {"cron": True}])
def test_invalid_retention_is_off(raw):
    from hermes_cli.session_retention_config import auto_prune_kwargs
    assert auto_prune_kwargs({"source_retention_days": raw}) is None


def test_source_map_does_not_enable_global_prune_or_change_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.session_retention_config import auto_prune_kwargs
    assert DEFAULT_CONFIG["sessions"]["source_retention_days"] == {}
    assert auto_prune_kwargs(DEFAULT_CONFIG["sessions"]) is None
    kwargs = auto_prune_kwargs({"auto_prune": True, "source_retention_days": {"cron": 30}})
    assert kwargs["global_prune"] is True
    assert kwargs["source_retention_days"] == {"cron": 30}


@pytest.mark.parametrize("global_prune", [False, True])
def test_bad_global_retention_cannot_stop_gateway_archive(monkeypatch, global_prune):
    from gateway.run import _housekeeping_state_db_maintenance

    cfg = {"auto_prune": global_prune, "retention_days": "90d", "auto_archive": True,
           "source_retention_days": {"cron": 30}}
    db = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"sessions": cfg})
    monkeypatch.setattr("hermes_state_registry.acquire", lambda: db)
    monkeypatch.setattr("hermes_state_registry.release_or_close", lambda _: None)
    if global_prune:
        with pytest.raises(ValueError):
            _housekeeping_state_db_maintenance()
    else:
        _housekeeping_state_db_maintenance()
        db.maybe_auto_prune_and_vacuum.assert_called_once()
        assert db.maybe_auto_prune_and_vacuum.call_args.kwargs["retention_days"] == 90
        assert db.maybe_auto_prune_and_vacuum.call_args.kwargs["global_prune"] is False
    db.maybe_auto_archive.assert_called_once()


def test_builder_never_warns_about_rejected_sources(caplog):
    from hermes_cli.session_retention_config import auto_prune_kwargs

    assert auto_prune_kwargs({"source_retention_days": {"telegram": 30}}) is None
    assert not caplog.records
