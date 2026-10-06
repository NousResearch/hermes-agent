from types import SimpleNamespace


def _setup(monkeypatch, tmp_path, scope):
    from gateway import run

    homes = [("default", tmp_path / "default"), ("main", tmp_path / "main")]
    monkeypatch.setattr(run, "_multiplex_profile_homes", lambda _config: homes)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "main")
    monkeypatch.setattr(
        "hermes_cli.profiles.configured_profile_scope",
        lambda key: scope if key == "cron_profile_scope" else None,
    )
    return run, homes


def test_cron_scope_absent_preserves_upstream(monkeypatch, tmp_path):
    run, homes = _setup(monkeypatch, tmp_path, None)
    assert run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True)) == homes


def test_cron_scope_selects_main_only(monkeypatch, tmp_path):
    run, homes = _setup(monkeypatch, tmp_path, {"main"})
    assert run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True)) == [homes[1]]


def test_cron_scope_unknown_warns_and_valid_survives(monkeypatch, tmp_path, caplog):
    run, homes = _setup(monkeypatch, tmp_path, {"main", "missing"})
    with caplog.at_level("WARNING"):
        assert run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True)) == [homes[1]]
    assert "unknown profiles" in caplog.text


def test_cron_scope_malformed_and_empty_are_fail_closed(monkeypatch, tmp_path):
    run, _homes = _setup(monkeypatch, tmp_path, frozenset())
    assert run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True)) == []


def test_cron_scope_cannot_activate_unloaded_profile(monkeypatch, tmp_path):
    run, _homes = _setup(monkeypatch, tmp_path, {"staging"})
    assert run._cron_tick_profile_homes(SimpleNamespace(multiplex_profiles=True)) == []
