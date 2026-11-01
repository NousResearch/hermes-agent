"""A failed stale-config cleanup must not half-apply an environment setting."""

import pytest


@pytest.mark.parametrize("operation", ["set", "unset"])
def test_unreadable_config_preserves_env_settings_across_profiles(tmp_path, monkeypatch, operation):
    from hermes_cli.config_env_routing import remove_env_setting, save_env_setting
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home

    homes = [tmp_path / "home-a", tmp_path / "home-b"]
    for home in homes:
        home.mkdir()
        (home / ".env").write_text("HERMES_TIMEZONE=UTC\n", encoding="utf-8")
        (home / "config.yaml").write_text("model: [unterminated", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))

    set_multiplex_active(True)
    try:
        for home in [homes[0], homes[1], homes[0]]:
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                assert get_hermes_home() == home
                env_before = (home / ".env").read_bytes()
                config_before = (home / "config.yaml").read_bytes()
                with pytest.raises(RuntimeError, match="formatting error"):
                    if operation == "set":
                        save_env_setting("HERMES_TIMEZONE", "Europe/London")
                    else:
                        remove_env_setting("HERMES_TIMEZONE")
                assert (home / ".env").read_bytes() == env_before
                assert (home / "config.yaml").read_bytes() == config_before
        assert get_hermes_home() == homes[0]
    finally:
        set_multiplex_active(False)
