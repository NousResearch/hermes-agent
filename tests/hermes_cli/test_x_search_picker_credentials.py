"""CLI and Desktop share a persisted X Search credential choice."""

import pytest


@pytest.mark.parametrize("index,source", [(0, "oauth"), (1, "api_key")])
def test_picker_selection_roundtrips_into_runtime(tmp_path, monkeypatch, index, source):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tools_config import TOOL_CATEGORIES, apply_provider_selection, _is_provider_active
    from tools.x_search_tool import _load_x_search_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    rows = TOOL_CATEGORIES["x_search"]["providers"]
    apply_provider_selection("x_search", rows[index]["name"], config)
    save_config(config)
    assert _load_x_search_config()["credential_source"] == source
    reloaded = load_config()
    assert _is_provider_active(rows[index], reloaded)
    assert not _is_provider_active(rows[1-index], reloaded)


def test_shared_xai_setup_does_not_promise_oauth_precedence(monkeypatch):
    from hermes_cli import tools_config_post_setup as setup

    monkeypatch.setattr("hermes_cli.auth.get_xai_oauth_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(setup, "get_env_value", lambda _: "test-key")
    messages = []
    monkeypatch.setattr(setup, "_print_success", messages.append)
    setup._post_setup_xai_grok()
    assert not any("will use" in msg for msg in messages)
    assert any("configured" in msg for msg in messages)


def test_oauth_picker_setup_does_not_accept_api_key(monkeypatch):
    from hermes_cli import tools_config_post_setup as setup

    monkeypatch.setattr("hermes_cli.auth.get_xai_oauth_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(setup, "get_env_value", lambda _: "test-key")
    monkeypatch.setattr("hermes_cli.setup_tts._run_xai_oauth_login_from_setup", lambda: False)
    messages = []
    monkeypatch.setattr(setup, "_print_warning", messages.append)
    setup._run_post_setup("x_search_oauth")
    assert any("No API-key fallback" in msg for msg in messages)
    assert setup._POST_SETUP_AUTH_READY["x_search_oauth"]() is False
    monkeypatch.setattr("hermes_cli.auth.get_xai_oauth_auth_status", lambda: {"logged_in": True})
    assert setup._POST_SETUP_AUTH_READY["x_search_oauth"]() is True
