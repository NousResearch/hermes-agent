"""Contract tests for alias authentication against the bundled email adapter."""

import asyncio
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.email import adapter as email_base

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins" / "email-alias-login"


def plugin_module():
    name = "email_alias_login_plugin_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "env_login,config_login,expected",
    [
        (None, None, "alias@example.com"),
        ("account@provider.example", None, "account@provider.example"),
        (None, "yaml@provider.example", "yaml@provider.example"),
        ("env@provider.example", "yaml@provider.example", "env@provider.example"),
        ("  ", "yaml@provider.example", "yaml@provider.example"),
        (None, "  ", "alias@example.com"),
    ],
)
def test_gateway_and_standalone_login_identity(env_login, config_login, expected):
    plugin = plugin_module().adapter
    env = {"EMAIL_ADDRESS": "alias@example.com", "EMAIL_PASSWORD": "test-secret",
           "EMAIL_IMAP_HOST": "imap.example.com", "EMAIL_SMTP_HOST": "smtp.example.com"}
    if env_login is not None:
        env["EMAIL_LOGIN_USER"] = env_login
    extra = {"login_user": config_login} if config_login is not None else {}
    with patch.dict(os.environ, env, clear=False), \
         patch.object(email_base, "_send_imap_id"):
        if env_login is None:
            os.environ.pop("EMAIL_LOGIN_USER", None)
        adapter = plugin.EmailAliasAdapter(PlatformConfig(enabled=True, extra=extra))
        imap, smtp, standalone = MagicMock(), MagicMock(), MagicMock()
        imap.uid.return_value = ("OK", [b""])
        with patch.object(adapter, "_connect_imap", return_value=imap), \
             patch.object(adapter, "_connect_smtp", return_value=smtp), \
             patch.object(email_base, "_open_smtp", return_value=standalone):
            assert adapter._probe_imap(False)
            assert adapter._probe_smtp()
            sent = asyncio.run(adapter.send("recipient@example.com", "hello"))
            out = asyncio.run(plugin.standalone_send(
                PlatformConfig(enabled=True, extra={**extra, "address": "alias@example.com"}),
                "recipient@example.com", "hello"))
    assert sent.success and out["success"]
    assert imap.login.call_args.args == (expected, "test-secret")
    assert smtp.login.call_args_list[0].args == (expected, "test-secret")
    assert smtp.login.call_args_list[1].args == (expected, "test-secret")
    assert standalone.login.call_args.args == (expected, "test-secret")
    assert smtp.send_message.call_args.args[0]["From"] == "alias@example.com"
    assert standalone.send_message.call_args.args[0]["From"] == "alias@example.com"


def test_registration_preserves_bundled_metadata():
    plugin = plugin_module().adapter
    captured = {}

    class Context:
        def register_platform(self, **fields):
            captured.update(fields)

    plugin.register(Context())
    assert captured["name"] == "email"
    assert captured["adapter_factory"] is plugin.EmailAliasAdapter
    assert captured["standalone_sender_fn"] is plugin.standalone_send
    assert captured["allowed_users_env"] == "EMAIL_ALLOWED_USERS"
    assert "EMAIL_IMAP_HOST" in captured["required_env"]
    assert captured.get("apply_yaml_config_fn") is None


def test_real_plugin_discovery_and_yaml_scope_a_b_a(tmp_path, monkeypatch):
    """An enabled install overrides only its owning home, including YAML bridge."""
    from gateway.config import Platform, load_gateway_config
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    homes = [tmp_path / name for name in ("a", "b")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(
            "plugins:\n  enabled:\n" +
            ("    - email-alias-login\n" if home == homes[0] else "    []\n") +
            "platforms:\n  email:\n    enabled: true\n" +
            f"    login_user: {home.name}@provider.example\n" +
            "    extra:\n      address: alias@example.com\n" +
            "      imap_host: imap.example.com\n      smtp_host: smtp.example.com\n",
            encoding="utf-8",
        )

    for key in ("EMAIL_LOGIN_USER", "EMAIL_ADDRESS", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMAIL_PASSWORD", "synthetic-secret")
    for home, enabled in ((homes[0], True), (homes[1], False), (homes[0], True)):
        monkeypatch.setenv("HERMES_HOME", str(home))
        discover_plugins(force=True)
        entry = platform_registry.get("email")
        assert entry is not None
        config = load_gateway_config().platforms[Platform.EMAIL]
        if enabled:
            assert entry.plugin_name == "email-alias-login"
            assert config.extra["login_user"] == "a@provider.example"
            adapter = entry.adapter_factory(config)
            assert adapter._login_user == "a@provider.example"
            assert adapter._address == "alias@example.com"
        else:
            assert entry.plugin_name != "email-alias-login"


def test_multiplex_secret_scopes_do_not_borrow_another_profile_login(tmp_path, monkeypatch):
    from agent.secret_scope import (
        is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
    )

    plugin = plugin_module().adapter
    prior = is_multiplex_active()
    monkeypatch.setenv("EMAIL_LOGIN_USER", "wrong-profile@example.net")
    set_multiplex_active(True)
    try:
        for name, login, configured, expected in (
            ("a", "a@provider.example", None, "a@provider.example"),
            ("b", None, "b-config@provider.example", "b-config@provider.example"),
            ("a", "a@provider.example", None, "a@provider.example"),
        ):
            home = tmp_path / name
            home.mkdir(exist_ok=True)
            monkeypatch.setenv("HERMES_HOME", str(home))
            secrets = {"EMAIL_ADDRESS": "alias@example.com", "EMAIL_PASSWORD": "test-secret",
                       "EMAIL_IMAP_HOST": "imap.example.com", "EMAIL_SMTP_HOST": "smtp.example.com"}
            if login:
                secrets["EMAIL_LOGIN_USER"] = login
            extra = {"login_user": configured} if configured else {}
            token = set_secret_scope(secrets)
            try:
                adapter = plugin.EmailAliasAdapter(PlatformConfig(enabled=True, extra=extra))
                smtp = MagicMock()
                with patch.object(email_base, "_open_smtp", return_value=smtp):
                    result = asyncio.run(plugin.standalone_send(
                        PlatformConfig(enabled=True, extra={**extra, "address": "alias@example.com"}),
                        "recipient@example.com", "hello"))
                assert adapter._login_user == expected
                assert result["success"]
                assert smtp.login.call_args.args == (expected, "test-secret")
                assert smtp.send_message.call_args.args[0]["From"] == "alias@example.com"
            finally:
                reset_secret_scope(token)
    finally:
        set_multiplex_active(prior)


def test_yaml_merge_precedence_survives_plugin_registration(tmp_path, monkeypatch):
    """The plugin must not overwrite the loader's winning extra.login_user."""
    from gateway.config import Platform, load_gateway_config
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - email-alias-login\n"
        "gateway:\n  platforms:\n    email:\n      login_user: stale@example.net\n"
        "platforms:\n  email:\n    enabled: true\n"
        "    login_user: desired@example.net\n"
        "    extra:\n      address: public@example.com\n"
        "      imap_host: imap.example.com\n      smtp_host: smtp.example.com\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("EMAIL_LOGIN_USER", "EMAIL_ADDRESS", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMAIL_PASSWORD", "synthetic-secret")
    discover_plugins(force=True)
    entry = platform_registry.get("email")
    assert entry.plugin_name == "email-alias-login"
    config = load_gateway_config().platforms[Platform.EMAIL]
    assert config.extra["login_user"] == "desired@example.net"
    adapter = entry.adapter_factory(config)
    smtp = MagicMock()
    with patch.object(adapter, "_connect_smtp", return_value=smtp):
        assert adapter._probe_smtp()
    smtp.login.assert_called_once_with("desired@example.net", "synthetic-secret")


def test_manifest_requires_the_host_the_adapter_needs():
    from hermes_yaml import safe_load

    manifest = safe_load((PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8-sig"))
    required = {item["name"] for item in manifest["requires_env"]}
    assert "EMAIL_IMAP_HOST" in required
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "alias@example.com", "EMAIL_PASSWORD": "synthetic-secret",
        "EMAIL_SMTP_HOST": "smtp.example.com", "EMAIL_IMAP_HOST": "imap.example.com",
    }):
        assert email_base.check_email_requirements()


def test_installer_prompts_for_password_without_echoing_it():
    from hermes_cli import plugins_cmd_install as installer
    from hermes_yaml import safe_load

    manifest = safe_load((PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8-sig"))
    password = next(item for item in manifest["requires_env"] if item["name"] == "EMAIL_PASSWORD")
    with patch.object(installer, "_pc") as plugin_commands, \
         patch.object(installer, "line_input") as visible_prompt, \
         patch("hermes_cli.config.save_env_value"), \
         patch.dict(os.environ, {}, clear=False):
        plugin_commands.return_value._missing_env_specs.return_value = [password]
        plugin_commands.return_value.masked_secret_prompt.return_value = "synthetic-secret"
        visible_prompt.return_value = "synthetic-visible"
        installer._prompt_plugin_env_vars(manifest, MagicMock())
        plugin_commands.return_value.masked_secret_prompt.assert_called_once()
        visible_prompt.assert_not_called()


def test_standalone_address_and_host_follow_gateway_env_precedence():
    plugin = plugin_module().adapter
    env = {"EMAIL_ADDRESS": "env@example.com", "EMAIL_PASSWORD": "synthetic-secret",
           "EMAIL_IMAP_HOST": "imap.env.example.com", "EMAIL_SMTP_HOST": "smtp.env.example.com"}
    config = PlatformConfig(enabled=True, extra={
        "address": "yaml@example.com", "smtp_host": "smtp.yaml.example.com"})
    smtp = MagicMock()
    with patch.dict(os.environ, env, clear=False), \
         patch.object(email_base, "_open_smtp", return_value=smtp) as opener:
        os.environ.pop("EMAIL_LOGIN_USER", None)
        adapter = plugin.EmailAliasAdapter(config)
        result = asyncio.run(plugin.standalone_send(config, "recipient@example.com", "hello"))
    assert result["success"]
    assert (adapter._address, adapter._login_user, adapter._smtp_host) == (
        "env@example.com", "env@example.com", "smtp.env.example.com")
    smtp.login.assert_called_once_with("env@example.com", "synthetic-secret")
    assert smtp.send_message.call_args.args[0]["From"] == "env@example.com"
    assert opener.call_args.args[0] == "smtp.env.example.com"


@pytest.mark.parametrize("failure", ["login", "send_message"])
def test_standalone_releases_smtp_after_failure(failure):
    plugin = plugin_module().adapter
    server = MagicMock()
    getattr(server, failure).side_effect = OSError("synthetic transport failure")
    env = {"EMAIL_ADDRESS": "alias@example.com", "EMAIL_PASSWORD": "synthetic-secret",
           "EMAIL_SMTP_HOST": "smtp.example.com"}
    with patch.dict(os.environ, env, clear=False), \
         patch.object(email_base, "_open_smtp", return_value=server):
        os.environ.pop("EMAIL_LOGIN_USER", None)
        result = asyncio.run(plugin.standalone_send(
            PlatformConfig(enabled=True), "recipient@example.com", "hello"))
    assert result.get("error")
    assert server.quit.called or server.close.called


def test_smtp_quit_error_closes_connection():
    plugin = plugin_module().adapter
    server = MagicMock()
    server.quit.side_effect = OSError("synthetic quit failure")
    with patch.dict(os.environ, {"EMAIL_ADDRESS": "alias@example.com",
                              "EMAIL_PASSWORD": "synthetic-secret", "EMAIL_SMTP_HOST": "smtp.example.com"}), \
         patch.object(email_base, "_open_smtp", return_value=server):
        adapter = plugin.EmailAliasAdapter(PlatformConfig(enabled=True))
        adapter._connect_smtp = lambda: server
        adapter._probe_smtp()
        asyncio.run(plugin.standalone_send(PlatformConfig(enabled=True), "recipient@example.com", "hello"))
    assert server.close.called
