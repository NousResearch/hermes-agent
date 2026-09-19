"""Message contracts for the config-parse and sign-in failure copy (CLI startup UX campaign).

Every test asserts the contract (plain lead sentence + the exact command pointer, raw exception
kept out of the lead) rather than the whole string.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
import hermes_yaml as yaml


def _yaml_error(text: str = "model:\n  default: x\n bad: [unterminated\n") -> Exception:
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return exc
    raise AssertionError("expected a YAML error")


# ── cli-03: config.yaml parse failure at startup ─────────────────────────────────────────────


class TestConfigParseWarning:
    @pytest.mark.parametrize("fallback", ["defaults", "last-known-good-backup", "refuse-write"])
    def test_warning_names_repair_commands_and_line(self, tmp_path, monkeypatch, capsys, fallback):
        from hermes_cli import config as config_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_mod._CONFIG_PARSE_WARNED.clear()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  default: x\n bad: [unterminated\n", encoding="utf-8")

        config_mod._warn_config_parse_failure(config_path, _yaml_error(), fallback=fallback)

        err = capsys.readouterr().err
        assert "formatting error" in err
        assert "line " in err
        assert "`hermes config edit`" in err and "`hermes config check`" in err
        assert "IGNORED" not in err and "LAST KNOWN GOOD" not in err
        # The raw PyYAML text is demoted to a Details line, never the lead sentence.
        lead = err.strip().splitlines()[0]
        assert "expected <block end>" not in lead.split("Details:")[0]


# ── cli-33: refuse-to-write and non-interactive startup guard ────────────────────────────────


class TestConfigGuardCopy:
    def test_require_parseable_user_config_says_what_happened_and_how_to_fix(self, tmp_path, monkeypatch):
        from hermes_cli import config as config_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")

        with pytest.raises(config_mod.InvalidUserConfigError) as excinfo:
            config_mod.require_parseable_user_config()

        msg = str(excinfo.value)
        assert msg.startswith("Hermes stopped because your settings file")
        assert "`hermes config edit`" in msg and "--ignore-user-config" in msg
        assert "non-interactive" not in msg and "built-in defaults" not in msg
        assert "Details:" in msg

    def test_refuse_write_names_repair_commands_and_absolute_backup_dir(self, tmp_path, monkeypatch):
        from hermes_cli import config as config_mod
        from hermes_constants import display_hermes_home

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_mod._CONFIG_PARSE_WARNED.clear()
        (tmp_path / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")

        with pytest.raises(RuntimeError) as excinfo:
            config_mod.require_readable_config_before_write()

        msg = str(excinfo.value)
        assert "this change was not saved" in msg
        assert "`hermes config edit`" in msg and "`hermes config check`" in msg
        assert f"{display_hermes_home()}/backups/config/" in msg
        assert not msg.startswith("Refusing")


# ── cli-04: no provider configured ───────────────────────────────────────────────────────────


def test_no_provider_configured_points_at_model_login_and_auth_add(monkeypatch, tmp_path):
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_constants import display_hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda env=None: False)

    with pytest.raises(AuthError) as excinfo:
        resolve_provider("auto")

    msg = str(excinfo.value)
    assert excinfo.value.code == "no_provider_configured"
    assert "`hermes model`" in msg and "`/login`" in msg and "`hermes auth add <provider>`" in msg
    assert "~/.hermes/.env" not in msg or display_hermes_home() == "~/.hermes"
    assert f"{display_hermes_home()}/.env" in msg
    assert "inference provider" not in msg


# ── cli-07: Nous device-code sign-in failures ────────────────────────────────────────────────


class TestSignInFailureCopy:
    @pytest.mark.parametrize("code, pointer", [
        ("expired_token", "`hermes portal`"),
        ("access_denied", "`hermes model`"),
    ])
    def test_poll_error_codes_map_to_plain_copy(self, code, pointer):
        from hermes_cli import auth_device_flow

        class _Client:
            def post(self, url, data=None):
                request = httpx.Request("POST", url)
                return httpx.Response(400, json={"error": code, "error_description": "server text"}, request=request)

        with pytest.raises(RuntimeError) as excinfo:
            auth_device_flow._poll_for_token(
                client=_Client(), portal_base_url="https://portal.nousresearch.com", client_id="hermes-cli",
                device_code="device", expires_in=5, poll_interval=1)

        msg = str(excinfo.value)
        assert not msg.startswith(code)
        assert "`hermes portal`" in msg and pointer in msg

    def test_network_failure_names_host_and_retry_command(self):
        from hermes_cli.auth_error_copy import sign_in_failure_lines

        lines = sign_in_failure_lines(httpx.ConnectError("[Errno -2] Name or service not known"),
                                      service_host="portal.nousresearch.com")

        assert lines[0].startswith("Could not sign in")
        assert "portal.nousresearch.com" in lines[0] and "`hermes portal`" in lines[0]
        assert "Errno" not in lines[0]
        assert any(line.strip().startswith("Details:") for line in lines[1:])

    def test_run_login_prints_plain_copy_not_raw_exception(self, capsys):
        from hermes_cli.model_setup_flows_common import _run_login

        def _boom(*_a, **_k):
            raise httpx.ConnectError("[Errno -2] Name or service not known")

        assert _run_login(_boom) is False
        out = capsys.readouterr().out
        assert "Could not sign in" in out
        assert not out.lstrip().startswith("Login failed:")

    def test_login_nous_prints_plain_copy(self, monkeypatch, capsys):
        from hermes_cli import auth_nous

        def _boom(**_k):
            raise httpx.ConnectError("[Errno -2] Name or service not known")

        monkeypatch.setattr(auth_nous, "_offer_shared_nous_import", lambda *_a, **_k: None, raising=False)
        monkeypatch.setattr("hermes_cli.auth._nous_device_code_login", _boom)
        args = SimpleNamespace(timeout=1.0, portal_url=None, inference_url=None, client_id=None, scope=None,
                               no_browser=True, insecure=False, ca_bundle=None)
        from hermes_cli.auth import PROVIDER_REGISTRY

        with pytest.raises(SystemExit) as excinfo:
            auth_nous._login_nous(args, PROVIDER_REGISTRY["nous"])

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Could not sign in" in out and "`hermes portal`" in out
        assert "Login failed:" not in out


# ── cli-25: auth refresh / unknown provider ──────────────────────────────────────────────────


class TestAuthCommandsCopy:
    def test_unknown_provider_offers_did_you_mean_and_pointers(self, monkeypatch):
        from hermes_cli import auth_commands

        monkeypatch.setattr(auth_commands, "_configured_provider_entry", lambda _p: None)
        with pytest.raises(SystemExit) as excinfo:
            auth_commands.auth_add_command(SimpleNamespace(provider="openai"))

        msg = str(excinfo.value)
        assert "Unknown provider 'openai'" in msg
        assert "Did you mean" in msg and "openai-codex" in msg
        assert "`hermes auth`" in msg and "`hermes model`" in msg

    def test_refresh_failure_names_relogin_command(self, monkeypatch):
        from hermes_cli import auth_commands

        from agent.credential_pool import AUTH_TYPE_OAUTH

        matched = SimpleNamespace(id="abc", label="device-code", source="device_code", provider="nous",
                                  auth_type=AUTH_TYPE_OAUTH, refresh_token="rt", last_status="ok")

        class _Pool:
            def entries(self):
                return [matched]

            def try_refresh_matching(self, credential_id=None):
                matched.last_status = "refresh_failed"
                return None

        monkeypatch.setattr(auth_commands, "load_pool", lambda _provider: _Pool())
        with pytest.raises(SystemExit) as excinfo:
            auth_commands.auth_refresh_command(SimpleNamespace(provider="nous", target=None))

        msg = str(excinfo.value)
        assert "refresh_failed" not in msg and "status now" not in msg
        assert "`hermes auth add nous --type oauth`" in msg


# ── cli-24: /model onto a provider without credentials ───────────────────────────────────────


def test_creds_for_target_provider_names_auth_add_and_model(monkeypatch):
    from hermes_cli import model_switch

    st = model_switch._Switch(
        raw_input="kimi-coding:x", current_provider="openrouter", current_model="m", current_base_url="",
        current_api_key="", is_global=False, explicit_provider="", user_providers=None, custom_providers=None,
        target_provider="kimi-coding", provider_label="Kimi")

    def _fail(**_k):
        raise RuntimeError("KIMI_API_KEY missing (raw resolver text)")

    monkeypatch.setattr(st, "resolve_runtime", _fail, raising=False)
    result = model_switch._creds_for_switched_provider(st)

    assert result is not None and result.success is False
    msg = result.error_message
    assert msg.startswith("Kimi is not connected")
    assert "`hermes auth add kimi-coding`" in msg and "/model" in msg
    assert "Could not resolve credentials" not in msg
    assert "Details:" in msg


# ── cli-16: setup wizard provider step failure ───────────────────────────────────────────────


class TestProviderSetupFailureCopy:
    def test_setup_provider_step_reports_reason_and_that_nothing_changed(self, monkeypatch, capsys, tmp_path):
        from hermes_cli import setup as setup_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def _boom(*_a, **_k):
            raise httpx.ConnectError("[Errno -2] Name or service not known")

        monkeypatch.setattr("hermes_cli.main.select_provider_and_model", _boom)
        monkeypatch.setattr(setup_mod, "print_header", lambda *_a, **_k: None)
        monkeypatch.setattr(setup_mod, "_info", lambda *_a, **_k: None)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        setup_mod.setup_model_provider({}, quick=True)

        out = capsys.readouterr().out
        assert "Could not finish connecting a provider" in out
        assert "internet" in out
        assert "not changed" in out or "Nothing was saved" in out
        assert "hermes model" in out
        assert "encountered an error" not in out and "Errno" not in out.split("Details:")[0]

    def test_quick_setup_error_hook_uses_same_copy(self, capsys):
        from hermes_cli.auth_error_copy import provider_setup_failure_lines

        lines = provider_setup_failure_lines(RuntimeError("keyring locked"), retry_command="hermes portal")
        assert lines[0].startswith("Could not finish connecting a provider")
        assert "hermes portal" in " ".join(lines)
        assert any(line.strip().startswith("Details:") for line in lines)


class TestRunLoginProviderAwareCopy:
    def test_minimax_failure_names_minimax_retry_not_portal(self, capsys):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.model_setup_flows_common import _run_login

        def _boom(*_a, **_k):
            raise httpx.ConnectError("[Errno -2] Name or service not known")

        assert _run_login(_boom, SimpleNamespace(), PROVIDER_REGISTRY["minimax-oauth"]) is False
        out = capsys.readouterr().out
        assert "hermes auth add minimax-oauth" in out
        assert "hermes portal" not in out and "portal.nousresearch.com" not in out
        assert "hermes login" not in out

    def test_nous_failure_keeps_portal_retry(self, capsys):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.model_setup_flows_common import _run_login

        def _boom(*_a, **_k):
            raise httpx.ConnectError("boom")

        assert _run_login(_boom, SimpleNamespace(), PROVIDER_REGISTRY["nous"]) is False
        out = capsys.readouterr().out
        assert "hermes portal" in out and "portal.nousresearch.com" in out

    def test_silent_nonzero_system_exit_still_tells_user_how_to_retry(self, capsys):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.model_setup_flows_common import _run_login

        def _quiet_exit(*_a, **_k):
            raise SystemExit(1)

        assert _run_login(_quiet_exit, SimpleNamespace(), PROVIDER_REGISTRY["openai-codex"]) is False
        out = capsys.readouterr().out
        assert "Sign-in did not complete" in out and "hermes auth add openai-codex" in out

    def test_system_exit_with_message_surfaces_the_message(self, capsys):
        from hermes_cli.model_setup_flows_common import _run_login

        def _exit_msg(*_a, **_k):
            raise SystemExit("token endpoint returned 400")

        assert _run_login(_exit_msg) is False
        out = capsys.readouterr().out
        assert "token endpoint returned 400" in out and "hermes model" in out

    def test_cancel_codes_stay_a_single_cancel_line(self, capsys):
        from hermes_cli.model_setup_flows_common import _run_login

        def _cancel(*_a, **_k):
            raise SystemExit(130)

        assert _run_login(_cancel) is False
        out = capsys.readouterr().out
        assert "cancelled" in out.lower() and "did not complete" not in out
