"""Dashboard actions scrub launch credentials without unscoped reads (#130663)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import secret_scope
from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import env_passthrough, terminal_scope
from tui_gateway import launch_profile_policy


@pytest.fixture
def action_homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    homes = {"default": root, "alpha": root / "profiles" / "alpha",
             "beta": root / "profiles" / "beta"}
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "terminal: {env_passthrough: [GRAFANA_SERVICE_ACCOUNT_TOKEN]}\n",
            encoding="utf-8",
        )
        (home / ".env").write_text(
            f"GRAFANA_SERVICE_ACCOUNT_TOKEN={name}-token\n", encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", "default-token")
    monkeypatch.setenv("OPENAI_API_KEY", "launch-provider-key")
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    return homes


@pytest.mark.parametrize("argv,name", [
    (["doctor"], "doctor"),
    (["backup", "create"], "backup"),
    (["skills", "update"], "skills-update"),
])
@pytest.mark.parametrize("routed", [False, True], ids=["unscoped", "routed"])
@pytest.mark.parametrize("env_only", [False, True], ids=["dotenv", "service-environment"])
def test_multiplexed_actions_spawn_isolated_children_and_restore_caller_scope(
    action_homes, monkeypatch, argv, name, routed, env_only,
):
    from hermes_cli import web_server_gateway
    from hermes_cli.web_routers._common import spawn_profile_action
    from tools.environments.local import build_subprocess_env

    if env_only:
        for home in action_homes.values():
            (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("ACTION_OPAQUE_VALUE", "launch-only-value")
    allowed_token = env_passthrough._allowed_env_vars_var.set(set())
    env_passthrough.register_env_passthrough(["ACTION_OPAQUE_VALUE", "LANG"])
    launch_profile_policy.activate_multi_profile_hosting()
    caller_secrets = {"GRAFANA_SERVICE_ACCOUNT_TOKEN": "beta-token"} if routed else None
    caller_terminal = {"TERMINAL_ENV": "local"} if routed else None
    home = str(action_homes["beta"]) if routed else None
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(caller_secrets)
    terminal_token = terminal_scope.set_terminal_scope(caller_terminal)
    popen = Mock(return_value=SimpleNamespace(pid=12345))
    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", popen)
    monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", action_homes["default"] / "logs")
    for attr in ("_ACTION_PROCS", "_ACTION_COMMANDS", "_ACTION_IDS", "_ACTION_RESULTS"):
        monkeypatch.setattr(web_server_gateway, attr, {})
    try:
        if not routed:
            # The real sanitizer still refuses a credential read with no scope.
            with pytest.raises(secret_scope.UnscopedSecretError):
                build_subprocess_env(scrub_secrets=True)
        for target in ("alpha", "beta", "alpha", "default"):
            result = spawn_profile_action(
                target, argv, name, log_msg="action failed", prefix="action failed",
            )
            assert result == {"ok": True, "pid": popen.return_value.pid, "name": name}
            command = popen.call_args.args[0]
            assert command[-len(argv) - 2:] == ["-p", target, *argv]
            env = popen.call_args.kwargs["env"]
            assert env["HERMES_HOME"] == str(action_homes[target])
            assert env["HERMES_NONINTERACTIVE"] == "1"
            own_env_only = target == "default" and env_only
            assert env.get("GRAFANA_SERVICE_ACCOUNT_TOKEN") == (
                "default-token" if own_env_only else None
            )
            assert env.get("ACTION_OPAQUE_VALUE") == (
                "launch-only-value" if target == "default" else None
            )
            assert env["LANG"] == web_server_gateway.os.environ["LANG"]
            assert "OPENAI_API_KEY" not in env
            assert "_HERMES_GATEWAY" not in env
            assert get_hermes_home_override() == home
            assert secret_scope.current_secret_scope() is caller_secrets
            assert terminal_scope.get_terminal_scope() is caller_terminal
            assert web_server_gateway._ACTION_PROCS[name] is popen.return_value
        assert secret_scope.is_multiplex_active()
        assert web_server_gateway.os.environ["GRAFANA_SERVICE_ACCOUNT_TOKEN"] == "default-token"
    finally:
        terminal_scope.reset_terminal_scope(terminal_token)
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        env_passthrough._allowed_env_vars_var.reset(allowed_token)


def test_single_profile_action_keeps_ambient_scrubbing_and_caller_overrides(action_homes):
    from hermes_cli.web_server_gateway import _profile_action_environment

    assert not secret_scope.is_multiplex_active()
    before = (get_hermes_home_override(), secret_scope.current_secret_scope(),
              terminal_scope.get_terminal_scope())
    env = _profile_action_environment(
        ["-p", "alpha", "doctor"], {"HERMES_ACTION_ID": "action-id"},
    )
    assert env["HERMES_HOME"] == str(action_homes["alpha"])
    assert env["HERMES_ACTION_ID"] == "action-id"
    assert "GRAFANA_SERVICE_ACCOUNT_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "_HERMES_GATEWAY" not in env
    assert (get_hermes_home_override(), secret_scope.current_secret_scope(),
            terminal_scope.get_terminal_scope()) == before
