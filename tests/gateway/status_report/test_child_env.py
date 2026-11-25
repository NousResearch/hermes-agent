"""The URLs a deployment stamps into the gateway's launch environment reach a gateway this process
starts, and no other child: not the terminal tool, not a credentialed child, not a child that acts
for another profile."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gateway.scale_to_zero import SLEEP_URL_ENV
from gateway.status_report.target import STATUS_REPORT_URL_ENV
from tests.gateway.status_report.support import new_credential, stamped_url
from tests.tools._child_env_fixtures import child_env as child_env  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
from tests.tools._child_env_fixtures import observe_terminal
from tools.environments import local

STAMPS = (STATUS_REPORT_URL_ENV, SLEEP_URL_ENV)


@pytest.fixture
def stamped(child_env, monkeypatch) -> dict[str, str]:
    """A gateway process whose launch environment carries both stamps."""
    values = {STATUS_REPORT_URL_ENV: stamped_url(new_credential()),
              SLEEP_URL_ENV: f"https://sleep.example/instance?t={new_credential()}"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def _stamps(env) -> dict:
    return {name: env.get(name) for name in STAMPS}


def test_the_terminal_tool_never_sees_them(stamped, child_env):
    terminal = local.LocalEnvironment(cwd=str(child_env), timeout=30)
    try:
        assert observe_terminal(terminal, STAMPS) == dict.fromkeys(STAMPS)
    finally:
        terminal.cleanup()


@pytest.mark.parametrize("build", [
    pytest.param(lambda: local._sanitize_subprocess_env(dict(os.environ)), id="background command"),
    pytest.param(local.hermes_subprocess_env, id="plain child"),
    pytest.param(lambda: local.hermes_subprocess_env(inherit_credentials=True), id="credentialed child"),
    pytest.param(lambda: local.served_profile_child_env(inherit_credentials=True), id="child for the launch profile"),
])
def test_no_child_of_the_launch_profile_sees_them(stamped, build):
    assert _stamps(build()) == dict.fromkeys(STAMPS)


def test_a_child_for_another_profile_never_sees_them(stamped, child_env):
    """Launch-only keys count as global, so the routed strip keeps them: the scrub must not."""
    other = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    other.mkdir(parents=True)

    for credentials in (False, True):
        env = local.served_profile_child_env(target_home=other, inherit_credentials=credentials)
        assert _stamps(env) == dict.fromkeys(STAMPS), f"inherit_credentials={credentials}"


def test_a_gateway_this_process_starts_keeps_them(stamped):
    from gateway.run_shutdown import GatewayShutdownMixin
    from hermes_cli.gateway_restart_env import restart_watcher_env

    assert _stamps(local.host_gateway_child_env()) == stamped
    assert _stamps(GatewayShutdownMixin._restart_watcher_env()) == stamped
    assert _stamps(restart_watcher_env(["gateway", "run", "--replace"], host=True, home=None)) == stamped


def test_a_named_profile_gateway_restarting_itself_keeps_them(stamped, monkeypatch):
    """A standalone named-profile gateway restarts on its own home, not the host's."""
    from gateway.run_shutdown import GatewayShutdownMixin

    named = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    named.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(named))

    env = GatewayShutdownMixin._restart_watcher_env()

    assert env["HERMES_HOME"] == str(named)
    assert _stamps(env) == stamped


def test_a_gateway_started_for_another_profile_keeps_them(stamped):
    other = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    other.mkdir(parents=True)

    assert _stamps(local.gateway_child_env(other)) == stamped


def test_a_dashboard_gateway_action_for_another_profile_keeps_them_and_other_actions_do_not(stamped, monkeypatch):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server_gateway import _profile_action_environment

    other = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    other.mkdir(parents=True)
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "other")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: other)

    assert _stamps(_profile_action_environment(["-p", "other", "gateway", "restart"])) == stamped
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in {"other", "gateway"})
    # A profile may be called like the command: only the selector's value is skipped.
    assert _stamps(_profile_action_environment(["-p", "gateway", "gateway", "restart"])) == stamped
    assert _stamps(_profile_action_environment(["-p", "other", "gateway", "install"])) == stamped
    assert _stamps(_profile_action_environment(["-p", "other", "skills", "install", "x", "--yes"])) == dict.fromkeys(STAMPS)


def test_a_served_child_built_from_the_process_env_drops_them(stamped, child_env):
    """An explicit base (the bot relay's delivery child passes os.environ) is scrubbed too."""
    from tools.bot_relay import delivery_env

    assert _stamps(local.served_profile_child_env(base=os.environ, inherit_credentials=True)) == dict.fromkeys(STAMPS)
    assert _stamps(delivery_env(None, Path(os.environ["HERMES_HOME"]))) == dict.fromkeys(STAMPS)


def test_a_child_for_a_profile_whose_env_sets_the_sleep_url_never_sees_it(stamped, child_env):
    other = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    other.mkdir(parents=True)
    (other / ".env").write_text(f"{SLEEP_URL_ENV}=https://sleep.example/own?t={new_credential()}\n")

    assert _stamps(local.served_profile_child_env(target_home=other, inherit_credentials=True)) == dict.fromkeys(STAMPS)


def test_a_gateway_for_a_profile_keeps_that_profiles_own_sleep_url(stamped, child_env):
    other = Path(os.environ["HERMES_HOME"]) / "profiles" / "other"
    other.mkdir(parents=True)
    own = f"https://sleep.example/own?t={new_credential()}"
    (other / ".env").write_text(f"{SLEEP_URL_ENV}={own}\n")

    env = local.gateway_child_env(other)

    assert env[SLEEP_URL_ENV] == own, "the launch environment's sleep URL replaced the profile's own"
    assert env[STATUS_REPORT_URL_ENV] == stamped[STATUS_REPORT_URL_ENV]


def test_dashboard_actions_without_a_profile_drop_them_except_update(stamped, child_env):
    from hermes_cli.web_server_gateway import _profile_action_environment

    assert _stamps(_profile_action_environment(["skills", "install", "x", "--yes"])) == dict.fromkeys(STAMPS)
    assert _stamps(_profile_action_environment(["update"])) == stamped
