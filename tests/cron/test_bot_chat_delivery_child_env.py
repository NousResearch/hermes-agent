"""``deliver: bot-chat:<other profile>`` spawns its turn with the TARGET profile's environment.

The Bot Chat CLI lane is the only cron child built for a profile other than the one whose tick
spawned it, so it is the only lane whose launch-residue strip cannot be resolved from the ambient
home override. It built the child env with ``strip_launch_profile_env(env)`` and no target, which
is a no-op whenever the gateway runs its own launch profile — the child then carried the launch
profile's ``.env`` settings, bridged ``TERMINAL_*`` policy, authorization gates (#113270) and
credentials into another profile's turn.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import cron.scheduler_delivery as delivery

# What a gateway process that loaded the ROOT profile's .env holds in os.environ.
LAUNCH_ENV = {
    "HERMES_MODEL": "root-model",
    "HERMES_LANGUAGE": "en",
    "TERMINAL_ENV": "docker",
    "TERMINAL_DOCKER_IMAGE": "root-only-image",
    "DISCORD_ALLOWED_USERS": "root-operator",  # an authorization gate (#113270)
    "DISCORD_IGNORED_CHANNELS": "999",
    "ANTHROPIC_API_KEY": "sk-root",
}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A root-profile gateway (``hermes gateway run``) and a second profile to deliver into."""
    root = tmp_path / "hermes"
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True)
    (root / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in LAUNCH_ENV.items()) + "\n", encoding="utf-8")
    (beta / ".env").write_text("ANTHROPIC_API_KEY=sk-beta\nHERMES_LANGUAGE=ja\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for key, value in LAUNCH_ENV.items():
        monkeypatch.setenv(key, value)
    return root, beta


def _capture_child_env(monkeypatch) -> dict:
    """Run the lane up to its spawn and return the env it would have used."""
    captured: dict = {}

    def _fake_turn(argv, env, report_path, timeout):
        captured.update(env=dict(env), argv=list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(delivery, "_run_bot_chat_turn", _fake_turn)
    return captured


def test_the_turn_for_another_profile_carries_that_profile_s_environment(fleet, monkeypatch):
    root, beta = fleet
    captured = _capture_child_env(monkeypatch)

    assert delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta") is None

    env = captured["env"]
    assert env["HERMES_HOME"] == str(beta)
    # The launch profile's settings and bridged terminal policy are not beta's.
    for key in ("HERMES_MODEL", "TERMINAL_ENV", "TERMINAL_DOCKER_IMAGE"):
        assert key not in env, f"{key} leaked from the launch profile"
    # Authorization gates decide who may talk to the agent — never inherited across profiles.
    for key in ("DISCORD_ALLOWED_USERS", "DISCORD_IGNORED_CHANNELS"):
        assert key not in env, f"{key} leaked from the launch profile (#113270)"
    # Credentials are the target profile's own, not the gateway's.
    assert env["ANTHROPIC_API_KEY"] == "sk-beta"
    # Beta's own .env is loaded by the child itself; what matters here is that root's value is gone.
    assert env.get("HERMES_LANGUAGE") != "en"


def test_a_delivery_into_the_gateway_s_own_bot_chat_keeps_its_environment(fleet, monkeypatch):
    """``deliver: bot-chat`` (no profile) is the job's own home: nothing to strip."""
    root, _beta = fleet
    captured = _capture_child_env(monkeypatch)

    assert delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "") is None

    env = captured["env"]
    assert env["HERMES_HOME"] == str(root)
    assert env["HERMES_MODEL"] == "root-model"
    assert env["TERMINAL_DOCKER_IMAGE"] == "root-only-image"
    assert env["ANTHROPIC_API_KEY"] == "sk-root"
    assert env["DISCORD_ALLOWED_USERS"] == "root-operator"


def test_the_delivery_child_still_runs_this_install_and_targets_the_named_chat(fleet, monkeypatch):
    """Guard rails the env change must not disturb: the running install, and `-p default`
    only for a root home."""
    root, beta = fleet
    captured = _capture_child_env(monkeypatch)
    delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta")
    argv = captured["argv"]
    assert argv[:3] == [os.sys.executable, "-m", "hermes_cli.main"]
    assert "-p" not in argv  # a profile home is addressed by HERMES_HOME, not a flag
    assert argv[3:10] == ["chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing", "-Q"]

    captured.clear()
    delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "")
    assert captured["argv"][3:5] == ["-p", "default"]  # a root home keeps its explicit profile flag


def test_a_missing_target_home_is_refused_before_any_child_is_built(fleet, monkeypatch):
    root, beta = fleet
    captured = _capture_child_env(monkeypatch)
    (beta / ".env").unlink()
    beta.rmdir()

    result = delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta")

    assert result is not None and "no longer exists" in result
    assert captured == {}


def test_an_unbuildable_target_environment_is_refused_and_no_turn_is_spawned(fleet, monkeypatch):
    """The lane reports every failure as a string; building the target env must not raise past it
    (``_deliver_result``'s fan-out does not catch, unlike the deferred drain)."""
    from tools.environments import local as local_env

    _root, _beta = fleet
    captured = _capture_child_env(monkeypatch)
    monkeypatch.setattr(local_env, "served_profile_child_env",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("home is 0700 for another user")))

    result = delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta")

    assert result is not None and "do not resend" in result and "PermissionError" in result
    assert captured == {}


def test_the_child_env_is_built_for_the_target_even_while_a_sibling_home_override_is_active(fleet, monkeypatch):
    """Under multiplexing the tick runs with the JOB's profile as the home override; the strip must
    still be resolved against the delivery target, not against whichever home is ambient."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root, beta = fleet
    gamma = root / "profiles" / "gamma"
    gamma.mkdir(parents=True)
    (gamma / ".env").write_text("GAMMA_ONLY=1\nDISCORD_ALLOWED_USERS=gamma-operator\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "gamma-operator")
    captured = _capture_child_env(monkeypatch)

    token = set_hermes_home_override(str(gamma))
    try:
        delivery._deliver_to_bot_chat({"id": "j", "name": "nightly"}, "the brief", "beta")
    finally:
        reset_hermes_home_override(token)

    env = captured["env"]
    assert env["HERMES_HOME"] == str(beta)
    assert "DISCORD_ALLOWED_USERS" not in env, "the firing profile's gate reached another profile's turn"
    assert env["ANTHROPIC_API_KEY"] == "sk-beta"
