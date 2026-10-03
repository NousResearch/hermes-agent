"""A pooled Desktop backend (``hermes --profile X serve``, HERMES_HOME=<root>/profiles/X) answers its
REST without ``?profile=``. Those UNSCOPED reads/verbs are about X — a profile the live default
multiplexer serves — and must agree with the scoped ``?profile=X`` answer and with ``hermes -p X status``:
running-via-multiplexer, start/stop refused, restart addressed to the multiplexer's home.

Live repro (Desktop over a multiplexed HOME): Command Center said "Messaging gateway stopped" for X,
the Messaging page pinned every platform "gateway stopped", and Restart spawned a bare ``gateway
restart`` under X's HERMES_HOME that exited 78 while the UI reported success.
"""

from __future__ import annotations

import contextlib
import json
import os

import pytest


@pytest.fixture
def pooled_served_process(tmp_path, monkeypatch):
    """Process whose HERMES_HOME is a served named profile; the default home records a live multiplexer."""
    root = tmp_path / "hermes"
    for name in ("alpha", "solo"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n")  # identity marker
    (root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
        "served_profiles": ["default", "alpha"],
        "platforms": {"api_server": {"state": "connected"}, "alpha:telegram": {"state": "connected"}}}))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "alpha"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    import hermes_constants
    import gateway.status as status
    # Liveness is a verified identity; this pytest process passes as the default gateway only by
    # wearing a gateway command line.
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: False)
    return root


def test_unscoped_liveness_in_a_served_profile_process_matches_the_scoped_answer(pooled_served_process):
    from gateway.status import profile_platforms_from_multiplexer, resolve_gateway_liveness
    alpha = pooled_served_process / "profiles" / "alpha"
    scoped = resolve_gateway_liveness(profile_dir=alpha, health_probe=None, use_cache=False)
    unscoped = resolve_gateway_liveness(health_probe=None, use_cache=False)
    assert (unscoped.running, unscoped.pid, unscoped.source) == (scoped.running, scoped.pid, "multiplexer")
    plats = profile_platforms_from_multiplexer(unscoped.runtime, "alpha")
    assert plats["telegram"] == {"state": "connected"} and plats["api_server"]["state"] == "connected"


def test_profile_action_environment_binds_launch_scope_before_scrubbing(pooled_served_process, monkeypatch):
    import hermes_cli.web_server_gateway as web_server_gateway
    import tools.environments.local as local_env
    import tui_gateway.launch_profile_policy as launch_policy

    entered = []

    @contextlib.contextmanager
    def _scope():
        entered.append("enter")
        yield
        entered.append("exit")

    monkeypatch.setattr(launch_policy, "launch_profile_scope_if_multiplexed", _scope)
    original_build = local_env.build_subprocess_env

    def _build(*args, **kwargs):
        assert entered == ["enter"]
        return original_build(*args, **kwargs)

    monkeypatch.setattr(local_env, "build_subprocess_env", _build)
    env = web_server_gateway._profile_action_environment(["-p", "alpha", "doctor"])

    assert env["HERMES_HOME"] == str(pooled_served_process / "profiles" / "alpha")
    assert entered == ["enter", "exit"]


def test_profile_action_does_not_forward_managed_scope_secret_to_child_profile(pooled_served_process, monkeypatch):
    import hermes_cli.env_loader as env_loader
    import hermes_cli.managed_scope as managed_scope
    import hermes_cli.web_server_gateway as web_server_gateway
    import tools.env_passthrough as env_passthrough
    import tui_gateway.launch_profile_policy as launch_policy
    from agent import secret_scope

    managed_dir = pooled_served_process / "managed"
    managed_dir.mkdir()
    (managed_dir / ".env").write_text("MANAGED_ONLY_TOKEN=managed-secret-value\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    managed_scope.invalidate_managed_cache()
    assert managed_scope.load_managed_env() == {"MANAGED_ONLY_TOKEN": "managed-secret-value"}
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", {"MANAGED_ONLY_TOKEN"})
    monkeypatch.delenv("MANAGED_ONLY_TOKEN", raising=False)
    env_passthrough.register_env_passthrough(["MANAGED_ONLY_TOKEN"])

    @contextlib.contextmanager
    def _launch_scope():
        token = secret_scope.set_secret_scope(
            {"MANAGED_ONLY_TOKEN": "managed-secret-value"},
            profile_home=str(pooled_served_process),
        )
        try:
            yield
        finally:
            secret_scope.reset_secret_scope(token)

    monkeypatch.setattr(launch_policy, "launch_profile_scope_if_multiplexed", _launch_scope)
    active_token = secret_scope.set_multiplex_context(True)
    try:
        env = web_server_gateway._profile_action_environment(["-p", "alpha", "doctor"])
    finally:
        secret_scope.reset_multiplex_context(active_token)

    assert env["HERMES_HOME"] == str(pooled_served_process / "profiles" / "alpha")
    assert "MANAGED_ONLY_TOKEN" not in env


def test_profile_action_does_not_forward_launch_only_registered_passthrough(pooled_served_process, monkeypatch):
    import contextlib

    import hermes_cli.web_server_gateway as web_server_gateway
    import tools.env_passthrough as env_passthrough
    import tui_gateway.launch_profile_policy as launch_policy
    from agent import secret_scope

    name = "PR_130671_LAUNCH_ONLY_TOKEN"
    monkeypatch.setenv(name, "launch-profile-secret")
    env_passthrough.register_env_passthrough([name])

    @contextlib.contextmanager
    def _launch_scope():
        token = secret_scope.set_secret_scope({}, profile_home=str(pooled_served_process))
        try:
            yield
        finally:
            secret_scope.reset_secret_scope(token)

    monkeypatch.setattr(launch_policy, "launch_profile_scope_if_multiplexed", _launch_scope)
    active_token = secret_scope.set_multiplex_context(True)
    try:
        env = web_server_gateway._profile_action_environment(["-p", "alpha", "doctor"])
    finally:
        secret_scope.reset_multiplex_context(active_token)

    assert name not in env


def test_unscoped_lifecycle_verbs_in_a_served_profile_process_address_the_multiplexer(pooled_served_process):
    from hermes_cli.web_server_gateway import _gateway_subcommand, _profile_action_environment, multiplexed_profile_refusal
    # `stop` on a served profile PARKS it under the host (no refusal); `start` while unparked refuses.
    assert multiplexed_profile_refusal(None, "stop") is None and multiplexed_profile_refusal(None, "start")
    restart = _gateway_subcommand(None, "restart")
    assert restart[-2:] == ["gateway", "restart"]
    # The child must run under the DEFAULT home (the multiplexer's), not inherit alpha's HERMES_HOME.
    assert _profile_action_environment(restart)["HERMES_HOME"] == str(pooled_served_process)
    # A profile with no multiplexer relationship is still managed as its own gateway.
    assert _gateway_subcommand("solo", "restart") == ["-p", "solo", "gateway", "restart"]
    assert multiplexed_profile_refusal("solo", "stop") is None
