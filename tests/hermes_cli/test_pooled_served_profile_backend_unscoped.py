"""A pooled Desktop backend (``hermes --profile X serve``, HERMES_HOME=<root>/profiles/X) answers its
REST without ``?profile=``. Those UNSCOPED reads/verbs are about X — a profile the live default
multiplexer serves — and must agree with the scoped ``?profile=X`` answer and with ``hermes -p X status``:
running-via-multiplexer, start/stop refused, restart addressed to the multiplexer's home.

Live repro (Desktop over a multiplexed HOME): Command Center said "Messaging gateway stopped" for X,
the Messaging page pinned every platform "gateway stopped", and Restart spawned a bare ``gateway
restart`` under X's HERMES_HOME that exited 78 while the UI reported success.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def pooled_served_process(tmp_path, monkeypatch):
    """Process whose HERMES_HOME is a served named profile; the default home records a live multiplexer."""
    root = tmp_path / "hermes"
    (root / "profiles" / "alpha").mkdir(parents=True)
    (root / "profiles" / "solo").mkdir(parents=True)
    (root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
        "served_profiles": ["default", "alpha"],
        "platforms": {"api_server": {"state": "connected"}, "alpha:telegram": {"state": "connected"}}}))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "alpha"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    import hermes_constants
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
    assert profile_platforms_from_multiplexer(unscoped.runtime, "alpha") == {"telegram": {"state": "connected"}}


def test_unscoped_lifecycle_verbs_in_a_served_profile_process_address_the_multiplexer(pooled_served_process):
    from hermes_cli.web_server_gateway import _gateway_subcommand, _profile_action_environment, multiplexed_profile_refusal
    assert multiplexed_profile_refusal(None, "stop") and multiplexed_profile_refusal(None, "start")
    restart = _gateway_subcommand(None, "restart")
    assert restart[-2:] == ["gateway", "restart"]
    # The child must run under the DEFAULT home (the multiplexer's), not inherit alpha's HERMES_HOME.
    assert _profile_action_environment(restart)["HERMES_HOME"] == str(pooled_served_process)
    # A profile with no multiplexer relationship is still managed as its own gateway.
    assert _gateway_subcommand("solo", "restart") == ["-p", "solo", "gateway", "restart"]
    assert multiplexed_profile_refusal("solo", "stop") is None
