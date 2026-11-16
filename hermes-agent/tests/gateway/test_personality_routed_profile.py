"""#89161: a routed multiplex profile's personality must reach its turns.

``GatewayRunner`` used to snapshot ``_ephemeral_system_prompt`` once at boot
from the launch profile's config and hand that string to every routed turn,
so a secondary profile's ``display.personality`` / ``agent.system_prompt`` never
injected. ``_get_system_prompt_for_channel`` now resolves from the config of
the profile currently in scope (``run_sync`` runs inside
``_profile_runtime_scope``).
"""

from __future__ import annotations

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope


def test_routed_profile_prompt_resolves_from_its_own_config(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "beta"
    default_home.mkdir()
    routed_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text("agent:\n  system_prompt: DEFAULT-PERSONA\n")
    (routed_home / "config.yaml").write_text(
        "agent:\n  system_prompt: BETA-PERSONA\n  personalities:\n    pirate: ARR\n"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", raising=False)

    runner = object.__new__(GatewayRunner)
    runner.config = None

    with _profile_runtime_scope(routed_home):
        assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, "c") == "BETA-PERSONA"
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, "c") == "DEFAULT-PERSONA"

    # /personality from the routed chat writes the routed profile and only it.
    from hermes_cli.personality import persist_personality

    with _profile_runtime_scope(routed_home):
        assert persist_personality("pirate")
        assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, "c") == "ARR"
    assert "pirate" not in (default_home / "config.yaml").read_text()
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, "c") == "DEFAULT-PERSONA"
