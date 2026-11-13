"""``hermes -z`` runs on the gateway session authority, never on an agent built in-process.

Refs #145171 (EP-11a): the former ``hermes_agent.hermes_cli.oneshot.run_oneshot`` / ``_run_agent`` path built a
bare ``AIAgent`` outside the gateway and was reachable only from tests.
"""

import sys

import pytest


def test_top_level_oneshot_routes_through_the_gateway_launcher(monkeypatch):
    import hermes_agent.hermes_cli.main as main_mod

    seen = []
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "hello"])
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr("hermes_agent.hermes_cli.gateway_chat_startup.launch_gateway_chat",
                        lambda args: seen.append(args.oneshot) or 0)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 0
    assert seen == ["hello"]



def test_get_fallback_chain_stays_importable_from_oneshot():
    """Public name kept on purpose: external scripts set ``hermes_agent.hermes_cli.oneshot.get_fallback_chain``."""
    from hermes_agent.hermes_cli import fallback_config, oneshot

    assert oneshot.get_fallback_chain is fallback_config.get_fallback_chain
