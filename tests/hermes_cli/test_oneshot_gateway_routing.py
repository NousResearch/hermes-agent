"""``hermes -z`` runs on the gateway session authority, never on an agent built in-process.

Refs #145171 (EP-11a): the former ``hermes_cli.oneshot.run_oneshot`` / ``_run_agent`` path built a
bare ``AIAgent`` outside the gateway and was reachable only from tests.
"""

import sys

import pytest


def test_top_level_oneshot_routes_through_the_gateway_launcher(monkeypatch):
    import hermes_cli.main as main_mod

    seen = []
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "hello"])
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr("hermes_cli.gateway_chat_startup.launch_gateway_chat",
                        lambda args: seen.append(args.oneshot) or 0)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 0
    assert seen == ["hello"]


def test_no_in_process_oneshot_agent_runner_remains():
    from hermes_cli import main as main_mod, oneshot

    assert not hasattr(oneshot, "run_oneshot")
    assert not hasattr(oneshot, "_run_agent")
    assert not hasattr(main_mod, "_run_and_exit_oneshot")
