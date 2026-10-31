"""busy_ledger sees turns the in-process ``gateway run`` host drives (Desktop on the canonical gateway)."""

import sys
import types
import weakref


def test_idle_proof_reports_a_gateway_runner_turn_as_busy(monkeypatch):
    from hermes_cli.web_server_idle_proof import idle_proof

    class Runner:
        _running_agents: dict = {}

    runner = Runner()
    fake_run = types.ModuleType("gateway.run")
    fake_run._gateway_runner_ref = weakref.ref(runner)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    assert idle_proof(input_probe=lambda: 0)["idle"] is True

    runner._running_agents = {"agent:main:desktop:dm:abc": object()}
    verdict = idle_proof(input_probe=lambda: 0)

    assert verdict["idle"] is False
    assert verdict["detail"] == "session:agent:main:desktop:dm:abc"


def test_idle_proof_reports_a_claimed_admission_before_its_agent_is_mapped(monkeypatch):
    """A messaging/API turn is running from its claim stamp on; the agent map fills in later."""
    from hermes_cli.web_server_idle_proof import idle_proof

    class Task:
        def done(self):
            return False

    live = types.SimpleNamespace(route="agent:main:telegram:dm:42", task=Task(),
                                 event_stream=types.SimpleNamespace(execution={"execution_generation": 1}))
    authority = types.SimpleNamespace(sessions={"sid-42": live})

    class Runner:
        _running_agents: dict = {}
        session_authority = authority

    runner = Runner()
    fake_run = types.ModuleType("gateway.run")
    fake_run._gateway_runner_ref = weakref.ref(runner)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    verdict = idle_proof(input_probe=lambda: 0)

    assert verdict["idle"] is False
    assert verdict["detail"] == "session:sid-42"
