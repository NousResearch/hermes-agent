"""The dashboard's external-dependency check must outlive a CLI's slow cold start.

A provider's ``external_dependencies[].check`` spawns real CLIs (npm, brv, uv). On low-power
platforms (Termux) the brv/Node cold start alone runs 22-23s, so the old 20s budget raised
TimeoutExpired on a correctly installed provider and the dashboard marked it unavailable behind
"Install provider dependencies" (#139954).
"""

from types import SimpleNamespace

import hermes_cli.web_server_memory as wsm


def test_external_dependency_check_budget_clears_slow_cold_starts(monkeypatch):
    seen = {}

    def fake_run_setup_command(command, *, display, shell=False, timeout=180):
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wsm, "_run_setup_command", fake_run_setup_command)

    setup = {"external_dependencies": [{"name": "brv", "check": "brv --version"}]}
    assert wsm._memory_provider_external_dependencies_installed(setup) is True
    # A ~22s CLI cold start must fit inside the check budget with margin.
    assert seen["timeout"] >= 60
