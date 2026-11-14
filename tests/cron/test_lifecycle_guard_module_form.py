"""Gateway lifecycle guards cover CLI and Python module invocations (#152712)."""

import pytest

from cron import lifecycle_guard
from cron.jobs import create_job


@pytest.mark.parametrize("launcher", [
    "hermes",
    "python -m hermes_cli.main",
    "python3 -m hermes_cli.main",
    "python3.12 -I -u -m hermes_cli.main",
    "python.exe -m hermes_cli.main",
    "/opt/example/bin/python3 -m hermes_cli.main",
])
@pytest.mark.parametrize("verb", ["restart", "stop", "uninstall"])
def test_create_job_refuses_lifecycle_launchers(launcher, verb):
    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked, match="gateway lifecycle command"):
        create_job(prompt=f"{launcher} gateway {verb}", schedule="every 1h")


@pytest.mark.parametrize("launcher", ["hermes", "python -m hermes_cli.main", "python3.12 -I -m hermes_cli.main"])
@pytest.mark.parametrize("verb", ["restart", "stop"])
def test_create_job_refuses_self_profile_lifecycle_launchers(launcher, verb, monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_current_profile_name", lambda: "default")

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked, match="gateway lifecycle command"):
        create_job(prompt=f"{launcher} --profile=default gateway {verb}", schedule="every 1h")


@pytest.mark.parametrize("command", [
    "hermes gateway start",
    "python -m hermes_cli.main gateway start",
    "python -m hermes_cli.main gateway status",
    "python -m hermes_cli.main -p sibling gateway restart",
    "python -m hermes_cli.main --profile sibling gateway stop",
    "python -m another_module gateway restart",
    "python -m hermes_cli.main_extra gateway restart",
])
def test_module_form_preserves_allowed_commands(command, monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_current_profile_name", lambda: "default")

    assert not lifecycle_guard.contains_gateway_lifecycle_command(command)
