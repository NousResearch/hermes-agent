"""Profile-isolation regression for /goal quality-gate subprocesses."""

import subprocess
from unittest.mock import patch

from hermes_cli.goals import GoalGate, run_gate
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_goal_gate_subprocess_uses_routed_profile_env(tmp_path, monkeypatch):
    """A served profile's quality gate must not inherit the launch profile env."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()

    # ``strip_launch_profile_env`` can identify this value as launch-profile residue.
    (launch_home / ".env").write_text("GOAL_GATE_LAUNCH_ONLY=launch\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("GOAL_GATE_LAUNCH_ONLY", "launch")

    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args[0], 0, stdout="ok\n", stderr="")

    token = set_hermes_home_override(served_home)
    try:
        with patch("hermes_cli.goals.subprocess.run", side_effect=fake_run):
            passed, code, out = run_gate(GoalGate(command="printf ok"))
    finally:
        reset_hermes_home_override(token)

    assert passed is True
    assert code == 0
    assert "ok" in out
    child_env = captured["env"]
    assert child_env is not None, "quality gate must receive an explicit routed child env"
    assert child_env.get("HERMES_HOME") == str(served_home)
    assert "GOAL_GATE_LAUNCH_ONLY" not in child_env
