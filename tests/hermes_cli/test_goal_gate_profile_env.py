"""Profile-isolation regression for /goal quality-gate subprocesses."""

import subprocess
from unittest.mock import patch

from hermes_cli.goals import GoalGate, run_gate
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_goal_gate_child_env_tracks_routed_profile_a_b_a(tmp_path, monkeypatch):
    """A goal gate must follow the active profile, never the launch process env."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()

    # This marker models launch-profile .env residue. A routed child must lose it,
    # while returning to the launch profile must restore the launch environment.
    (launch_home / ".env").write_text("GOAL_GATE_LAUNCH_ONLY=launch\n", encoding="utf-8")
    (served_home / ".env").write_text("GOAL_GATE_SERVED_ONLY=served\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("GOAL_GATE_LAUNCH_ONLY", "launch")

    captured_envs = []

    def fake_run(*args, **kwargs):
        env = kwargs.get("env")
        captured_envs.append(dict(env) if env is not None else None)
        return subprocess.CompletedProcess(args[0], 0, stdout="ok\n", stderr="")

    with patch("hermes_cli.goals.subprocess.run", side_effect=fake_run):
        assert run_gate(GoalGate(command="printf ok"))[0] is True

        token = set_hermes_home_override(served_home)
        try:
            assert run_gate(GoalGate(command="printf ok"))[0] is True
        finally:
            reset_hermes_home_override(token)

        assert run_gate(GoalGate(command="printf ok"))[0] is True

    assert all(env is not None for env in captured_envs), (
        "quality gates must receive an explicit profile-routed child environment"
    )
    first, served, last = captured_envs
    assert first["HERMES_HOME"] == str(launch_home)
    assert first["GOAL_GATE_LAUNCH_ONLY"] == "launch"
    assert served["HERMES_HOME"] == str(served_home)
    assert "GOAL_GATE_LAUNCH_ONLY" not in served
    assert served["GOAL_GATE_SERVED_ONLY"] == "served"
    assert last["HERMES_HOME"] == str(launch_home)
    assert last["GOAL_GATE_LAUNCH_ONLY"] == "launch"
    assert "GOAL_GATE_SERVED_ONLY" not in first
    assert "GOAL_GATE_SERVED_ONLY" not in last


def test_goal_gate_real_shell_child_receives_only_served_profile_env(tmp_path, monkeypatch):
    """Exercise the real shell boundary, not only subprocess.run's mocked kwargs."""
    import json
    import os
    import shlex
    import sys

    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    (launch_home / ".env").write_text("GOAL_GATE_LAUNCH_ONLY=launch\n", encoding="utf-8")
    (served_home / ".env").write_text("GOAL_GATE_SERVED_ONLY=served\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("GOAL_GATE_LAUNCH_ONLY", "launch")

    probe = tmp_path / "probe_gate_env.py"
    probe.write_text(
        "import json, os\n"
        "print(json.dumps({key: os.environ.get(key) for key in "
        "('HERMES_HOME', 'GOAL_GATE_LAUNCH_ONLY', 'GOAL_GATE_SERVED_ONLY')}))\n",
        encoding="utf-8",
    )
    argv = [sys.executable, str(probe)]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)

    token = set_hermes_home_override(served_home)
    try:
        passed, exit_code, output = run_gate(GoalGate(command=command))
    finally:
        reset_hermes_home_override(token)

    assert (passed, exit_code) == (True, 0), output
    child = json.loads(output)
    assert child == {
        "HERMES_HOME": str(served_home),
        "GOAL_GATE_LAUNCH_ONLY": None,
        "GOAL_GATE_SERVED_ONLY": "served",
    }
    assert os.environ["GOAL_GATE_LAUNCH_ONLY"] == "launch"
