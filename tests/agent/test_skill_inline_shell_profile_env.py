"""Profile-isolation regression for SKILL.md inline shell expansion."""

import subprocess
from unittest.mock import patch

from agent.skill_preprocessing import run_inline_shell
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_inline_shell_subprocess_uses_routed_profile_env(tmp_path, monkeypatch):
    """A served profile's inline shell must not inherit the launch profile env."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()

    # ``strip_launch_profile_env`` can identify this value as launch-profile residue.
    (launch_home / ".env").write_text("SKILL_INLINE_LAUNCH_ONLY=launch\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("SKILL_INLINE_LAUNCH_ONLY", "launch")

    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args[0], 0, stdout="ok\n", stderr="")

    token = set_hermes_home_override(served_home)
    try:
        with patch("agent.skill_preprocessing.subprocess.run", side_effect=fake_run):
            assert run_inline_shell("printf ok", served_home, 5) == "ok"
    finally:
        reset_hermes_home_override(token)

    child_env = captured["env"]
    assert child_env is not None, "inline shell must receive an explicit routed child env"
    assert child_env.get("HERMES_HOME") == str(served_home)
    assert "SKILL_INLINE_LAUNCH_ONLY" not in child_env
