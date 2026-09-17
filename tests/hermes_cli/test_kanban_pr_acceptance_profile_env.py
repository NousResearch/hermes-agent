"""Profile-isolation regression for Kanban GitHub acceptance subprocesses."""

import subprocess
from unittest.mock import patch

from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_acceptance_gh_child_env_tracks_routed_profile_a_b_a(tmp_path, monkeypatch):
    """Acceptance probes must follow the active profile, never the launch process env."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()

    # This marker models launch-profile .env residue. A routed child must lose it,
    # while returning to the launch profile must restore the launch environment.
    (launch_home / ".env").write_text("KANBAN_ACCEPTANCE_LAUNCH_ONLY=launch\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("KANBAN_ACCEPTANCE_LAUNCH_ONLY", "launch")

    captured_envs = []

    def fake_run(*args, **kwargs):
        env = kwargs.get("env")
        captured_envs.append(dict(env) if env is not None else None)
        return subprocess.CompletedProcess(args[0], 0, stdout="{}\n", stderr="")

    with patch("hermes_cli.kanban_pr_acceptance.subprocess.run", side_effect=fake_run):
        assert acceptance._api("rate_limit") == {}

        token = set_hermes_home_override(served_home)
        try:
            assert acceptance._api("rate_limit") == {}
        finally:
            reset_hermes_home_override(token)

        assert acceptance._api("rate_limit") == {}

    assert all(env is not None for env in captured_envs), (
        "Kanban acceptance probes must receive an explicit profile-routed child environment"
    )
    first, served, last = captured_envs
    assert first["HERMES_HOME"] == str(launch_home)
    assert first["KANBAN_ACCEPTANCE_LAUNCH_ONLY"] == "launch"
    assert served["HERMES_HOME"] == str(served_home)
    assert "KANBAN_ACCEPTANCE_LAUNCH_ONLY" not in served
    assert last["HERMES_HOME"] == str(launch_home)
    assert last["KANBAN_ACCEPTANCE_LAUNCH_ONLY"] == "launch"
