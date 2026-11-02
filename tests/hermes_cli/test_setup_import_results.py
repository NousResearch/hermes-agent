"""Setup must preserve import output and repeat newly paused routines (#148134)."""

from argparse import Namespace
import subprocess

import pytest

from cron import jobs
from hermes_cli import setup, setup_agent_import, setup_summary


@pytest.mark.parametrize("profile", ["default", "research"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_routines_only_import_waits_and_summary_keeps_resume_commands(tmp_path, monkeypatch, capsys, profile, returncode):
    root = tmp_path / "hermes"
    home = root if profile == "default" else root / "profiles" / profile
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    events = []
    imported = []
    no_longer_paused = []
    with jobs.use_cron_store(home):
        existing = jobs.create_job(prompt="old", schedule="every 1h", name="Unrelated paused job", paused=True)

    def run(argv, env, check=False):
        assert env["HERMES_HOME"] == str(home)
        with jobs.use_cron_store(home):
            imported.append(jobs.create_job(prompt="new", schedule="every 1h", name="Scout: Daily brief", paused=True))
            jobs.create_job(prompt="active", schedule="every 1h", name="Already active routine")
            for name in ("Resumed before setup ended", "Deleted before setup ended"):
                no_longer_paused.append(jobs.create_job(prompt="temporary", schedule="every 1h", name=name, paused=True))
        events.append("child output")
        return subprocess.CompletedProcess(argv, returncode)

    def prompt(question, default=None, password=False):
        assert question == "Press Enter to continue setup"
        events.append("acknowledged")
        return ""

    def wizard(args):
        # Exercise the real import and common completion path under the real wizard invocation scope.
        assert setup_agent_import._run_import(setup_agent_import.IMPORT_SOURCES[4], home) is False
        events.append("mode menu")
        with jobs.use_cron_store(home):
            jobs.resume_job(no_longer_paused[0]["id"])
            jobs.remove_job(no_longer_paused[1]["id"])
        setup_summary._print_setup_summary({}, home)

    monkeypatch.setattr(setup_agent_import.subprocess, "run", run)
    monkeypatch.setattr(setup, "prompt", prompt)
    monkeypatch.setattr(setup, "_run_setup_wizard_impl", wizard)
    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda: object())
    monkeypatch.setattr(setup_summary, "_TOOL_ROW_BUILDERS", ())
    monkeypatch.setattr(setup_summary.nous_subscription, "get_nous_subscription_features", lambda config: None)
    setup.run_setup_wizard(Namespace())
    assert events == ["child output", "acknowledged", "mode menu"]
    output = capsys.readouterr().out
    assert f"hermes -p {profile} cron resume {imported[0]['id']}" in output
    assert "Scout: Daily brief" in output
    assert existing["id"] not in output
    assert "Already active routine" not in output
    assert all(job["id"] not in output for job in no_longer_paused)
    assert "turn off" in output.lower()

    # A subsequent setup invocation cannot inherit stale import hints from the previous one.
    monkeypatch.setattr(setup, "_run_setup_wizard_impl", lambda args: setup_summary._print_setup_summary({}, home))
    setup.run_setup_wizard(Namespace())
    assert imported[0]["id"] not in capsys.readouterr().out


def test_standalone_migrate_never_waits_for_setup_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(setup_agent_import.subprocess, "run", lambda argv, env, check=False: subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(setup, "prompt", lambda *args, **kwargs: pytest.fail("standalone migrate must not pause"))
    assert setup_agent_import._run_import(setup_agent_import.IMPORT_SOURCES[4], tmp_path, from_setup=False) is False
