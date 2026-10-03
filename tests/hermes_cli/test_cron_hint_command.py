"""The in-session cron hint command persists the same option used by the scheduler."""

import pytest

from cron.jobs import create_job, get_job, load_jobs
from cron.scheduler_prompt import _build_job_prompt, _CRON_HINT
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def test_slash_hint_can_inspect_disable_and_restore_by_name_or_id(capsys):
    job = create_job(prompt="Reply with pong.", schedule="every 5h", name="Sample ping", paused=True)
    host = CLICommandsMixin()
    host._handle_cron_command('/cron hint "Sample ping"')
    assert "on" in capsys.readouterr().out
    assert not get_job(job["id"])["skip_cron_hint"]

    host._handle_cron_command('/cron hint "Sample ping" off')
    assert "off" in capsys.readouterr().out
    stored = get_job(job["id"])
    assert stored["skip_cron_hint"] is True
    assert _build_job_prompt(stored) == job["prompt"]
    assert stored["enabled"] is False

    host._handle_cron_command(f'/cron hint {job["id"]}')
    assert "off" in capsys.readouterr().out
    host._handle_cron_command(f'/cron hint {job["id"]} on')
    assert "on" in capsys.readouterr().out
    assert get_job(job["id"])["skip_cron_hint"] is False
    assert _build_job_prompt(get_job(job["id"])) == _CRON_HINT + job["prompt"]

    host._handle_cron_command("/cron")
    assert "/cron hint" in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["", "same maybe", "same off extra", "missing off", "same off"])
def test_slash_hint_rejects_invalid_or_ambiguous_requests_without_writes(argument, capsys):
    for _ in range(2):
        create_job(prompt="Reply with pong.", schedule="every 5h", name="same", paused=True)
    before = load_jobs()
    CLICommandsMixin()._handle_cron_command(f"/cron hint {argument}")
    output = capsys.readouterr().out
    assert "Unknown /cron command" not in output
    assert any(word in output.lower() for word in ("usage", "not found", "ambiguous"))
    assert load_jobs() == before
