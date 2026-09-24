"""An opt-in report floor must suppress content and record failure, not silence (#118367)."""
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler


@pytest.mark.parametrize("response,minimum,expected", [
    ("🔍 **", 20, ""),
    ("  12345  ", 6, ""),
    ("  12345  ", 5, "  12345  "),
    ("OK", 0, "OK"),
    ("OK", None, "OK"),
    ("[SILENT]", 100, "[SILENT]"),
    ("SILENT", 100, "SILENT"),
    ("NO_REPLY", 100, "NO_REPLY"),
    ("NO REPLY", 100, "NO REPLY"),
    ("[CRON_FAILURE]\nupstream unavailable", 100, "[CRON_FAILURE]\nupstream unavailable"),
    ("", 100, ""),
    ("(No response generated)", 100, ""),
])
@pytest.mark.parametrize("notice", [None, "Provider fallback notice long enough to exceed the report floor."])
def test_run_job_respects_report_floor(tmp_path, response, minimum, expected, notice):
    from cron.jobs import create_job, get_job, update_job
    job = create_job("Give a report", "every 2h", model="fixture-model", deliver="local")
    job = update_job(job["id"], {"min_response_chars": minimum})
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": response, "completed": True, "failed": False,
        "turn_exit_reason": "text_response(finish_reason=stop)", "messages": [],
    }
    db = MagicMock()
    db.get_compression_tip.side_effect = lambda sid: sid
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._preflight_job_config", return_value=None),
        patch("hermes_state_registry.acquire", return_value=db),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
            "api_key": "fixture", "base_url": "https://example.invalid/v1",
            "provider": "openrouter", "api_mode": "chat_completions", "_fallback_notice": notice}),
        patch("run_agent.AIAgent", return_value=agent),
    ):
        success, output, final_response, error = scheduler.run_job(job)
    assert success is True, error
    assert error is None
    declared_failure = scheduler._cron_failure_marker_error(expected) is not None
    if notice and expected.strip() and not scheduler._is_cron_silence_response(expected) and not declared_failure:
        expected = f"{notice}\n\n{expected}"
    assert final_response == expected
    agent.run_conversation.assert_called_once()
    with (
        patch("cron.scheduler.run_job", return_value=(success, output, final_response, error)),
        patch("cron.scheduler._deliver_result", return_value=None) as deliver,
    ):
        assert scheduler.run_one_job(job)
    stored = get_job(job["id"])
    assert stored["last_status"] == ("ok" if expected and not declared_failure else "error")
    if not expected or scheduler._is_cron_silence_response(expected):
        deliver.assert_not_called()


def test_cli_floor_roundtrip_validation_and_reset(tmp_path, capsys):
    import argparse
    from cron.jobs import get_job, list_jobs
    from hermes_cli.cron import cron_create, cron_edit
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda args: None)
    args = parser.parse_args(["cron", "create", "every 2h", "Give a report",
                              "--paused", "--min-response-chars", "20"])
    with patch("hermes_cli.cron._warn_if_gateway_not_running"):
        assert cron_create(args) == 0
    assert "Minimum report length: 20 characters" in capsys.readouterr().out
    job = list_jobs(include_disabled=True)[0]
    assert job["min_response_chars"] == 20
    for value in ("10", "-1", "0"):
        args = parser.parse_args(["cron", "edit", job["id"], "--min-response-chars", value])
        assert cron_edit(args) == (1 if value == "-1" else 0)
        printed = capsys.readouterr().out
        if value == "10":
            assert "Minimum report length: 10 characters" in printed
        assert get_job(job["id"])["min_response_chars"] == (10 if value == "-1" else int(value))
    from cron.jobs import create_job, update_job
    for bad in (True, -1, 1.5, "20"):
        with pytest.raises(ValueError, match="min_response_chars"):
            create_job("Give a report", "every 2h", min_response_chars=bad)
        with pytest.raises(ValueError, match="min_response_chars"):
            update_job(job["id"], {"min_response_chars": bad})
    assert get_job(job["id"])["min_response_chars"] == 0
