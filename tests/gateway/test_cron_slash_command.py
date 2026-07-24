"""Tests for the /cron gateway slash command (list + run-now triggers)."""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.slash_commands import (
    GatewaySlashCommandsMixin,
    _resolve_cron_query,
)


def _job(job_id, name, enabled=True, state="scheduled",
         next_run="2026-06-13T08:00:00-07:00"):
    return {
        "id": job_id,
        "name": name,
        "enabled": enabled,
        "state": state,
        "schedule_display": "0 8 * * *",
        "next_run_at": next_run,
    }


JOBS = [
    _job("23a8851cefe8", "XRP Entry Check"),
    _job("4b30153e9959", "XRP Exit Check"),
    _job("5291d8c77c80", "Oil Thesis Audit"),
    _job("9445d0ea1dd9", "Use-Case Scout", enabled=False, state="paused"),
]


class TestResolveCronQuery:
    def test_exact_id_match(self):
        status, job = _resolve_cron_query(JOBS, "23a8851cefe8")
        assert status == "match"
        assert job["name"] == "XRP Entry Check"

    def test_id_prefix_unique(self):
        status, job = _resolve_cron_query(JOBS, "5291")
        assert status == "match"
        assert job["name"] == "Oil Thesis Audit"

    def test_id_prefix_ambiguous(self):
        jobs = [_job("aa11", "One"), _job("aa22", "Two")]
        status, candidates = _resolve_cron_query(jobs, "aa")
        assert status == "ambiguous"
        assert {j["id"] for j in candidates} == {"aa11", "aa22"}

    def test_exact_id_beats_prefix(self):
        jobs = [_job("aa", "Short"), _job("aab", "Long")]
        status, job = _resolve_cron_query(jobs, "aa")
        assert status == "match"
        assert job["name"] == "Short"

    def test_name_substring_unique_case_insensitive(self):
        status, job = _resolve_cron_query(JOBS, "OIL")
        assert status == "match"
        assert job["id"] == "5291d8c77c80"

    def test_name_substring_ambiguous(self):
        status, candidates = _resolve_cron_query(JOBS, "xrp")
        assert status == "ambiguous"
        assert len(candidates) == 2

    def test_no_match(self):
        assert _resolve_cron_query(JOBS, "zzz") == ("none", None)

    def test_empty_query(self):
        assert _resolve_cron_query(JOBS, "  ") == ("none", None)

    def test_none_query(self):
        assert _resolve_cron_query(JOBS, None) == ("none", None)

    def test_empty_jobs_list(self):
        assert _resolve_cron_query([], "abc") == ("none", None)


class _Event:
    """Minimal stand-in for MessageEvent — handler only needs args."""

    def __init__(self, args):
        self._args = args

    def get_command_args(self):
        return self._args


_UNSET = object()


def _run_handler(args, jobs=JOBS, trigger_result=_UNSET):
    """Invoke the unbound handler with patched cron.jobs collaborators."""
    list_jobs = MagicMock(return_value=list(jobs))
    if trigger_result is _UNSET:
        trigger = MagicMock(side_effect=lambda jid: dict(
            next(j for j in jobs if j["id"] == jid),
            enabled=True, state="scheduled",
        ))
    else:
        trigger = MagicMock(return_value=trigger_result)
    with (
        patch("cron.jobs.list_jobs", list_jobs),
        patch("cron.jobs.trigger_job", trigger),
    ):
        reply = asyncio.run(
            GatewaySlashCommandsMixin._handle_cron_command(
                object(), _Event(args)
            )
        )
    return reply, trigger


class TestCronCommandHandler:
    def test_bare_cron_shows_usage(self):
        reply, _ = _run_handler("")
        assert "/cron list" in reply
        assert "/cron run" in reply

    def test_unknown_subcommand_shows_usage(self):
        reply, _ = _run_handler("frobnicate")
        assert "/cron run" in reply

    def test_list_shows_all_jobs_with_ids(self):
        reply, _ = _run_handler("list")
        for j in JOBS:
            assert j["id"] in reply
            assert j["name"] in reply

    def test_list_marks_paused_jobs(self):
        reply, _ = _run_handler("list")
        assert "paused" in reply.lower()

    def test_list_empty(self):
        reply, _ = _run_handler("list", jobs=[])
        assert "no cron jobs" in reply.lower()

    def test_run_unique_match_triggers_by_exact_id(self):
        reply, trigger = _run_handler("run oil")
        trigger.assert_called_once_with("5291d8c77c80")
        assert "Oil Thesis Audit" in reply
        assert "within a minute" in reply

    def test_run_query_may_contain_spaces(self):
        reply, trigger = _run_handler("run xrp entry")
        trigger.assert_called_once_with("23a8851cefe8")

    def test_run_subcommand_case_insensitive(self):
        reply, trigger = _run_handler("RUN Oil")
        trigger.assert_called_once_with("5291d8c77c80")

    def test_run_paused_job_notes_reenable(self):
        reply, trigger = _run_handler("run scout")
        trigger.assert_called_once_with("9445d0ea1dd9")
        assert "re-enabled" in reply

    def test_run_ambiguous_lists_candidates_without_triggering(self):
        reply, trigger = _run_handler("run xrp")
        trigger.assert_not_called()
        assert "XRP Entry Check" in reply
        assert "XRP Exit Check" in reply

    def test_run_no_match(self):
        reply, trigger = _run_handler("run zzz")
        trigger.assert_not_called()
        assert "/cron list" in reply

    def test_run_without_query_shows_usage(self):
        reply, trigger = _run_handler("run")
        trigger.assert_not_called()
        assert "Usage" in reply

    def test_run_trigger_returns_none_reports_failure(self):
        reply, _ = _run_handler("run oil", trigger_result=None)
        assert "Failed to trigger" in reply

    def test_run_errored_job_notes_reschedule(self):
        jobs = JOBS + [_job("e3a1b2c3d4e5", "Broken Job", state="error")]
        reply, trigger = _run_handler("run broken", jobs=jobs)
        trigger.assert_called_once_with("e3a1b2c3d4e5")
        assert "was error — re-scheduled" in reply

    def test_trigger_error_reports_not_silence(self):
        list_jobs = MagicMock(return_value=list(JOBS))
        trigger = MagicMock(side_effect=RuntimeError("disk full"))
        with (
            patch("cron.jobs.list_jobs", list_jobs),
            patch("cron.jobs.trigger_job", trigger),
        ):
            reply = asyncio.run(
                GatewaySlashCommandsMixin._handle_cron_command(
                    object(), _Event("run oil")
                )
            )
        assert "error" in reply.lower()

    def test_jobs_file_error_reports_not_silence(self):
        with patch(
            "cron.jobs.list_jobs",
            MagicMock(side_effect=RuntimeError("corrupt jobs.json")),
        ):
            reply = asyncio.run(
                GatewaySlashCommandsMixin._handle_cron_command(
                    object(), _Event("list")
                )
            )
        assert "error" in reply.lower()


class TestCronRegistration:
    def test_cron_is_gateway_known_command(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "cron" in GATEWAY_KNOWN_COMMANDS

    def test_cron_in_slack_manifest_slashes(self):
        from hermes_cli.commands import slack_native_slashes
        names = [name for name, _desc, _usage in slack_native_slashes()]
        assert "cron" in names

    def test_run_py_dispatches_cron(self):
        # The handler tests cover the handler logic; this guards against
        # the dispatch block being accidentally dropped from run.py.
        import gateway.run
        import pathlib
        src = pathlib.Path(gateway.run.__file__).read_text()
        assert 'canonical == "cron"' in src
        assert "_handle_cron_command" in src
