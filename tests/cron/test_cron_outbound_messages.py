"""HERMES-022: job-scoped native outbound messages for cron."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from cron.outbound import (
    claim_or_reuse,
    classify_send_result,
    is_cron_messaging_session,
    job_allows_messaging,
    mark_result,
)
from cron.scheduler import _build_job_prompt, _resolve_cron_disabled_toolsets
from cron.jobs import create_job
from tools.send_message_tool import send_message_tool


@pytest.fixture
def tmp_outbound(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.outbound.OUTBOUND_FILE", tmp_path / "cron" / "outbound.db")
    return tmp_path


class TestJobOptIn:
    def test_default_job_keeps_messaging_disabled(self):
        disabled = _resolve_cron_disabled_toolsets({}, {"id": "plain"})
        assert "messaging" in disabled

    def test_allow_messaging_removes_only_that_jobs_denylist(self):
        opted_in = _resolve_cron_disabled_toolsets(
            {},
            {"id": "opted", "allow_messaging": True},
        )
        default = _resolve_cron_disabled_toolsets({}, {"id": "plain"})
        assert "messaging" not in opted_in
        assert "messaging" in default
        assert "cronjob" in opted_in
        assert "clarify" in opted_in

    def test_profile_denylist_still_blocks_opt_in(self):
        disabled = _resolve_cron_disabled_toolsets(
            {"id": "opted", "allow_messaging": True},
            {"agent": {"disabled_toolsets": ["messaging"]}},
        )
        assert "messaging" in disabled

    def test_prompt_changes_only_for_opted_in_jobs(self):
        opted = _build_job_prompt({"id": "opted", "allow_messaging": True, "prompt": "do work"})
        default = _build_job_prompt({"id": "plain", "prompt": "do work"})
        assert "target='origin'" in opted
        assert "do NOT use send_message" in default
        assert "do NOT use send_message" not in opted


class TestOutboundLedger:
    def test_two_keys_are_distinct(self, tmp_outbound):
        first = claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:one",
            target="origin",
            body="first",
            platform="telegram",
            chat_id="2027045491",
            thread_id="104992",
        )
        second = claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="manual-review:one",
            target="origin",
            body="second",
            platform="telegram",
            chat_id="2027045491",
            thread_id="104992",
        )
        assert first["action"] == "claim"
        assert second["action"] == "claim"
        assert first["record"]["message_key"] != second["record"]["message_key"]

    def test_retry_reuses_verified_message(self, tmp_outbound):
        claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:one",
            target="origin",
            body="hello",
            platform="telegram",
            chat_id="2027045491",
            thread_id=None,
        )
        mark_result(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:one",
            status="verified",
            transport_message_id="131192",
        )
        reused = claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:one",
            target="origin",
            body="hello",
            platform="telegram",
            chat_id="2027045491",
            thread_id=None,
        )
        assert reused["action"] == "reuse"
        assert reused["record"]["status"] == "verified"
        assert reused["record"]["transport_message_id"] == "131192"

    def test_terminal_result_cannot_be_overwritten(self, tmp_outbound):
        claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:immutable",
            target="origin",
            body="hello",
            platform="telegram",
            chat_id="2027045491",
            thread_id=None,
        )
        mark_result(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:immutable",
            status="verified",
            transport_message_id="131192",
        )

        record = mark_result(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:immutable",
            status="failed",
            error="late failure",
        )

        assert record["status"] == "verified"
        assert record["transport_message_id"] == "131192"
        assert record["error"] is None

    def test_confirmed_failure_can_be_retried(self, tmp_outbound):
        params = {
            "job_id": "job-1",
            "run_id": "run-1",
            "message_key": "automatic-action:retry",
            "target": "origin",
            "body": "hello",
            "platform": "telegram",
            "chat_id": "2027045491",
            "thread_id": None,
        }
        claim_or_reuse(**params)
        mark_result(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:retry",
            status="failed",
            error="confirmed pre-send failure",
        )
        retried = claim_or_reuse(**params)
        assert retried["action"] == "claim"
        assert retried["record"]["status"] == "queued"
        assert retried["record"]["error"] is None

    def test_same_key_different_body_fails_closed(self, tmp_outbound):
        claim_or_reuse(
            job_id="job-1",
            run_id="run-1",
            message_key="automatic-action:one",
            target="origin",
            body="hello",
            platform="telegram",
            chat_id="2027045491",
            thread_id=None,
        )
        with pytest.raises(ValueError, match="different body or target"):
            claim_or_reuse(
                job_id="job-1",
                run_id="run-1",
                message_key="automatic-action:one",
                target="origin",
                body="changed",
                platform="telegram",
                chat_id="2027045491",
                thread_id=None,
            )

    def test_classify_unconfirmed_result_is_ambiguous(self):
        classified = classify_send_result({"ok": True})
        assert classified["status"] == "ambiguous"


class TestSendGate:
    def _bind_cron(self, monkeypatch, *, allow=True):
        from gateway.session_context import _VAR_MAP

        _VAR_MAP["HERMES_CRON_SESSION"].set("1")
        _VAR_MAP["HERMES_CRON_ALLOW_MESSAGING"].set("1" if allow else "")
        _VAR_MAP["HERMES_CRON_JOB_ID"].set("job-1")
        _VAR_MAP["HERMES_CRON_RUN_ID"].set("run-1")
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set("telegram")
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set("2027045491")
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set("104992")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    def test_non_origin_target_is_rejected(self, tmp_outbound, monkeypatch):
        self._bind_cron(monkeypatch)
        raw = send_message_tool({
            "target": "telegram:8868177922",
            "message": "payroll",
            "message_key": "manual-review:sharon",
        })
        payload = json.loads(raw)
        assert payload.get("error")
        assert "origin" in payload["error"]

    def test_account_override_cannot_be_selected(self, tmp_outbound, monkeypatch):
        self._bind_cron(monkeypatch)
        raw = send_message_tool({
            "target": "origin",
            "message": "payroll",
            "message_key": "manual-review:sharon",
            "account": "brianle",
        })
        with patch(
            "tools.send_message_tool._handle_send",
            return_value=json.dumps({"success": True, "message_id": "131192"}),
        ) as send_mock:
            payload = json.loads(send_message_tool({
                "target": "origin",
                "message": "payroll",
                "message_key": "manual-review:sharon-2",
                "account": "brianle",
            }))
        send_mock.assert_called_once()
        sent_args = send_mock.call_args[0][0]
        assert sent_args["target"] == "telegram:2027045491:104992"
        assert "account" not in sent_args
        assert payload["status"] == "verified"

    def test_two_native_sends_are_separate(self, tmp_outbound, monkeypatch):
        self._bind_cron(monkeypatch)
        with patch(
            "tools.send_message_tool._handle_send",
            side_effect=[
                json.dumps({"success": True, "message_id": "1"}),
                json.dumps({"success": True, "message_id": "2"}),
            ],
        ) as send_mock:
            first = json.loads(send_message_tool({
                "target": "origin",
                "message": "action done",
                "message_key": "automatic-action:one",
            }))
            second = json.loads(send_message_tool({
                "target": "origin",
                "message": "review this",
                "message_key": "manual-review:one",
            }))
        assert send_mock.call_count == 2
        assert first["message_id"] == "1"
        assert second["message_id"] == "2"

    def test_send_exception_is_recorded_as_ambiguous(self, tmp_outbound, monkeypatch):
        self._bind_cron(monkeypatch)
        with patch(
            "tools.send_message_tool._handle_send",
            side_effect=RuntimeError("secret provider detail"),
        ):
            payload = json.loads(send_message_tool({
                "target": "origin",
                "message": "action done",
                "message_key": "automatic-action:exception",
            }))
        assert payload["success"] is False
        assert payload["status"] == "ambiguous"
        assert payload["error"] == "send engine raised RuntimeError"
        assert "secret provider detail" not in json.dumps(payload)

    def test_job_without_opt_in_cannot_send(self, tmp_outbound, monkeypatch):
        self._bind_cron(monkeypatch, allow=False)
        raw = send_message_tool({
            "target": "origin",
            "message": "payroll",
            "message_key": "manual-review:sharon",
        })
        payload = json.loads(raw)
        assert "not opted in" in payload.get("error", "")

    def test_create_job_persists_allow_messaging(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        job = create_job(
            prompt="monitor",
            schedule="every 15m",
            allow_messaging=True,
        )
        assert job_allows_messaging(job)
        assert job["allow_messaging"] is True
        default = create_job(prompt="other", schedule="every 15m")
        assert default["allow_messaging"] is False
        assert not is_cron_messaging_session()
