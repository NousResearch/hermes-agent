"""Cron delivery text must be secret-redacted before it reaches any platform.

Shell-job stdout/stderr is redacted where it is captured, but an LLM cron job's
response text reached ``_deliver_result`` unscanned, so a job that surfaced a
credential in its answer delivered it verbatim to the chat.
"""
from unittest.mock import AsyncMock, MagicMock, patch


def _telegram_cfg():
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
    return mock_cfg


def _job():
    return {
        "id": "report-job",
        "name": "daily-report",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "123"},
    }


# A synthetic OpenAI-style key: long enough to trip the redactor, and not a
# real credential.
FAKE_SECRET = "sk-" + "A" * 32


class TestCronDeliveryRedaction:
    def test_standalone_delivery_redacts_secret(self):
        """A secret in the job's answer must not reach the platform send.

        Redaction happens once on ``cleaned_delivery_content`` right after
        media extraction — i.e. before the live-adapter / standalone branch —
        so both delivery paths consume the same redacted string. This test
        drives the standalone branch; the live-adapter branch reads the very
        same variable.
        """
        from cron.scheduler import _deliver_result

        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("sys.is_finalizing", return_value=False):
            _deliver_result(_job(), f"Job finished. Token was {FAKE_SECRET} (oops).")

        send_mock.assert_called_once()
        delivered = " ".join(str(a) for a in send_mock.call_args.args)
        delivered += " " + " ".join(str(v) for v in send_mock.call_args.kwargs.values())
        assert FAKE_SECRET not in delivered, "secret reached the platform send"

    def test_clean_content_is_unchanged(self):
        """Redaction must not mangle ordinary delivery text."""
        from cron.scheduler import _deliver_result

        body = "Daily report: 3 tasks done, 1 pending. All systems nominal."
        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("sys.is_finalizing", return_value=False):
            result = _deliver_result(_job(), body)

        send_mock.assert_called_once()
        delivered = " ".join(str(a) for a in send_mock.call_args.args)
        assert "3 tasks done" in delivered
        assert result is None

    def _deliver_with_mirror(self, job, content):
        """Drive a delivery with the mirror enabled, through the REAL
        ``_maybe_mirror_cron_delivery``, capturing what reaches the
        ``mirror_to_session`` sink. Mocking the mirror helper itself would hide
        anything the sink splices in around the payload (e.g. the job-name
        prefix), so only the outermost session write is stubbed."""
        from cron.scheduler import _deliver_result

        send_mock = AsyncMock(return_value={"success": True})
        sink_mock = MagicMock(return_value=True)
        with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler._cron_mirror_delivery_enabled", return_value=True), \
             patch("cron.scheduler._target_matches_origin", return_value=True), \
             patch("gateway.mirror.mirror_to_session", new=sink_mock), \
             patch("sys.is_finalizing", return_value=False):
            _deliver_result(job, content)

        assert sink_mock.called, "mirror sink did not run — test would vacuously pass"
        mirrored = " ".join(str(a) for a in sink_mock.call_args.args)
        mirrored += " " + " ".join(str(v) for v in sink_mock.call_args.kwargs.values())
        return mirrored

    def test_session_mirror_payload_redacts_secret(self):
        """The delivery mirror must not write an unredacted secret to the session.

        ``mirror_text`` is derived from the raw ``content``, not from
        ``cleaned_delivery_content``, so it does not inherit the delivery-path
        redaction. With ``cron.mirror_delivery`` enabled the mirror payload is
        appended to the origin chat's session transcript — an unscanned value
        there is just as exposed as one sent to the chat, and survives longer.
        """
        mirrored = self._deliver_with_mirror(
            _job(), f"Job finished. Token was {FAKE_SECRET} (oops).",
        )
        # An empty mirror payload would satisfy the secret assertion for the
        # wrong reason, so pin that the real body still went through.
        assert "Job finished." in mirrored, "mirror payload was empty — assertion below is vacuous"
        assert FAKE_SECRET not in mirrored, "secret reached the session mirror payload"

    def test_session_mirror_job_name_does_not_leak_secret(self):
        """The sink splices the job NAME around the redacted payload.

        The name is user-controlled config; redacting only the body would leave
        ``[Cron delivery: <name-with-secret>]`` re-leaking a credential right
        next to the scrubbed text.
        """
        job = _job()
        job["name"] = f"rotate {FAKE_SECRET} daily"
        mirrored = self._deliver_with_mirror(job, "Job finished. All good.")
        assert "[Cron delivery:" in mirrored, "sink prefix missing — assembly path not exercised"
        assert FAKE_SECRET not in mirrored, "secret in the job name reached the session mirror"

    def test_redaction_survives_disabled_logging_preference(self, monkeypatch):
        """``security.redact_secrets: false`` must not disable delivery redaction.

        That preference governs how much is scrubbed from the user's own logs.
        Delivery is an egress boundary, so it redacts with ``force=True`` — the
        same rule the other safety boundaries in the tree already follow (e.g.
        ``tools/delegation_live_log.py``, ``tools/approval.py``). Without it a
        user who turned down log scrubbing would silently also turn off secret
        scrubbing on the way out to the chat.
        """
        import importlib

        import agent.redact

        monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
        importlib.reload(agent.redact)
        try:
            from cron.scheduler import _deliver_result

            send_mock = AsyncMock(return_value={"success": True})
            with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
                 patch("tools.send_message_tool._send_to_platform", new=send_mock), \
                 patch("sys.is_finalizing", return_value=False):
                _deliver_result(_job(), f"Job finished. Token was {FAKE_SECRET} (oops).")

            send_mock.assert_called_once()
            delivered = " ".join(str(a) for a in send_mock.call_args.args)
            delivered += " " + " ".join(str(v) for v in send_mock.call_args.kwargs.values())
            assert FAKE_SECRET not in delivered, (
                "logging preference disabled delivery redaction — needs force=True"
            )
        finally:
            monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
            importlib.reload(agent.redact)

    def test_redaction_failure_does_not_leak(self):
        """If the redactor itself raises, the delivery must fail closed rather
        than send the unscanned text."""
        from cron.scheduler import _deliver_result

        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("boom")), \
             patch("sys.is_finalizing", return_value=False):
            _deliver_result(_job(), f"Token {FAKE_SECRET}")

        send_mock.assert_called_once()
        delivered = " ".join(str(a) for a in send_mock.call_args.args)
        assert FAKE_SECRET not in delivered
        assert "REDACTED" in delivered
