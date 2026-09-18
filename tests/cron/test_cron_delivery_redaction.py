"""Cron output is secret-redacted on every outward lane, fail-closed.

Shell-job stdout/stderr is redacted where it is captured, but an LLM cron job's response text
reaches ``_deliver_result`` unscanned. Every egress lane — platform send, session mirror (payload
and spliced job name), bot-chat turn — must apply ``redact_sensitive_text(force=True)``: the
``security.redact_secrets`` preference governs the user's own logs, not egress, and a raising
redactor must replace the payload rather than let it through.
"""
import importlib
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Synthetic vendor-prefixed key, long enough to trip the redactor; never a real credential.
FAKE_SECRET = "sk-" + "A" * 32
BODY = "Job finished. 3 tasks done, 1 pending."


def _telegram_cfg():
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    cfg = MagicMock()
    cfg.platforms = {Platform.TELEGRAM: pconfig}
    return cfg


def _job(name: str = "daily-report") -> dict:
    return {"id": "report-job", "name": name, "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"}}


def _flat(call) -> str:
    return " ".join(str(a) for a in call.args) + " " + " ".join(str(v) for v in call.kwargs.values())


def _deliver_platform_and_mirror(job: dict, content: str) -> tuple[str, str]:
    """Drive ``_deliver_result`` with the mirror on; stub only the outermost sinks (the platform
    send and the session write) so the real assembly — including the job-name splice — runs."""
    from cron.scheduler_delivery import _deliver_result

    send = AsyncMock(return_value={"success": True})
    sink = MagicMock(return_value=True)
    with patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()), \
         patch("tools.send_message_tool._send_to_platform", new=send), \
         patch("cron.scheduler_delivery._cron_mirror_delivery_enabled", return_value=True), \
         patch("cron.scheduler_delivery._target_matches_origin", return_value=True), \
         patch("gateway.mirror.mirror_to_session", new=sink), \
         patch("sys.is_finalizing", return_value=False):
        _deliver_result(job, content)
    assert send.called and sink.called, "a sink did not run — assertions would be vacuous"
    return _flat(send.call_args), _flat(sink.call_args)


def _deliver_bot_chat(job: dict, content: str) -> str:
    """Drive ``_deliver_to_bot_chat`` down to the CLI lane and capture the inbound turn."""
    from cron.scheduler_delivery import _deliver_to_bot_chat

    captured = {}

    def _fake_run(argv, *args, **kwargs):
        with open(argv[argv.index("--query-file") + 1], encoding="utf-8") as fh:
            captured["message"] = fh.read()
        return MagicMock(returncode=0, stdout="", stderr="")

    # The CLI lane is seamed at whichever spawn helper the tree has: ``subprocess.run`` or the
    # report-driven ``_run_bot_chat_turn(argv, env, report_path, timeout)`` (#113608); ``create``
    # keeps the second patch a no-op where the helper does not exist, so no real child is spawned.
    with patch("cron.scheduler_delivery.subprocess.run", side_effect=_fake_run), \
            patch("cron.scheduler_delivery._run_bot_chat_turn", create=True, side_effect=_fake_run):
        err = _deliver_to_bot_chat(job, content, "")
    assert err is None, err
    return captured["message"]


@pytest.mark.parametrize("lane", ["platform_send", "session_mirror", "bot_chat"])
def test_every_outward_lane_masks_secret_in_payload_and_job_name(lane):
    """Secret in the body AND in the user-controlled job name is masked on every lane, while
    ordinary text and the lane's own framing survive (no over-redaction, no empty payload)."""
    job = _job(name=f"rotate {FAKE_SECRET} daily")
    content = f"{BODY} Token was {FAKE_SECRET} (oops)."
    if lane == "bot_chat":
        out = _deliver_bot_chat(job, content)
        assert "[Cronjob " in out
    else:
        sent, mirrored = _deliver_platform_and_mirror(job, content)
        out = sent if lane == "platform_send" else mirrored
        if lane == "session_mirror":
            assert "[Cron delivery:" in out, "mirror prefix missing — assembly path not exercised"
    assert "3 tasks done" in out, "payload mangled or empty — secret assertion would be vacuous"
    assert FAKE_SECRET not in out, f"secret reached the {lane} lane"


@pytest.mark.parametrize("mode", ["redact_secrets_off", "redactor_raises"])
def test_delivery_redaction_is_forced_and_fails_closed(mode, monkeypatch):
    """Egress redaction is independent of ``security.redact_secrets`` (force=True) and a raising
    redactor replaces the payload instead of passing the unscanned text through."""
    import agent.redact
    from cron.scheduler_delivery import _deliver_result

    send = AsyncMock(return_value={"success": True})
    try:
        with ExitStack() as stack:
            stack.enter_context(patch("gateway.config.load_gateway_config", return_value=_telegram_cfg()))
            stack.enter_context(patch("tools.send_message_tool._send_to_platform", new=send))
            stack.enter_context(patch("sys.is_finalizing", return_value=False))
            if mode == "redact_secrets_off":
                monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
                importlib.reload(agent.redact)
            else:
                stack.enter_context(
                    patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("boom")))
            _deliver_result(_job(), f"Token was {FAKE_SECRET}")
    finally:
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        importlib.reload(agent.redact)

    delivered = _flat(send.call_args)
    assert FAKE_SECRET not in delivered, f"{mode}: secret reached the platform send"
    if mode == "redactor_raises":
        assert "REDACTED" in delivered
