"""The PUBLIC debug paste is an allowlisted summary; full logs only reach private storage (#46006).

Subtractive redaction scrubs credentials but not IPs, room IDs, display names or message bodies,
so every surface that publishes to the public paste (``hermes debug share``, the ``/debug`` gateway
command, the dashboard ``POST /api/ops/debug-share``) must upload no log line text at all. The
private ``--nous`` bundle still carries the redacted tails.
"""

import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# PII that subtractive redaction does not recognise (message text, names, IDs, IPs, paths).
_MESSAGE = "please wire 4000 to my landlord at 12 Elm Street"
_DISPLAY_NAME = "Alice Wonderland"
_ROOM_ID = "!roomXYZ123:matrix.example.org"
_IP = "203.0.113.77"
_HOME_PATH = "/home/alice/secret-project"
_PII = (_MESSAGE, _DISPLAY_NAME, _ROOM_ID, _IP, _HOME_PATH, "landlord", "Elm Street")


@pytest.fixture
def pii_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    logs = home / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  default: secret-model-name\n  provider: openrouter\n"
        f"terminal:\n  cwd: {_HOME_PATH}\n"
        "custom_providers:\n  - name: alice-private-endpoint\n"
        "toolsets: [web, alice-custom-toolset]\n"
        "alice_unknown_key: hello\n")
    (logs / "agent.log").write_text(
        f"2026-04-12 17:00:00,001 INFO [sess-1] gateway.platforms.matrix: inbound from "
        f"{_DISPLAY_NAME} in {_ROOM_ID}: {_MESSAGE}\n"
        f"2026-04-12 17:00:01,002 WARNING [sess-1] agent.tool_executor: Tool terminal returned "
        f"error: cd {_HOME_PATH} failed for {_DISPLAY_NAME}\n"
        f"2026-04-12 17:00:02,003 ERROR [sess-1] agent.conversation_loop: API call failed: "
        f"openai.RateLimitError: HTTP 429 from {_IP} while answering '{_MESSAGE}'\n"
        "Traceback (most recent call last):\n"
        f'  File "{_HOME_PATH}/x.py", line 1, in <module>\n'
        f"openai.RateLimitError: Error code: 429 - quota for {_DISPLAY_NAME}\n")
    (logs / "errors.log").write_text(
        f"2026-04-12 17:00:02,003 ERROR gateway.run: ConnectionResetError talking to {_IP}\n")
    (logs / "gateway.log").write_text(
        f"2026-04-12 17:00:03,004 INFO gateway.run: {_DISPLAY_NAME} said {_MESSAGE}\n")
    return home


def _assert_no_pii(text: str) -> None:
    for needle in _PII:
        assert needle not in text, f"public paste leaked {needle!r}"
    for log_line in ("inbound from", "Tool terminal returned", "API call failed", "Traceback"):
        assert log_line not in text, f"public paste carries log line text: {log_line!r}"


def _capture_uploads():
    uploaded: list[str] = []

    def _upload(content, expiry_days=1):
        uploaded.append(content)
        return f"https://paste.rs/p{len(uploaded)}"

    return uploaded, _upload


def test_cli_public_share_uploads_only_the_allowlisted_summary(pii_home, capsys):
    from hermes_cli.debug import run_debug_share

    uploaded, upload = _capture_uploads()
    with patch("hermes_cli.debug.upload_to_pastebin", side_effect=upload), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        # --no-redact must not be able to put logs on the public paste either.
        for no_redact in (False, True):
            run_debug_share(SimpleNamespace(lines=200, expire=1, local=False, nous=False,
                                            yes=True, no_redact=no_redact))

    assert len(uploaded) == 2  # one summary paste per share, no full-log pastes
    for text in uploaded:
        _assert_no_pii(text)
        # The useful, allowlisted signal is still there.
        assert "RateLimitError" in text and "ConnectionResetError" in text
        assert "429" in text
        assert "agent.tool_executor" in text
        assert "terminal" in text and "model" in text  # config key names
        assert "secret-model-name" not in text and "alice-custom-toolset" not in text
        assert "alice_unknown_key" not in text and "alice-private-endpoint" not in text


def test_dashboard_builder_uploads_only_the_summary(pii_home):
    from hermes_cli.debug import build_debug_share

    uploaded, upload = _capture_uploads()
    with patch("hermes_cli.debug.upload_to_pastebin", side_effect=upload), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        result = build_debug_share(log_lines=200, redact=False)

    assert list(result.urls) == ["Report"]
    assert len(uploaded) == 1
    _assert_no_pii(uploaded[0])


@pytest.mark.asyncio
async def test_gateway_debug_uploads_only_the_summary(pii_home):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}

    async def _inline(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = _inline
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    uploaded: list[str] = []
    with patch("hermes_cli.debug.upload_to_pastebin",
               lambda report: uploaded.append(report) or "https://paste.rs/x"), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        await runner._handle_debug_command(MessageEvent(text="/debug", source=source))

    assert len(uploaded) == 1
    _assert_no_pii(uploaded[0])
    assert "RateLimitError" in uploaded[0]


def test_nous_share_still_carries_redacted_log_tails(pii_home):
    from hermes_cli.debug import run_debug_share

    with patch("hermes_cli.diagnostics_upload.share_to_nous",
               return_value={"id": "x", "viewUrl": "https://v"}) as share, \
            patch("hermes_cli.debug.upload_to_pastebin") as paste, \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        run_debug_share(SimpleNamespace(lines=200, expire=1, local=False, nous=True,
                                        yes=True, no_redact=False))

    paste.assert_not_called()
    envelope = json.loads(gzip.decompress(share.call_args[0][0]))
    assert _MESSAGE in envelope["files"]["agent.log"]
    assert _MESSAGE in envelope["files"]["report"]
