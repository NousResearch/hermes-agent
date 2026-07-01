"""Email adapter integration tests for loop guard."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.email import EmailAdapter


def _email_env() -> dict[str, str]:
    return {
        "EMAIL_ADDRESS": "alfred@sqmnet.es",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.invalid",
        "EMAIL_SMTP_HOST": "smtp.test.invalid",
        "EMAIL_SMTP_PORT": "587",
        "EMAIL_POLL_INTERVAL": "15",
        "EMAIL_ALLOWED_USERS": "selina@sqmnet.es,miquel@sqmnet.es",
    }


def _adapter(tmp_path: Path, *, mode: str = "protect") -> EmailAdapter:
    extra = {
        "loop_guard": {
            "enabled": True,
            "mode": mode,
            "ai_enabled": False,
            "state_path": str(tmp_path / "email-loop-state.json"),
            "agent_identities": ["alfred@sqmnet.es", "selina@sqmnet.es"],
            "quarantine_ttl_seconds": 120,
        }
    }
    with patch.dict(os.environ, _email_env(), clear=False):
        return EmailAdapter(PlatformConfig(enabled=True, extra=extra))


def _msg(body: str, *, sender: str = "selina@sqmnet.es", subject: str = "Re: Hermes Agent") -> dict:
    return {
        "uid": b"42",
        "sender_addr": sender,
        "sender_name": sender,
        "subject": subject,
        "message_id": "<loop-msg@sqmnet.es>",
        "in_reply_to": "<prev@sqmnet.es>",
        "body": body,
        "attachments": [],
        "date": "",
        "headers": {},
    }


class TestEmailLoopGuard(unittest.TestCase):
    def test_email_pre_dispatch_suppresses_agent_meta_restart_notice(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td))
            adapter.handle_message = AsyncMock()

            asyncio.run(
                adapter._dispatch_message(
                    _msg("♻ Gateway restarted successfully. Your session continues.")
                )
            )

            adapter.handle_message.assert_not_called()
            self.assertNotIn("selina@sqmnet.es", adapter._thread_context)

    def test_email_pre_send_suppresses_agent_to_agent_meta_reply(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td))
            adapter._send_email = MagicMock(side_effect=AssertionError("SMTP should not be called"))

            result = asyncio.run(
                adapter.send(
                    "selina@sqmnet.es",
                    "♻ Gateway restarted successfully. Your session continues.",
                )
            )

            self.assertTrue(result.success)
            self.assertIsNone(result.message_id)
            self.assertTrue(result.raw_response["loop_guard"]["suppressed"])
            adapter._send_email.assert_not_called()

    def test_email_pre_send_blocks_next_hop_at_limit(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td))
            adapter._thread_context["selina@sqmnet.es"] = {
                "subject": "Re: Work handoff",
                "message_id": "<peer@sqmnet.es>",
                "hermes_hop_count": "3",
            }
            adapter._send_email = MagicMock(side_effect=AssertionError("SMTP should not be called"))

            result = asyncio.run(
                adapter.send(
                    "selina@sqmnet.es",
                    "Here is a substantive update that would otherwise be allowed.",
                )
            )

            self.assertTrue(result.success)
            self.assertIsNone(result.message_id)
            self.assertEqual(result.raw_response["loop_guard"]["decision"]["category"], "agent_agent_loop")
            adapter._send_email.assert_not_called()

    def test_loop_suppressed_raw_email_does_not_cache_attachments(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td))
            adapter.handle_message = AsyncMock()
            raw = MIMEMultipart()
            raw.attach(MIMEText("♻ Gateway restarted successfully. Your session continues.", "plain", "utf-8"))
            part = MIMEBase("image", "jpeg")
            part.set_payload(b"fake image bytes")
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment; filename=loop.jpg")
            raw.attach(part)
            msg_data = _msg("♻ Gateway restarted successfully. Your session continues.")
            msg_data["attachments"] = None
            msg_data["raw_email"] = raw.as_bytes()

            with patch("gateway.platforms.email.cache_image_from_bytes") as cache_image:
                asyncio.run(adapter._dispatch_message(msg_data))

            adapter.handle_message.assert_not_called()
            cache_image.assert_not_called()

    def test_email_observe_mode_logs_but_allows_agent_to_agent_message(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td), mode="observe")
            adapter.handle_message = AsyncMock()

            asyncio.run(
                adapter._dispatch_message(
                    _msg("♻ Gateway restarted successfully. Your session continues.")
                )
            )

            adapter.handle_message.assert_awaited_once()

    def test_email_outbound_headers_mark_hermes_messages(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = _adapter(Path(td))
            adapter._thread_context["miquel@sqmnet.es"] = {
                "subject": "Consulta",
                "message_id": "<human@sqmnet.es>",
            }

            with patch("smtplib.SMTP") as mock_smtp:
                server = MagicMock()
                mock_smtp.return_value = server
                msg_id = adapter._send_email(
                    "miquel@sqmnet.es",
                    "Respuesta sustantiva para un humano.",
                    None,
                    metadata={"hermes_intent": "assistant_reply"},
                )

            self.assertTrue(msg_id.startswith("<hermes-"))
            sent_msg = server.send_message.call_args[0][0]
            self.assertEqual(sent_msg["X-Hermes-Origin-Agent"], "alfred")
            self.assertEqual(sent_msg["X-Hermes-Origin-Address"], "alfred@sqmnet.es")
            self.assertEqual(sent_msg["X-Hermes-Intent"], "assistant_reply")
            self.assertEqual(sent_msg["X-Hermes-Hop-Count"], "1")
            self.assertEqual(sent_msg["X-Hermes-Reply-Policy"], "human-or-allow-once")


if __name__ == "__main__":
    unittest.main()
