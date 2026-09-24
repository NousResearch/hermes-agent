"""Slack driver for the adapter contract: binds the shared contract to ``SlackStandin``.

Same surface as ``_drv_telegram.TelegramDriver`` (see its docstring). The child gateway runs the real
``plugins/platforms/slack`` adapter (slack_bolt Socket Mode + slack_sdk Web API); the
``slack_shim/sitecustomize.py`` module on the child's ``PYTHONPATH`` points slack_sdk at the stand-in
(``HERMES_STANDIN_SLACK_API``). Chat ids are Slack channel ids: DMs are ``D<user>``, the group is a
public channel ``C…``. Message ids are Slack ``ts`` strings.

Known limits: ``document()`` delivers a real ``file_share`` event, but the adapter (by design) only
downloads ``https://*.slack.com`` URLs that pass the SSRF guard, so the stand-in's 127.0.0.1 file URL
is refused (logged "Blocked unsafe Slack file URL") and the turn runs on the caption alone. Replies are
threaded under the inbound ts (``reply_in_thread`` default), including DMs. ``sends()`` counts
``chat.startStream`` too: with ``streaming.enabled`` the adapter streams natively
(startStream/appendStream/stopStream) instead of postMessage + chat.update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tests.fakes.platforms._standin import Call, Fault, Visible
from tests.fakes.platforms.slack_standin import SlackStandin

_SHIM = Path(__file__).resolve().parents[3] / "fakes" / "platforms" / "slack_shim"


@dataclass
class Inbound:
    chat_id: str
    message_id: str
    raw: Any


class SlackDriver:
    name = "slack"
    limit = 39_000  # SlackAdapter.MAX_MESSAGE_LENGTH (Slack rejects > 40,000 with msg_too_long)
    user_id = "U0000111"
    other_user_id = "U0000222"
    stream_users = tuple(f"U0000{i}" for i in range(301, 307))
    group_chat_id = "C0000123"
    home_channel = "C0000999"

    def __init__(self) -> None:
        self.standin = SlackStandin()

    def start(self) -> None:
        self.standin.start()

    def stop(self) -> None:
        self.standin.stop()

    def gateway_config(self) -> Dict[str, Any]:
        return {"platforms": {"slack": {"enabled": True, "extra": {"require_mention": True}}}}

    def gateway_env(self) -> Dict[str, str]:
        return {"SLACK_BOT_TOKEN": self.standin.bot_token, "SLACK_APP_TOKEN": self.standin.app_token,
                "SLACK_ALLOWED_USERS": ",".join((self.user_id, *self.stream_users)), "SLACK_HOME_CHANNEL": self.home_channel,
                "HERMES_STANDIN_SLACK_API": self.standin.api_base, "PYTHONPATH_PREPEND": str(_SHIM)}

    def connected(self) -> bool:
        # auth.test proves the shim redirected the SDK; an open socket proves Socket Mode is live.
        return bool(self.standin.calls_of("auth.test")) and self.standin.socket_count() > 0

    # inbound -----------------------------------------------------------------------------------
    @staticmethod
    def _wrap(envelope: Dict[str, Any]) -> Inbound:
        event = envelope["payload"]["event"]
        return Inbound(event["channel"], event["ts"], envelope)

    def dm(self, text: str, user_id: Optional[str] = None) -> Inbound:
        return self._wrap(self.standin.dm(user_id or self.user_id, text))

    def group(self, text: str, *, mention: bool) -> Inbound:
        envelopes = self.standin.channel_post(self.group_chat_id, self.user_id, text, mention=mention)
        return self._wrap(envelopes[0])

    def redeliver(self, inbound: Inbound) -> None:
        self.standin.redeliver(inbound.raw)

    def document(self, filename: str, data: bytes, mime: str, caption: str = "") -> Inbound:
        return self._wrap(self.standin.dm_file(self.user_id, filename, data, mime, caption))

    def buttons(self, chat_id: str) -> List[Dict[str, Any]]:
        return self.standin.buttons(chat_id)

    def click(self, chat_id: str, button: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.standin.block_action(user_id or self.user_id, chat_id, str(button["message_id"]), button)

    def click_answered(self, handle: Dict[str, Any]) -> bool:
        """Bolt acks an interactive envelope first, then authorizes the clicker synchronously; a
        refused click is only logged, so the ack is the last platform-visible sign of it."""
        return self.standin.acked(handle)

    def slash(self, command: str, text: str = "", chat_id: Optional[str] = None) -> Dict[str, Any]:
        return self.standin.slash(self.user_id, chat_id or self.standin.dm_channel(self.user_id), command, text)

    def callback_answers(self) -> List[Call]:
        """Socket Mode acks for interactive envelopes (Slack's analogue of answerCallbackQuery)."""
        ids = {e["envelope_id"] for e in self.standin.envelopes if e["type"] == "interactive"}
        return [c for c in self.standin.calls_of("socket_ack") if c.params.get("envelope_id") in ids]

    # ground truth ------------------------------------------------------------------------------
    def visible(self, chat_id: str) -> List[Visible]:
        return self.standin.visible(chat_id)

    def ephemerals(self, chat_id: str) -> List[Dict[str, Any]]:
        """Replies only the invoking user sees (slash ``response_url`` POSTs, chat.postEphemeral)."""
        return self.standin.ephemerals(chat_id)

    def _ok(self, methods: tuple, chat_id: str) -> List[Call]:
        return [c for c in self.standin.calls_of(*methods)
                if not c.faulted and str(c.params.get("channel")) == str(chat_id)]

    def sends(self, chat_id: str) -> List[Call]:
        return self._ok(("chat.postMessage", "chat.startStream"), chat_id)

    def edits(self, chat_id: str) -> List[Call]:
        return self._ok(("chat.update", "chat.appendStream"), chat_id)

    def format_rejections(self, chat_id: str) -> List[Call]:
        return []  # mrkdwn never fails to parse; Slack renders what it cannot format as text

    def describe(self) -> str:
        return self.standin.describe()

    # faults ------------------------------------------------------------------------------------
    def fail_send(self, *, times: int = 1, match: Optional[Callable[[str], bool]] = None) -> List[Fault]:
        pred = (lambda p: match(str(p.get("text", "")))) if match else None
        return [self.standin.fail("chat.postMessage", {"ok": False, "error": "channel_not_found"}, times=times,
                                  match=pred)]

    def fail_edit(self, *, times: int = 1, match: Optional[Callable[[str], bool]] = None) -> List[Fault]:
        pred = (lambda p: match(str(p.get("text", "")))) if match else None
        return [self.standin.fail("chat.update", {"ok": False, "error": "cant_update_message"}, times=times,
                                  match=pred)]

    def fail_finalize(self, has_footer: Callable[[str], bool], *, group: bool) -> List[Fault]:
        """Native streaming (startStream/appendStream/stopStream) is the transport in DMs and channels:
        refuse the stream call whose resulting text completes the reply (``_stream_text`` = what the
        message would read after it), however the deltas were cut."""
        pred = lambda p: has_footer(str(p.get("_stream_text", "")))  # noqa: E731
        return [self.standin.fail(m, {"ok": False, "error": "message_not_in_streaming_state"}, times=50, match=pred)
                for m in ("chat.appendStream", "chat.stopStream")]
