"""Discord driver for the adapter contract: binds the shared contract to ``DiscordStandin``.

Same surface as ``_drv_telegram.TelegramDriver``. Discord specifics:

* a DM lives in its own DM channel (id != user id). ``dm()`` returns ``Inbound.chat_id`` = that DM
  channel id; ``visible()/sends()/edits()/buttons()`` also accept the user id and map it to the DM.
* the child gateway reaches the stand-in through ``tests/fakes/platforms/discord_shim`` (a
  ``sitecustomize`` on ``PYTHONPATH`` that repoints discord.py's ``Route.BASE``/``DEFAULT_GATEWAY``).
* ``connected()`` is true once the adapter finished ``on_ready`` post-connect work: the bulk slash
  command sync (``PUT /applications/{id}/commands``) only runs after ``_ready_event`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tests.fakes.platforms._standin import Call, Fault, Visible
from tests.fakes.platforms.discord_standin import BOT_ID, MAX_TEXT, DiscordStandin

SHIM_DIR = Path(__file__).resolve().parents[3] / "fakes" / "platforms" / "discord_shim"


@dataclass
class Inbound:
    chat_id: str
    message_id: str
    raw: Any


class DiscordDriver:
    name = "discord"
    limit = MAX_TEXT
    bot_id = BOT_ID
    user_id = "200000000000000111"
    other_user_id = "200000000000000222"
    stream_users = tuple(f"2000000000000003{i:02d}" for i in range(1, 7))
    group_chat_id = "1300000000000000001"  # #general in the stand-in guild
    home_channel = "1300000000000000002"  # #home in the stand-in guild

    def __init__(self) -> None:
        self.standin = DiscordStandin()
        self.standin.add_text_channel(self.group_chat_id, "general")
        self.standin.add_text_channel(self.home_channel, "home")
        for uid in (self.user_id, self.other_user_id, *self.stream_users):
            self.standin.user(uid)["_in_guild"] = True

    def start(self) -> None:
        self.standin.start()

    def stop(self) -> None:
        self.standin.stop()

    def gateway_config(self) -> Dict[str, Any]:
        return {"platforms": {"discord": {"enabled": True, "extra": {
            "require_mention": True,
            # replies land inline in the group channel (contract parity); threads are implemented too
            "auto_thread": False,
            # history backfill deliberately feeds recent unmentioned chatter into a mentioned turn as
            # context; off here so the require-mention contract measures the trigger gate alone
            "history_backfill": False,
        }}}}

    def gateway_env(self) -> Dict[str, str]:
        return {
            "DISCORD_BOT_TOKEN": self.standin.token, "DISCORD_ALLOWED_USERS": ",".join((self.user_id, *self.stream_users)),
            "DISCORD_HOME_CHANNEL": self.home_channel,
            # one PUT instead of ~100 paced POSTs (safe policy sleeps 4.5 s per mutation)
            "DISCORD_COMMAND_SYNC_POLICY": "bulk",
            # no text-batch debounce: one inbound is one turn, immediately
            "DISCORD_TEXT_BATCH_DELAY_SECONDS": "0", "DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS": "0",
            "HERMES_STANDIN_DISCORD_API": self.standin.api_base,
            "HERMES_STANDIN_DISCORD_GATEWAY": self.standin.gateway_url,
            "PYTHONPATH_PREPEND": str(SHIM_DIR),
        }

    def connected(self) -> bool:
        return bool(self.standin.sessions() and self.standin.calls_of("bulk_commands"))

    # inbound -----------------------------------------------------------------------------------
    def _chat(self, chat_id: Any) -> str:
        return self.standin.dm_of_user.get(str(chat_id), str(chat_id))

    @staticmethod
    def _wrap(payload: Dict[str, Any]) -> Inbound:
        return Inbound(str(payload["channel_id"]), str(payload["id"]), payload)

    def dm(self, text: str, user_id: Optional[str] = None) -> Inbound:
        uid = str(user_id or self.user_id)
        channel = self.standin.dm_channel(uid)
        return self._wrap(self.standin.inbound_message(channel["id"], uid, text))

    def group(self, text: str, *, mention: bool) -> Inbound:
        content, mentions = (f"<@{BOT_ID}> {text}", [BOT_ID]) if mention else (text, [])
        return self._wrap(self.standin.inbound_message(self.group_chat_id, self.user_id, content, mentions=mentions))

    def redeliver(self, inbound: Inbound) -> None:
        self.standin.redeliver(inbound.raw)

    def document(self, filename: str, data: bytes, mime: str, caption: str = "") -> Inbound:
        channel = self.standin.dm_channel(self.user_id)
        att = self.standin.attachment(filename, data, mime)
        return self._wrap(self.standin.inbound_message(channel["id"], self.user_id, caption, attachments=[att]))

    def buttons(self, chat_id: str) -> List[Dict[str, Any]]:
        return self.standin.buttons(self._chat(chat_id))

    def click(self, chat_id: str, button: Dict[str, Any], user_id: Optional[str] = None) -> str:
        inter = self.standin.click(str(user_id or self.user_id), self._chat(chat_id), str(button["message_id"]),
                                   button["custom_id"], component_type=int(button.get("type") or 2))
        return str(inter["id"])

    def click_answered(self, handle: str) -> bool:
        return any(str(c.params.get("interaction_id")) == handle and not c.faulted for c in self.callback_answers())

    def callback_answers(self) -> List[Call]:
        return self.standin.calls_of("interaction_callback")

    # ground truth ------------------------------------------------------------------------------
    def visible(self, chat_id: str) -> List[Visible]:
        return self.standin.visible(self._chat(chat_id))

    def _ok(self, methods: tuple, chat_id: str) -> List[Call]:
        chat = self._chat(chat_id)
        return [c for c in self.standin.calls_of(*methods)
                if not c.faulted and str(c.params.get("channel_id")) == chat]

    def sends(self, chat_id: str) -> List[Call]:
        return self._ok(("create_message",), chat_id)

    def edits(self, chat_id: str) -> List[Call]:
        return self._ok(("edit_message",), chat_id)

    def format_rejections(self, chat_id: str) -> List[Call]:
        return []  # Discord renders markdown client-side: nothing to reject for formatting

    def describe(self) -> str:
        return self.standin.describe()

    # faults ------------------------------------------------------------------------------------
    def fail_send(self, *, times: int = 1, match: Optional[Callable[[str], bool]] = None) -> List[Fault]:
        pred = (lambda p: match(str(p.get("content", "")))) if match else None
        return [self.standin.fail("create_message", {"message": "Missing Access", "code": 50001},
                                  status=403, times=times, match=pred)]

    def fail_edit(self, *, times: int = 1, match: Optional[Callable[[str], bool]] = None) -> List[Fault]:
        pred = (lambda p: match(str(p.get("content", "")))) if match else None
        return [self.standin.fail("edit_message", {"message": "Unknown Message", "code": 10008}, status=404,
                                  times=times, match=pred)]

    def fail_finalize(self, has_footer: Callable[[str], bool], *, group: bool) -> List[Fault]:
        """Streaming edits one message in place (DMs and channels alike): every edit carrying the footer
        is refused, so the finalize edit never lands."""
        return self.fail_edit(times=50, match=has_footer)
