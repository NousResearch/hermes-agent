"""Telegram driver for the adapter contract: binds the shared contract to ``TelegramStandin``.

Every adapter driver exposes the same surface (see ``_contract.py``):

* ``name``, ``limit`` (the platform text cap the adapter must split at), ``user_id`` (allowlisted)
* ``start()/stop()`` the stand-in, ``gateway_config()``/``gateway_env()`` for the child, ``connected()``
* inbound: ``dm(text)``, ``group(text, mention=)``, ``redeliver(inbound)``, ``document(...)``,
  ``click(inbound_chat, button)``
* ground truth: ``visible(chat_id)`` (bot messages a human sees now), ``sends(chat_id)`` /
  ``edits(chat_id)`` (successful outbound create/edit calls), ``describe()``
* faults: ``fail_send(match=, times=)`` / ``fail_edit(times=)`` make the platform reject the call
  with its documented error body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tests.fakes.platforms._standin import Call, Visible
from tests.fakes.platforms.telegram_standin import MAX_TEXT, TelegramStandin


@dataclass
class Inbound:
    chat_id: str
    message_id: str
    raw: Any


class TelegramDriver:
    name = "telegram"
    limit = MAX_TEXT
    user_id = "111"
    other_user_id = "222"
    group_chat_id = "-1001234567890"
    home_channel = "999"

    def __init__(self) -> None:
        self.standin = TelegramStandin()

    def start(self) -> None:
        self.standin.start()

    def stop(self) -> None:
        self.standin.stop()

    def gateway_config(self) -> Dict[str, Any]:
        return {"platforms": {"telegram": {"enabled": True, "extra": {
            "base_url": self.standin.api_base, "base_file_url": self.standin.file_base,
            "require_mention": True,
        }}}}

    def gateway_env(self) -> Dict[str, str]:
        return {"TELEGRAM_BOT_TOKEN": self.standin.token, "TELEGRAM_ALLOWED_USERS": self.user_id,
                "TELEGRAM_HOME_CHANNEL": self.home_channel, "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS": "1",
                # no text-batch debounce: one inbound is one turn, immediately
                "HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS": "0", "HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS": "0"}

    def connected(self) -> bool:
        return bool(self.standin.calls_of("getUpdates"))

    # inbound -----------------------------------------------------------------------------------
    def _wrap(self, update: Dict[str, Any]) -> Inbound:
        msg = update["message"]
        return Inbound(str(msg["chat"]["id"]), str(msg["message_id"]), update)

    def dm(self, text: str, user_id: Optional[str] = None) -> Inbound:
        return self._wrap(self.standin.dm(int(user_id or self.user_id), text))

    def group(self, text: str, *, mention: bool) -> Inbound:
        return self._wrap(self.standin.group(int(self.group_chat_id), int(self.user_id), text, mention=mention))

    def redeliver(self, inbound: Inbound) -> None:
        self.standin.redeliver(inbound.raw)

    def document(self, filename: str, data: bytes, mime: str, caption: str = "") -> Inbound:
        return self._wrap(self.standin.dm_document(int(self.user_id), filename, data, mime, caption))

    def buttons(self, chat_id: str) -> List[Dict[str, Any]]:
        return self.standin.buttons(chat_id)

    def click(self, chat_id: str, button: Dict[str, Any], user_id: Optional[str] = None) -> None:
        self.standin.callback(int(user_id or self.user_id), int(chat_id), int(button["message_id"]),
                              button["callback_data"])

    def callback_answers(self) -> List[Call]:
        return self.standin.calls_of("answerCallbackQuery")

    # ground truth ------------------------------------------------------------------------------
    def visible(self, chat_id: str) -> List[Visible]:
        return self.standin.visible(chat_id)

    def _ok(self, methods: tuple, chat_id: str) -> List[Call]:
        return [c for c in self.standin.calls_of(*methods)
                if not c.faulted and str(c.params.get("chat_id")) == str(chat_id)]

    def sends(self, chat_id: str) -> List[Call]:
        return self._ok(("sendMessage",), chat_id)

    def edits(self, chat_id: str) -> List[Call]:
        return self._ok(("editMessageText",), chat_id)

    def describe(self) -> str:
        return self.standin.describe()

    # faults ------------------------------------------------------------------------------------
    def fail_send(self, *, times: int = 1, match: Optional[Callable[[str], bool]] = None) -> None:
        pred = (lambda p: match(str(p.get("text", "")))) if match else None
        self.standin.fail("sendMessage", {"ok": False, "error_code": 400,
                                          "description": "Bad Request: chat not found"},
                          status=400, times=times, match=pred)

    def fail_edit(self, *, times: int = 1) -> None:
        self.standin.fail("editMessageText", {"ok": False, "error_code": 400,
                                              "description": "Bad Request: message can't be edited"},
                          status=400, times=times)
