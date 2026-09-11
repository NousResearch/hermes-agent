"""Bounded transport receipts, never message content or user read receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeliveryReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    platform: Literal["telegram", "slack", "local"]
    destination: str = Field(
        min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f]+$"
    )
    message_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f]+$"
    )
    thread_id: str = Field(default="", max_length=128, pattern=r"^[^\x00-\x1f\x7f]*$")
    scope_id: str = Field(default="", max_length=128, pattern=r"^[^\x00-\x1f\x7f]*$")
    acknowledgement: Literal["provider_accepted", "transport_accepted"]

    @model_validator(mode="after")
    def acknowledgement_matches_surface(self):
        expected = (
            "transport_accepted" if self.platform == "local" else "provider_accepted"
        )
        if self.acknowledgement != expected:
            raise ValueError("delivery acknowledgement does not match its surface")
        return self


def telegram_receipt(response, *, chat_id: str, thread_id: str = "") -> DeliveryReceipt:
    if not isinstance(response, Mapping):
        response = (
            response.to_dict() if callable(getattr(response, "to_dict", None)) else None
        )
    if isinstance(response, Mapping) and isinstance(response.get("result"), Mapping):
        if response.get("ok") is not True:
            raise ValueError("Telegram did not acknowledge delivery")
        response = response["result"]
    if not isinstance(response, Mapping):
        raise ValueError("Telegram delivery response is missing")
    message_id = response.get("message_id")
    chat = response.get("chat")
    if type(message_id) is not int or message_id <= 0 or not isinstance(chat, Mapping):
        raise ValueError("Telegram delivery identity is missing")
    if str(chat.get("id")) != str(chat_id):
        raise ValueError("Telegram delivery destination changed")
    actual_thread = str(response.get("message_thread_id") or "")
    if actual_thread != thread_id:
        raise ValueError("Telegram delivery thread changed")
    return DeliveryReceipt(
        platform="telegram",
        destination=str(chat_id),
        message_id=str(message_id),
        thread_id=thread_id,
        acknowledgement="provider_accepted",
    )


def slack_receipt(
    response, *, channel_id: str, thread_id: str = "", scope_id: str = ""
) -> DeliveryReceipt:
    data = (
        response if isinstance(response, Mapping) else getattr(response, "data", None)
    )
    if not isinstance(data, Mapping) or data.get("ok") is not True:
        raise ValueError("Slack did not acknowledge delivery")
    if (
        data.get("channel") != channel_id
        or not isinstance(data.get("ts"), str)
        or not data["ts"]
    ):
        raise ValueError("Slack delivery identity is missing or changed")
    message = data.get("message")
    if (
        isinstance(message, Mapping)
        and message.get("thread_ts", thread_id) != thread_id
    ):
        raise ValueError("Slack delivery thread changed")
    return DeliveryReceipt(
        platform="slack",
        destination=channel_id,
        message_id=data["ts"],
        thread_id=thread_id,
        scope_id=scope_id,
        acknowledgement="provider_accepted",
    )
