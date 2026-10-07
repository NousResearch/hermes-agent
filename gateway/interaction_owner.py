"""Immutable ownership records for native messaging interaction prompts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import secrets
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class InteractionOwner:
    actor_id: str
    chat_id: str
    channel_id: str
    thread_id: str
    source_message_id: str
    profile: str
    generation: str
    prompt_message_id: str = ""

    @classmethod
    def capture(
        cls, chat_id: str, metadata: Optional[Mapping[str, Any]], *, generation: str = ""
    ) -> "InteractionOwner":
        meta = metadata or {}
        channel_id = str(meta.get("channel_id") or meta.get("parent_chat_id") or chat_id or "")
        return cls(
            actor_id=str(meta.get("user_id") or meta.get("recipient_user_id") or ""),
            chat_id=str(chat_id or ""),
            channel_id=channel_id,
            thread_id=str(meta.get("thread_id") or meta.get("thread_ts") or ""),
            source_message_id=str(meta.get("message_id") or ""),
            profile=str(meta.get("hermes_profile") or ""),
            generation=generation or secrets.token_hex(12),
        )

    def bind_prompt(self, message_id: Any) -> "InteractionOwner":
        """Return a new record; prompt identity can never mutate an existing generation."""
        return replace(self, prompt_message_id=str(message_id or ""))

    def accepts(
        self, *, actor_id: Any, chat_id: Any, channel_id: Any, thread_id: Any,
        prompt_message_id: Any, generation: str,
    ) -> bool:
        """Require an exact match for every available immutable prompt anchor.

        An anchor the prompt never captured (empty expected value, e.g. no thread, or a
        system prompt with no originating user) is not enforced: there is nothing to bind
        to, and the adapter's allowlist gate still runs afterwards. An anchor that WAS
        captured must match exactly; an empty or different actual value is rejected."""
        actual = {
            "actor_id": str(actor_id or ""),
            "chat_id": str(chat_id or ""),
            "channel_id": str(channel_id or ""),
            "thread_id": str(thread_id or ""),
            "prompt_message_id": str(prompt_message_id or ""),
            "generation": str(generation or ""),
        }
        expected = {
            "actor_id": self.actor_id,
            "chat_id": self.chat_id,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "prompt_message_id": self.prompt_message_id,
            "generation": self.generation,
        }
        return all(not value or actual[key] == value for key, value in expected.items())
