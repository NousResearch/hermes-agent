"""Opt-in owner mentions on Discord blocking prompts (exec approvals, slash confirmations, clarify, update)."""
from __future__ import annotations

import re
from typing import Any, Optional


class DiscordPromptMentionsMixin:
    MAX_MESSAGE_LENGTH: int
    _allowed_user_ids: set
    _extra_or_env_flag: Any

    def _approval_mention_content(self) -> Optional[str]:
        """Owner mentions for blocking prompts, gated on ``discord.approval_mentions``
        (``DISCORD_APPROVAL_MENTIONS``) through the profile-scoped flag reader. Only numeric
        allowlist entries; default off."""
        if not self._extra_or_env_flag("approval_mentions", "DISCORD_APPROVAL_MENTIONS", "false", truthy=True):
            return None
        user_ids = sorted(uid for uid in self._allowed_user_ids if str(uid).isdigit())
        if not user_ids:
            return None
        # Keep enough room for the actual blocking prompt. Large enterprise
        # allowlists can otherwise make the mention line exceed Discord's
        # 2,000-character message limit before the prompt body is added.
        mention_budget = self.MAX_MESSAGE_LENGTH // 4
        mentions: list[str] = []
        used = 0
        for uid in user_ids:
            token = f"<@{uid}>"
            added = len(token) + (1 if mentions else 0)
            if used + added > mention_budget:
                break
            mentions.append(token)
            used += added
        return " ".join(mentions) or None

    def _mention_prompt_header(self, header: str) -> tuple[str, Optional[str]]:
        """``(header, mentions)`` with the owner ping line in front of *header* when opted in."""
        mention_content = self._approval_mention_content()
        return (f"{mention_content}\n{header}" if mention_content else header), mention_content

    @staticmethod
    def _interactive_prompt_send_kwargs(
        *, content: str, embed: Any, view: Any = None,
        mention_content: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build safe Discord kwargs for a blocking interactive prompt: only the owners named in the
        ping line may be notified, never a mention echoed in the prompt body."""
        from plugins.platforms.discord.adapter import discord

        send_kwargs: dict[str, Any] = {"content": content, "embed": embed}
        if view is not None:
            send_kwargs["view"] = view
        if mention_content:
            allowed_mentions_cls = getattr(discord, "AllowedMentions", None)
            object_cls = getattr(discord, "Object", None)
            if allowed_mentions_cls is not None and object_cls is not None:
                owner_ids = [int(uid) for uid in re.findall(r"<@(\d+)>", mention_content)]
                send_kwargs["allowed_mentions"] = allowed_mentions_cls(
                    users=[object_cls(id=uid) for uid in owner_ids],
                    roles=False,
                    everyone=False,
                    replied_user=False,
                )
        return send_kwargs
