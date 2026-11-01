"""Owning-profile policy for Discord attachment size and text projection."""

import logging

logger = logging.getLogger(__name__)


def seed_attachment_config(config, extra):
    for key in ('inline_text_attachments', 'max_attachment_bytes'):
        if key in config:
            extra[key] = config[key]


class DiscordAttachmentPolicyMixin:
    def _discord_max_attachment_bytes(self) -> int:
        """Per-attachment byte cap; 0 = unlimited (whole attachment is held in memory). Default 32 MiB."""
        from plugins.platforms.discord.adapter import _extra_or_secret

        configured = _extra_or_secret(self.config.extra, "max_attachment_bytes", "DISCORD_MAX_ATTACHMENT_BYTES", None)
        if configured is None or configured == "":
            return 32 * 1024 * 1024
        try:
            value = int(configured)
        except (TypeError, ValueError):
            logger.warning(
                "[Discord] Invalid max_attachment_bytes value %r, falling back to 32 MiB",
                configured,
            )
            return 32 * 1024 * 1024
        return max(0, value)

