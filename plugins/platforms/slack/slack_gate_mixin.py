"""Slack gate & mention/admission policy mixin.

Extracted from plugins/platforms/slack/adapter.py -- R5 slice C1 (gate mixin)
of the god-file kill campaign (epic #78647, target #78638).

Provenance:
- Source window: adapter.py lines 8233-8464 (12 members, contiguous); the
  ``# ── Channel mention gating ──`` banner at 8231-8232 rides along for
  provenance (consensus D1: cosmetic, may move with the block).
- Pin sha: 1be70d63548845eb8918c08ed698cda0674cf9a7 (consensus-verified;
  adapter.py blob at origin/main 392e3a8c53 is byte-identical to pin).
- Move: byte-verbatim. The moved members are pure stdlib config reads
  (module-global intersection: os, re, logger, typing.List) with zero
  adapter-module helper references, so no lazy imports were needed.
- Class line after slice:
    class SlackAdapter(SlackGateMixin, BasePlatformAdapter):
- No module-level import of adapter (circular-import guard). The logger keeps
  the qualified adapter name so log records retain their original identity.
"""

import logging
import os
import re
from typing import List

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackGateMixin:
    """Gate & mention policy for the Slack adapter.

    MRO contract: this mixin must precede ``BasePlatformAdapter`` in
    ``SlackAdapter``'s bases so its ``_slack_*`` methods resolve through the
    adapter without shadowing anything on the base (mixin-first, mirroring
    ``DiscordGateMixin``).
    """

    # ── Channel mention gating ─────────────────────────────────────────────

    def _slack_require_mention(self) -> bool:
        """Return whether channel messages require an explicit bot mention.

        Uses explicit-false parsing (like Discord/Matrix) rather than
        truthy parsing, since the safe default is True (gating on).
        Unrecognised or empty values keep gating enabled.
        """
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off"}
            return bool(configured)
        return os.getenv("SLACK_REQUIRE_MENTION", "true").lower() not in {
            "false",
            "0",
            "no",
            "off",
        }

    def _slack_strict_mention(self) -> bool:
        """When true, channel threads require an explicit @-mention on every
        message. Disables all auto-triggers (mentioned-thread memory,
        bot-message follow-up, session-presence). Defaults to False.
        """
        configured = self.config.extra.get("strict_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("SLACK_STRICT_MENTION", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _slack_ignore_other_user_mentions(self) -> bool:
        """When true, ignore channel/thread messages addressed to another user.

        A message whose first token @-mentions someone other than this bot is
        treated as directed at that person; the bot stays silent unless it is
        also mentioned. Defaults to False (opt-in) so existing behaviour is
        unchanged until enabled. Mirrors Discord's ``ignore_other_user_mentions``
        (PR #33501), adapted to Slack's thread model: the trigger is a *leading*
        mention ("addressed to"), so a message that merely references another
        user mid-sentence still reaches the bot.
        """
        configured = self.config.extra.get("ignore_other_user_mentions")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("SLACK_IGNORE_OTHER_USER_MENTIONS", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _slack_thread_require_mention(self) -> bool:
        """When true, Slack thread replies require an explicit @-mention.

        This is narrower than ``strict_mention``: top-level channel messages can
        still be processed without a mention when ``require_mention`` is false
        or the channel is listed in ``free_response_channels``. Thread replies
        remain gated to prevent a bot from joining every follow-up in busy
        support threads.
        """
        configured = self.config.extra.get("thread_require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("SLACK_THREAD_REQUIRE_MENTION", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _slack_message_addressed_to_other_user(self, text: str, self_uids: set) -> bool:
        """Return True when ``text`` opens by @-mentioning a non-bot user.

        Slack renders a user mention as ``<@U123>`` (or ``<@U123|name>``). A
        message whose first token is such a mention is addressed to that user.
        Returns False when the leading mention is the bot itself (``self_uids``),
        when there is no leading user mention, or for channel/broadcast tokens
        (``<!here>``, ``<#C…>``) which address the room rather than a person.
        """
        if not text:
            return False
        match = re.match(r"\s*<@([^>|\s]+)(?:\|[^>]*)?>", text)
        if not match:
            return False
        return match.group(1) not in self_uids

    def _slack_message_mentions_self(self, text: str, self_uids: set) -> bool:
        """Return True when ``text`` @-mentions this bot anywhere in the message.

        Matches both mention markups — ``<@U123>`` and the pipe form
        ``<@U123|name>`` — so the ignore_other_user_mentions gate treats a
        pipe-form bot mention as "also mentioned" even though the exact-markup
        ``is_mentioned`` check only recognises ``<@U123>``.
        """
        if not text:
            return False
        return any(
            re.search(rf"<@{re.escape(uid)}(?:\|[^>]*)?>", text)
            for uid in self_uids
        )

    def _slack_free_response_channels(self) -> set:
        """Return channel IDs where no @mention is required."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = os.getenv("SLACK_FREE_RESPONSE_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        # Coerce non-list scalars (str/int/float) to str before splitting.
        # A bare numeric YAML value (`free_response_channels: 1234567890`) is
        # loaded as int and was previously falling through the isinstance(str)
        # branch to return an empty set.  str() here accepts whatever scalar
        # the YAML loader hands us without changing existing string/CSV
        # semantics.
        s = str(raw).strip() if raw is not None else ""
        if s:
            return {part.strip() for part in s.split(",") if part.strip()}
        return set()

    def _slack_disable_dms(self) -> bool:
        """Return whether incoming Slack DMs should be ignored.

        Supports both profile config (``slack.disable_dms`` bridged into
        ``PlatformConfig.extra``) and the environment override
        ``SLACK_DISABLE_DMS``. Defaults to False for backward compatibility.
        """
        raw = self.config.extra.get("disable_dms")
        if raw is None:
            raw = os.getenv("SLACK_DISABLE_DMS", "false")
        if isinstance(raw, str):
            return raw.strip().lower() in {"true", "1", "yes", "on"}
        return bool(raw)

    def _slack_allowed_channels(self) -> set:
        """Return the whitelist of channel IDs the bot will respond in.

        When non-empty, messages from channels NOT in this set are silently
        ignored — even if the bot is @mentioned.  DMs are controlled separately
        by ``_slack_disable_dms()``. Empty set means no channel restriction
        (fully backward compatible).
        """
        raw = self.config.extra.get("allowed_channels")
        if raw is None:
            raw = os.getenv("SLACK_ALLOWED_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()

    def _slack_require_mention_channels(self) -> set:
        """Return channel IDs where a bot @mention is ALWAYS required.

        Per-channel override in the opposite direction of
        ``free_response_channels``: even when ``require_mention`` is disabled
        globally (or the channel would otherwise be free-response), messages
        in these channels only reach the bot via an explicit mention or one
        of the wake checks in :meth:`_should_wake_on_unmentioned_message`.
        Empty set means no per-channel force-mention override (#13855).
        """
        raw = self.config.extra.get("require_mention_channels")
        if raw is None:
            raw = os.getenv("SLACK_REQUIRE_MENTION_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()

    def _slack_mention_patterns(self) -> List["re.Pattern"]:
        """Compile optional regex wake-word patterns for channel triggers.

        Parity with the other adapters (Telegram, DingTalk, Mattermost,
        WhatsApp, BlueBubbles, Photon): when ``require_mention`` is on, a
        channel message matching one of these patterns triggers the bot even
        without a literal ``<@BOTUID>`` mention. Reads ``slack.mention_patterns``
        (a list or single string) or ``SLACK_MENTION_PATTERNS`` (a JSON list, or
        newline/comma-separated values). Compiled patterns are cached on the
        instance. Previously this documented field was silently dropped.
        """
        cached = getattr(self, "_compiled_mention_patterns", None)
        if cached is not None:
            return cached

        patterns = self.config.extra.get("mention_patterns") if self.config.extra else None
        if patterns is None:
            raw = os.getenv("SLACK_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    import json as _json
                    patterns = _json.loads(raw)
                except Exception:
                    patterns = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]

        if isinstance(patterns, str):
            patterns = [patterns]

        compiled: List["re.Pattern"] = []
        if isinstance(patterns, list):
            for pat in patterns:
                if not isinstance(pat, str) or not pat.strip():
                    continue
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error as exc:
                    logger.warning("[Slack] Invalid mention pattern %r: %s", pat, exc)
        elif patterns is not None:
            logger.warning(
                "[Slack] mention_patterns must be a list or string; got %s",
                type(patterns).__name__,
            )

        if compiled:
            logger.info("[Slack] Loaded %d mention pattern(s)", len(compiled))
        self._compiled_mention_patterns = compiled
        return compiled

    def _slack_message_matches_mention_patterns(self, text: str) -> bool:
        """Return True when ``text`` matches a configured wake-word pattern."""
        if not text:
            return False
        return any(pattern.search(text) for pattern in self._slack_mention_patterns())
