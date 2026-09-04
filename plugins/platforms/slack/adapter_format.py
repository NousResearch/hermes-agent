"""Slack format methods; SDK and mutable dependencies remain on the facade."""

import re
from typing import Any, Dict, List, Optional
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackFormatMixin:
    def _dm_top_level_threads_as_sessions(self) -> bool:
        """Each top-level DM reply thread is its own session (default True; set
        ``dm_top_level_threads_as_sessions: false`` for one session per DM channel)."""
        return self._extra_flag("dm_top_level_threads_as_sessions", default=True)

    def _cron_continuable_surface(self) -> str:
        """Continuable-cron surface: ``"thread"`` (default; seeded hidden thread) or
        ``"in_channel"`` (flat; shared session ``(slack, channel_id, None)``), from
        ``extra.cron_continuable_surface`` paired with ``reply_in_thread: false``. Unrecognised →
        ``"thread"`` (fail safe)."""
        raw = self.config.extra.get("cron_continuable_surface")
        return "in_channel" if str(raw).strip().lower() == "in_channel" else "thread"

    def _warn_if_inchannel_without_flat_reply(self, team_name: str) -> None:
        """Warn when ``in_channel`` is set without ``reply_in_thread: false``: both must hold for a
        flat cron seed to continue on a plain reply (same flat session). Warn only — the misconfig
        fails safe to a threaded continuation, never an orphaned session."""
        from . import adapter as _adapter

        try:
            if self._cron_continuable_surface() == "in_channel" and self.config.extra.get(
                "reply_in_thread", True):
                _adapter.logger.warning(
                    "[Slack] %s: cron_continuable_surface=in_channel is set WITHOUT "
                    "reply_in_thread=false. A continuable in-channel cron job will deliver flat, "
                    "but the bot will still reply to your continuation in a thread — so it falls "
                    "back to a threaded continuation (\u2248 default behaviour), not the flat "
                    "channel session you asked for. Set platforms.slack.extra.reply_in_thread: "
                    "false to pair them.", team_name)
        except Exception:
            pass

    def _slack_allow_bots(self) -> str:
        """Return normalized Slack bot-message policy."""
        from . import adapter as _adapter

        raw = self.config.extra.get("allow_bots", "") or _adapter.os.getenv("SLACK_ALLOW_BOTS", "none")
        value = str(raw).lower().strip()
        if value not in {"none", "mentions", "all"}:
            _adapter.logger.warning("[Slack] Unknown allow_bots=%r; treating as 'none'", raw)
            return "none"
        return value

    def _slack_api_human_users(self) -> frozenset:
        """User IDs whose Web-API posts count as human (``extra.api_human_users`` /
        ``SLACK_API_HUMAN_USERS``): ``xoxp-`` posts carry ``app_id`` and no ``client_msg_id`` so
        look like bots. Users only — an app-id allowlist would admit the app's own posts.

        A message posted with a *user* token (``xoxp-``) is authored by a real person, but Slack still
        stamps it with the posting ``app_id`` and it carries no ``client_msg_id`` — exactly the #35777
        app/bot signature in ``_event_declares_bot_sender``. Operators running their own front-end
        (dashboard, mobile shell) allowlist those *users* via ``platforms.slack.extra.api_human_users``
        (``SLACK_API_HUMAN_USERS`` fallback) instead of ``allow_bots: all``.
        """
        from . import adapter as _adapter

        cached = getattr(self, "_api_human_users_cache", None)
        if cached is None:
            raw = self.config.extra.get("api_human_users")
            if raw is None:
                raw = _adapter.os.getenv("SLACK_API_HUMAN_USERS", "")
            parts = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
            cached = self._api_human_users_cache = frozenset(
                str(p).strip() for p in parts if str(p).strip())
        return cached

    def _event_declares_bot_sender(self, event: dict) -> bool:
        """Return True when the Slack event itself identifies a bot sender."""
        if event.get("bot_id") or event.get("bot_profile") or event.get("subtype") == "bot_message":
            return True
        profile = event.get("user_profile")
        if isinstance(profile, dict) and bool(profile.get("is_bot")):
            return True
        # App-originated events may lack bot_id/subtype but carry app_id and no client_msg_id
        # (humans have one) → bot-authored unless the user is in _slack_api_human_users
        # (classic bot posts have no ``user`` so never match).
        # Real human-authored messages normally carry client_msg_id, so treat the combination as
        # app/bot-authored (#35777).
        if event.get("app_id") and not event.get("client_msg_id"):
            return event.get("user") not in self._slack_api_human_users()
        return False

    @staticmethod
    def _is_block_payload_rejection(error: BaseException) -> bool:
        """Errors recoverable by retrying without ``blocks`` (an enhancement over ``text``, so a
        rejected/oversized payload must not drop the whole response)."""
        recoverable_codes = {"invalid_blocks", "msg_too_long", "too_many_blocks"}
        response_get = getattr(getattr(error, "response", None), "get", None)
        if callable(response_get):
            try:
                if response_get("error") in recoverable_codes:
                    return True
            except Exception:
                pass
        return any(code in str(error) for code in recoverable_codes)

    def _extra_flag(self, key: str, default: bool = False) -> bool:
        """Boolean ``config.extra[key]`` (str forms accepted); ``default`` when unset."""
        raw = self.config.extra.get(key)
        return default if raw is None else str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _markdown_block_payload(self, content: str) -> Optional[list]:
        """Return a ``markdown`` block payload, or ``None`` when empty or over Slack's 12k cap."""
        ok = content and content.strip() and len(content) <= self._MARKDOWN_BLOCK_MAX
        return [{"type": "markdown", "text": content}] if ok else None

    def _feedback_block(self) -> Dict[str, Any]:
        """Return the Slack AI feedback-buttons block."""
        return {
            "type": "context_actions",
            "elements": [
                {
                    "type": "feedback_buttons",
                    "action_id": "hermes_feedback",
                    "positive_button": {
                        "text": {"type": "plain_text", "text": "Good Response"},
                        "accessibility_label": ("Submit positive feedback on this response"),
                        "value": "positive"},
                    "negative_button": {
                        "text": {"type": "plain_text", "text": "Bad Response"},
                        "accessibility_label": ("Submit negative feedback on this response"),
                        "value": "negative"}}]}

    def _append_feedback_block(self, blocks: Optional[list]) -> Optional[list]:
        """Append response feedback controls when enabled and block budget allows."""
        if blocks and self._extra_flag("feedback_buttons") and len(blocks) < 50:
            return [*blocks, self._feedback_block()]
        return blocks

    def _maybe_blocks(self, content: str) -> Optional[list]:
        """Block Kit for ``content``: ``markdown_blocks`` (native block, "platform AI" apps only,
        12k cap) over ``rich_blocks`` (local renderer). ``None`` when disabled or declined — a
        ``text`` fallback always accompanies blocks, so ``None`` is safe at any point.

        1. ``markdown_blocks`` — Slack's native ``markdown`` block renders the *raw* standard markdown
        (tables, headers, code fences with syntax highlighting) with Slack doing the translation (#8552). 2.
        """
        from . import adapter as _adapter

        if self._extra_flag("markdown_blocks"):
            md_blocks = self._markdown_block_payload(content)
            if md_blocks:
                return _adapter.sanitize_blocks(self._append_feedback_block(md_blocks))
        if not self._extra_flag("rich_blocks"):
            return None
        try:
            blocks = _adapter.render_blocks(content, mrkdwn_fn=self.format_message)
            return _adapter.sanitize_blocks(self._append_feedback_block(blocks))
        except Exception:  # pragma: no cover - renderer already guards itself
            _adapter.logger.debug("[Slack] block render failed; using plain text", exc_info=True)
            return None

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Slack mrkdwn.
        Tables are fenced first; code is protected from later passes; broadcast mentions are escaped
        before entity protection so output can't ping @channel."""
        from . import adapter as _adapter

        if not content:
            return content
        content = _adapter._wrap_markdown_tables(content)
        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash value behind a placeholder immune to later passes."""
            key = f"\x00SL{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        # <!everyone>/<!channel>/<!here> broadcast even from bots; escape the leading `<`.
        text = _adapter._SLACK_SPECIAL_MENTION_RE.sub(lambda m: m.group(0).replace("<", "&lt;", 1), content)

        def _protect_fence(m):
            # Slack renders the language tag literally, so drop it — only for a line-start
            # fence; a mid-line ``` is real content.
            block = m.group(0)
            if m.start() == 0 or m.string[m.start() - 1] == "\n":
                block = _adapter.re.sub(r"\A```[^\s`]+[ \t]*(\r?\n)", r"```\1", block)
            return _ph(block)

        def _convert_markdown_link(m):
            url = m.group(2).strip()
            if url.startswith("<") and url.endswith(">"):
                url = url[1:-1].strip()
            return _ph(f"<{url}|{m.group(1)}>")

        def _convert_header(m):
            inner = _adapter.re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1).strip())
            return _ph(f"*{inner}*")

        def _convert_bold(m):
            # Slack misses a closing * after a non-word char and silently truncates the
            # message; insert U+200B before it.
            inner = m.group(1)
            zw = "\u200b" if inner and not (inner[-1].isalnum() or inner[-1] == "_") else ""
            return _ph(f"*{inner}{zw}*")

        # Ordered passes: protect code/links/entities/quotes, escape, then convert emphasis.
        # Escaping unescapes first in ONE regex pass (sequential replaces would decode
        # "&amp;lt;" twice). ``None`` marks the escape step.
        passes = (
            (r"(```(?:[^\n]*\n)?[\s\S]*?```)", _protect_fence, 0),
            (r"(`[^`]+`)", lambda m: _ph(m.group(0)), 0),
            (r"(?<!!)\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)", _convert_markdown_link, 0),
            (r"(<(?:[@#!]|(?:https?|mailto|tel):)[^>\n]+>)", lambda m: _ph(m.group(1)), 0),
            (r"^(>+\s)", lambda m: _ph(m.group(0)), _adapter.re.MULTILINE),
            None,
            (r"^#{1,6}\s+(.+)$", _convert_header, _adapter.re.MULTILINE),
            (r"\*\*\*(.+?)\*\*\*", lambda m: _ph(f"*_{m.group(1)}_*"), 0),
            (r"\*\*(.+?)\*\*", _convert_bold, 0),
            # *text* → _text_ only when non-whitespace touches both delimiters ("a * b * c" stays).
            (r"(?<!\*)\*(\S(?:[^*\n]*?\S)?)\*(?!\*)", lambda m: _ph(f"_{m.group(1)}_"), 0),
            (r"~~(.+?)~~", lambda m: _ph(f"~{m.group(1)}~"), 0))
        for step in passes:
            if step is None:
                text = _adapter._SLACK_HTML_ENTITY_RE.sub(lambda m: _adapter._SLACK_HTML_ENTITIES[m.group(1)], text)
                text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            else:
                pattern, fn, flags = step
                text = _adapter.re.sub(pattern, fn, text, flags=flags)
        for key in reversed(placeholders):
            text = text.replace(key, placeholders[key])
        return text

    def _slack_require_mention(self) -> bool:
        """Whether channel messages need an @mention. Explicit-false parsing: unrecognised
        or empty values keep gating enabled (safe default True)."""
        from . import adapter as _adapter

        configured = self.config.extra.get("require_mention")
        if configured is None:
            configured = _adapter.os.getenv("SLACK_REQUIRE_MENTION", "true")
        if isinstance(configured, str):
            return configured.lower() not in {"false", "0", "no", "off"}
        return bool(configured)

    def _extra_or_env_flag(self, key: str, env_var: str, *, strip: bool = False) -> bool:
        """Opt-in boolean: ``config.extra[key]`` wins, else ``env_var`` (default false)."""
        from . import adapter as _adapter

        configured = self.config.extra.get(key)
        if configured is None:
            configured = _adapter.os.getenv(env_var, "false")
        if isinstance(configured, str):
            if strip:
                configured = configured.strip()
            return configured.lower() in {"true", "1", "yes", "on"}
        return bool(configured)

    def _slack_message_addressed_to_other_user(self, text: str, self_uids: set) -> bool:
        """True when the first token is a user mention (``<@U123>``/``<@U123|name>``)
        of someone other than the bot; ``<!here>``/``<#C…>`` address the room, not a person."""
        from . import adapter as _adapter

        match = text and _adapter.re.match(r"\s*<@([^>|\s]+)(?:\|[^>]*)?>", text)
        return bool(match) and match.group(1) not in self_uids

    def _slack_message_mentions_self(self, text: str, self_uids: set) -> bool:
        """True when ``text`` @-mentions this bot anywhere, in either ``<@U123>`` or
        ``<@U123|name>`` form (``is_mentioned`` only recognises the former)."""
        from . import adapter as _adapter

        return bool(text) and any(
            _adapter.re.search(rf"<@{_adapter.re.escape(uid)}(?:\|[^>]*)?>", text) for uid in self_uids)

    def _extra_or_env_channel_set(
        self, key: str, env_var: str, *, coerce_scalar: bool = False) -> set:
        """Channel-ID set from ``config.extra[key]`` (list or CSV) else ``env_var`` CSV.
        ``coerce_scalar`` accepts non-str scalars (a bare numeric YAML value loads as int)."""
        from . import adapter as _adapter

        raw = self.config.extra.get(key)
        if raw is None:
            raw = _adapter.os.getenv(env_var, "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if coerce_scalar:
            raw = str(raw).strip() if raw is not None else ""
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()

    def _slack_mention_patterns(self) -> List["re.Pattern"]:
        """Compile (cached) wake-word regexes from ``slack.mention_patterns`` (list/str) or
        ``SLACK_MENTION_PATTERNS`` (JSON list or newline/comma-separated)."""
        from . import adapter as _adapter

        cached = getattr(self, "_compiled_mention_patterns", None)
        if cached is not None:
            return cached
        patterns = self.config.extra.get("mention_patterns") if self.config.extra else None
        if patterns is None:
            raw = _adapter.os.getenv("SLACK_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    import json as _json
                    patterns = _json.loads(raw)
                except Exception:
                    patterns = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        if isinstance(patterns, str):
            patterns = [patterns]
        compiled: _adapter.List["re.Pattern"] = []
        if isinstance(patterns, list):
            for pat in patterns:
                if not isinstance(pat, str) or not pat.strip():
                    continue
                try:
                    compiled.append(_adapter.re.compile(pat, _adapter.re.IGNORECASE))
                except _adapter.re.error as exc:
                    _adapter.logger.warning("[Slack] Invalid mention pattern %r: %s", pat, exc)
        elif patterns is not None:
            _adapter.logger.warning(
                "[Slack] mention_patterns must be a list or string; got %s", type(patterns).__name__
            )
        if compiled:
            _adapter.logger.info("[Slack] Loaded %d mention pattern(s)", len(compiled))
        self._compiled_mention_patterns = compiled
        return compiled

    def _slack_message_matches_mention_patterns(self, text: str) -> bool:
        """Return True when ``text`` matches a configured wake-word pattern."""
        return bool(text) and any(p.search(text) for p in self._slack_mention_patterns())
