"""
Transport-agnostic WhatsApp behavior shared by the Baileys bridge adapter
and the official WhatsApp Cloud API adapter.

The mixin provides:
- Allow-list / DM / group gating
- Mention detection (explicit @-mentions + configurable regex patterns)
- Quoted-reply-to-bot detection
- Broadcast / Channel / Newsletter filtering
- WhatsApp-flavored markdown conversion
- Outgoing chunk length budgeting

It is the *behavior layer*. Transport-specific concerns (subprocess management,
HTTP webhooks, Graph API calls, media upload protocols) live in each adapter.

Mixin contract — the adapter must set these on ``self`` before any of the
mixin's methods are called (typically in ``__init__``):

    self.config        # gateway.config.PlatformConfig
    self.name          # str — adapter name (used in log lines)
    self._dm_policy             # str: "open" | "allowlist" | "disabled"
    self._allow_from            # set[str]
    self._group_policy          # str: "open" | "allowlist" | "disabled"
    self._group_allow_from      # set[str]
    self._mention_patterns      # list[re.Pattern]
    self._reply_prefix          # Optional[str]

Class attributes ``MAX_MESSAGE_LENGTH`` and ``DEFAULT_REPLY_PREFIX`` are
defined on the mixin and may be overridden per-adapter if needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# Files that make up the Baileys bridge source (everything except the installed
# ``node_modules`` and the install stamp). Used to mirror the bridge into a
# writable location when the install tree is read-only.
_BRIDGE_SOURCE_GLOBS = ("*.js", "*.mjs", "*.json")


def bridge_source_dir() -> Path:
    """Directory where the Baileys WhatsApp bridge source ships in the install."""
    return Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"


def _dir_is_writable(path: Path) -> bool:
    """True when ``npm install`` could create ``node_modules`` inside ``path``.

    Checks ``path`` itself when it exists, otherwise the nearest existing parent
    (npm would create the leaf dir). Used to detect read-only install trees such
    as ``/opt/hermes`` in container images.
    """
    try:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return os.access(probe, os.W_OK)
    except OSError:
        return False


def _sync_bridge_source(source_dir: Path, target_dir: Path) -> None:
    """Mirror bridge source files into ``target_dir`` when missing or changed.

    Leaves an existing ``node_modules`` in ``target_dir`` untouched; only source
    files are refreshed so an updated ``package.json`` (e.g. after ``hermes
    update``) propagates and the adapter's dependency-refresh stamp reinstalls.
    """
    import shutil

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create WhatsApp bridge dir %s: %s", target_dir, exc)
        return

    for pattern in _BRIDGE_SOURCE_GLOBS:
        for src in source_dir.glob(pattern):
            dst = target_dir / src.name
            try:
                if not dst.exists() or src.read_bytes() != dst.read_bytes():
                    shutil.copy2(src, dst)
            except OSError as exc:
                logger.warning("Could not copy bridge file %s -> %s: %s", src, dst, exc)
                continue


def resolve_whatsapp_bridge_dir() -> Path:
    """Return a writable WhatsApp bridge directory, mirroring source if needed.

    The bridge source ships under ``<install>/scripts/whatsapp-bridge`` and npm
    must create ``node_modules`` inside the bridge dir. In container images the
    install tree (e.g. ``/opt/hermes``) is read-only for the runtime user, so
    ``npm install`` fails with ``EACCES`` (issue #49561). When the install dir
    is not writable, mirror the bridge source into a writable ``HERMES_HOME``
    location and use that instead. Writable installs are unaffected.
    """
    from hermes_constants import get_hermes_home

    source_dir = bridge_source_dir()
    if _dir_is_writable(source_dir):
        return source_dir

    target_dir = get_hermes_home() / "scripts" / "whatsapp-bridge"
    _sync_bridge_source(source_dir, target_dir)
    return target_dir


class WhatsAppBehaviorMixin:
    """Shared behavior for all WhatsApp adapters (Baileys + Cloud API).

    See module docstring for the attribute contract the host adapter must
    satisfy. This mixin owns no state of its own — every value it touches
    is either a class attribute or set by the adapter's ``__init__``.
    """

    # WhatsApp message limits — practical UX limit, not protocol max.
    # WhatsApp allows ~65K but long messages are unreadable on mobile.
    MAX_MESSAGE_LENGTH: int = 4096
    supports_code_blocks = True  # WhatsApp renders fenced code blocks (monospace)

    DEFAULT_REPLY_PREFIX: str = "⚕ *Hermes Agent*\n────────────\n"

    @property
    def enforces_own_access_policy(self) -> bool:
        """WhatsApp gates DM/group access at intake via dm_policy/group_policy."""
        return True

    # ------------------------------------------------------------------ config
    def _effective_reply_prefix(self) -> str:
        """Return the prefix to add to outgoing replies in self-chat mode.

        Subclasses that don't have a self-chat concept (the Cloud API
        adapter) can override this to always return ``""`` or apply a
        different policy.
        """
        whatsapp_mode = os.getenv("WHATSAPP_MODE", "self-chat")
        if whatsapp_mode != "self-chat":
            return ""
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        env_prefix = os.getenv("WHATSAPP_REPLY_PREFIX")
        if env_prefix is not None:
            return env_prefix.replace("\\n", "\n")
        return self.DEFAULT_REPLY_PREFIX

    def _outgoing_chunk_limit(self) -> int:
        """Reserve room for the reply prefix so the final message fits."""
        prefix_len = len(self._effective_reply_prefix())
        # Keep enough space for truncate_message's pagination indicator and
        # code-fence repair even if a user configures a very long prefix.
        return max(1024, self.MAX_MESSAGE_LENGTH - prefix_len)

    def _whatsapp_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("WHATSAPP_REQUIRE_MENTION", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _whatsapp_free_response_chats(self) -> set[str]:
        raw = self.config.extra.get("free_response_chats")
        if raw is None:
            raw = os.getenv("WHATSAPP_FREE_RESPONSE_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    @staticmethod
    def _coerce_allow_list(raw) -> set[str]:
        """Parse allow_from / group_allow_from from config or env var."""
        if raw is None:
            return set()
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    # ------------------------------------------------------------------ JID helpers
    @staticmethod
    def _normalize_whatsapp_id(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = str(value).strip()
        if ":" in normalized and "@" in normalized:
            normalized = normalized.replace(":", "@", 1)
        return normalized

    @staticmethod
    def _is_broadcast_chat(chat_id: str) -> bool:
        """True for WhatsApp pseudo-chats that aren't real conversations.

        Covers Status updates (Stories) and Channel/Newsletter broadcasts.
        These show up as inbound messages on Baileys but the agent should
        never reply — answering a Story update spams the contact's status
        feed, and Channel posts aren't addressable in the first place.
        """
        if not chat_id:
            return False
        cid = chat_id.strip().lower()
        if cid == "status@broadcast":
            return True
        # @broadcast suffix covers status@broadcast plus any future
        # broadcast-list variants. @newsletter is the Channel JID suffix.
        if cid.endswith("@broadcast") or cid.endswith("@newsletter"):
            return True
        return False

    # ------------------------------------------------------------------ gating
    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Check whether a DM from the given sender should be processed."""
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return sender_id in self._allow_from
        # "open" — all DMs allowed
        return True

    def _is_group_allowed(self, chat_id: str) -> bool:
        """Check whether a group chat should be processed."""
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return chat_id in self._group_allow_from
        # "open" — all groups allowed
        return True

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("WHATSAPP_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    patterns = [
                        part.strip() for part in raw.splitlines() if part.strip()
                    ]
                    if not patterns:
                        patterns = [
                            part.strip() for part in raw.split(",") if part.strip()
                        ]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning(
                "[%s] whatsapp mention_patterns must be a list or string; got %s",
                self.name,
                type(patterns).__name__,
            )
            return []

        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning(
                    "[%s] Invalid WhatsApp mention pattern %r: %s",
                    self.name,
                    pattern,
                    exc,
                )
        if compiled:
            logger.info(
                "[%s] Loaded %d WhatsApp mention pattern(s)", self.name, len(compiled)
            )
        return compiled

    def _bot_ids_from_message(self, data: Dict[str, Any]) -> set[str]:
        bot_ids = set()
        for candidate in data.get("botIds") or []:
            normalized = self._normalize_whatsapp_id(candidate)
            if normalized:
                bot_ids.add(normalized)
        return bot_ids

    def _message_is_reply_to_bot(self, data: Dict[str, Any]) -> bool:
        quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
        if not quoted_participant:
            return False
        return quoted_participant in self._bot_ids_from_message(data)

    def _message_mentions_bot(self, data: Dict[str, Any]) -> bool:
        bot_ids = self._bot_ids_from_message(data)
        if not bot_ids:
            return False
        mentioned_ids = {
            nid
            for candidate in (data.get("mentionedIds") or [])
            if (nid := self._normalize_whatsapp_id(candidate))
        }
        if mentioned_ids & bot_ids:
            return True

        body = str(data.get("body") or "")
        lower_body = body.lower()
        for bot_id in bot_ids:
            bare_id = bot_id.split("@", 1)[0].lower()
            if bare_id and (f"@{bare_id}" in lower_body or bare_id in lower_body):
                return True
        return False

    def _message_matches_mention_patterns(self, data: Dict[str, Any]) -> bool:
        if not self._mention_patterns:
            return False
        body = str(data.get("body") or "")
        return any(pattern.search(body) for pattern in self._mention_patterns)

    def _clean_bot_mention_text(self, text: str, data: Dict[str, Any]) -> str:
        if not text:
            return text
        bot_ids = self._bot_ids_from_message(data)
        cleaned = text
        for bot_id in bot_ids:
            bare_id = bot_id.split("@", 1)[0]
            if bare_id:
                cleaned = re.sub(
                    rf"@{re.escape(bare_id)}\b[,:\-]*\s*", "", cleaned
                )
        return cleaned.strip() or text

    def _should_process_message(self, data: Dict[str, Any]) -> bool:
        chat_id_raw = str(data.get("chatId") or "")
        # WhatsApp uses pseudo-chats for Status updates (Stories) and
        # Channel/Newsletter broadcasts. These are not real conversations
        # and the agent should never reply to them — even in self-chat mode
        # where the bridge may surface them as "fromMe" events.
        if self._is_broadcast_chat(chat_id_raw):
            return False
        is_group = data.get("isGroup", False)
        if is_group:
            chat_id = chat_id_raw
            if not self._is_group_allowed(chat_id):
                return False
        else:
            sender_id = str(data.get("senderId") or data.get("from") or "")
            if not self._is_dm_allowed(sender_id):
                return False
            # DMs that pass the policy gate are always processed
            return True
        # Group messages: check mention / free-response settings
        chat_id = str(data.get("chatId") or "")
        if chat_id in self._whatsapp_free_response_chats():
            return True
        if not self._whatsapp_require_mention():
            return True
        body = str(data.get("body") or "").strip()
        if body.startswith("/"):
            return True
        if self._message_is_reply_to_bot(data):
            return True
        if self._message_mentions_bot(data):
            return True
        return self._message_matches_mention_patterns(data)

    # ------------------------------------------------------------------ formatting
    def format_message(self, content: str) -> str:
        """Convert standard markdown to WhatsApp-compatible formatting.

        WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```code```,
        and monospaced `inline`. Standard markdown uses different syntax
        for bold/italic/strikethrough, so we convert here.

        Code blocks (``` fenced) and inline code (`) are protected from
        conversion via placeholder substitution.
        """
        if not content:
            return content

        # --- 1. Protect fenced code blocks from formatting changes ---
        _FENCE_PH = "\x00FENCE"
        fences: list[str] = []

        def _save_fence(m: re.Match) -> str:
            fences.append(m.group(0))
            return f"{_FENCE_PH}{len(fences) - 1}\x00"

        result = re.sub(r"```[\s\S]*?```", _save_fence, content)

        # --- 2. Protect inline code ---
        _CODE_PH = "\x00CODE"
        codes: list[str] = []

        def _save_code(m: re.Match) -> str:
            codes.append(m.group(0))
            return f"{_CODE_PH}{len(codes) - 1}\x00"

        result = re.sub(r"`[^`\n]+`", _save_code, result)

        # --- 3. Convert markdown formatting to WhatsApp syntax ---
        # Bold: **text** or __text__ → *text*
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
        result = re.sub(r"__(.+?)__", r"*\1*", result)
        # Strikethrough: ~~text~~ → ~text~
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)
        # Italic: *text* is already WhatsApp italic — leave as-is
        # _text_ is already WhatsApp italic — leave as-is

        # --- 4. Convert markdown headers to bold text ---
        # # Header → *Header*. Strip any *...* wrapping already produced
        # by step 3 (e.g. "# **Title**" → "*Title*", not "**Title**",
        # which WhatsApp renders with literal asterisks).
        def _header_to_bold(m: re.Match) -> str:
            inner = m.group(1).strip()
            while len(inner) > 1 and inner.startswith("*") and inner.endswith("*"):
                inner = inner[1:-1].strip()
            return f"*{inner}*"

        result = re.sub(
            r"^#{1,6}\s+(.+)$", _header_to_bold, result, flags=re.MULTILINE
        )

        # --- 5. Convert markdown links: [text](url) → text (url) ---
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)

        # --- 6. Restore protected sections ---
        for i, fence in enumerate(fences):
            result = result.replace(f"{_FENCE_PH}{i}\x00", fence)
        for i, code in enumerate(codes):
            result = result.replace(f"{_CODE_PH}{i}\x00", code)

        return result
