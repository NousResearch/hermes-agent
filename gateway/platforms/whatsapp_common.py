"""
Transport-agnostic WhatsApp behavior shared by the Baileys bridge adapter and the
Cloud API adapter: allow-list / DM / group gating, mention detection, quoted-reply-
to-bot detection, broadcast filtering, WhatsApp markdown conversion, chunk budgeting.

Mixin contract — the host adapter sets these on ``self`` before calling any mixin
method: ``config`` (PlatformConfig), ``name``, ``_dm_policy`` / ``_group_policy``
("open" | "allowlist" | "disabled"), ``_allow_from`` / ``_group_allow_from`` (set[str]),
``_mention_patterns`` (list[re.Pattern]), ``_reply_prefix`` (Optional[str]).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from gateway.platforms._shared import get_scoped_secret as _get_wsecret


logger = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes", "on"}
_OPTIN_TRUTHY = {"true", "1", "yes"}


def _stash(pattern: str, text: str, tag: str) -> tuple[str, list[str]]:
    """Replace every ``pattern`` match with a ``\\x00<tag><n>\\x00`` placeholder."""
    saved: list[str] = []

    def keep(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"\x00{tag}{len(saved) - 1}\x00"

    return re.sub(pattern, keep, text), saved


def _header_to_bold(m: re.Match) -> str:
    """``# Header`` → ``*Header*``, stripping already-bolded ``*...*`` so ``# **Title**``
    doesn't render with literal asterisks."""
    inner = m.group(1).strip()
    while len(inner) > 1 and inner.startswith("*") and inner.endswith("*"):
        inner = inner[1:-1].strip()
    return f"*{inner}*"


class WhatsAppBehaviorMixin:
    """Shared behavior for all WhatsApp adapters (Baileys + Cloud API); owns no state
    of its own — see the module docstring for the host adapter's attribute contract."""

    # Practical UX limit, not the ~65K protocol max (long messages are unreadable on mobile).
    MAX_MESSAGE_LENGTH: int = 4096
    supports_code_blocks = True  # WhatsApp renders fenced code blocks (monospace)

    DEFAULT_REPLY_PREFIX: str = "⚕ *Hermes Agent*\n────────────\n"

    _OUTBOUND_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u2060\u2063\ufeff]")
    _OUTBOUND_ODD_SPACE_RE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]")

    @classmethod
    def _sanitize_outbound_text(cls, content: str) -> str:
        """Strip zero-width format chars (WORD JOINER etc.) and normalize odd unicode
        spaces — WhatsApp renders them as mojibake prefixes. Emoji joiners are kept."""
        if not content:
            return content
        return cls._OUTBOUND_ODD_SPACE_RE.sub(" ", cls._OUTBOUND_INVISIBLE_CHARS_RE.sub("", content))

    @property
    def enforces_own_access_policy(self) -> bool:
        """WhatsApp gates DM/group access at intake via dm_policy/group_policy."""
        return True

    def _effective_reply_prefix(self) -> str:
        """Prefix for outgoing replies in self-chat mode (Cloud API overrides to ``""``)."""
        if (_get_wsecret("WHATSAPP_MODE", default="self-chat") or "self-chat") != "self-chat":
            return ""
        if self._reply_prefix is not None:
            return self._reply_prefix.replace("\\n", "\n")
        env_prefix = _get_wsecret("WHATSAPP_REPLY_PREFIX")
        if env_prefix is not None:
            return env_prefix.replace("\\n", "\n")
        return self.DEFAULT_REPLY_PREFIX

    def _outgoing_chunk_limit(self) -> int:
        """Reserve room for the reply prefix; floor keeps space for pagination/fence repair."""
        return max(1024, self.MAX_MESSAGE_LENGTH - len(self._effective_reply_prefix()))

    def _whatsapp_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is None:
            configured = _get_wsecret("WHATSAPP_REQUIRE_MENTION", default="false") or "false"
        if isinstance(configured, str):
            return configured.lower() in _TRUTHY
        return bool(configured)

    def _whatsapp_free_response_chats(self) -> set[str]:
        raw = self.config.extra.get("free_response_chats")
        if raw is None:
            raw = _get_wsecret("WHATSAPP_FREE_RESPONSE_CHATS", default="") or ""
        return self._coerce_allow_list(raw)

    @staticmethod
    def _coerce_allow_list(raw) -> set[str]:
        """Parse allow_from / group_allow_from from config (list) or env var (CSV)."""
        if raw is None:
            return set()
        parts = raw if isinstance(raw, list) else str(raw).split(",")
        return {str(part).strip() for part in parts if str(part).strip()}

    def _select_dm_allowlist(self, extra: Dict[str, Any], env_keys, read_env) -> Any:
        """Pick the raw DM allowlist by key *presence*: ``allow_from``/``allowFrom`` in config (an
        explicit empty list stays authoritative), then the first truthy env carrier. Records the
        winning source in ``_dm_allowlist_source`` so live DM checks keep the same precedence."""
        for key in ("allow_from", "allowFrom"):
            if key in extra:
                self._dm_allowlist_source = "config"
                return extra.get(key)
        for env in env_keys:
            if read_env(env):
                self._dm_allowlist_source = env
                return read_env(env)
        self._dm_allowlist_source = None
        return None

    def _live_dm_allow_from(self) -> set[str]:
        """Allowlist currently enforced for DM intake / strict DM auth. Env-seeded adapters re-read
        the same key so pairing approve/revoke takes effect without restart; a removed key (sole-entry
        revoke) means empty, not the construction snapshot. Config-seeded adapters keep the in-memory
        set (pairing revoke purges it in place) — a stale env value must not broaden access."""
        source = getattr(self, "_dm_allowlist_source", None)
        if isinstance(source, str) and source != "config":
            return self._coerce_allow_list(os.environ[source]) if source in os.environ else set()
        return set(self._allow_from or ())

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
        """Status updates (Stories) and Channel/Newsletter broadcasts — never reply
        (answering a Story spams the status feed; Channel posts aren't addressable)."""
        cid = (chat_id or "").strip().lower()
        return cid == "status@broadcast" or cid.endswith(("@broadcast", "@newsletter"))

    # ------------------------------------------------------------------ gating
    def _open_dm_opted_in(self) -> bool:
        if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in _OPTIN_TRUTHY:
            return True
        return (_get_wsecret("WHATSAPP_ALLOW_ALL_USERS", default="") or "").lower() in _OPTIN_TRUTHY

    @staticmethod
    def _matches_whatsapp_allowlist(candidate: str, allow_from) -> bool:
        """Match a WhatsApp identifier against an allowlist across phone/LID forms. Inbound senders
        arrive as ``<id>@lid`` while allowlists hold phone numbers (or vice versa), so resolve both
        sides through the bridge's lid-mapping files via ``gateway.whatsapp_identity``."""
        if not allow_from:
            return False
        if candidate in allow_from:
            return True
        from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier
        candidate_aliases = expand_whatsapp_aliases(candidate)
        if not candidate_aliases:
            return False
        return any(
            entry == "*"
            or normalize_whatsapp_identifier(entry) in candidate_aliases
            or expand_whatsapp_aliases(entry) & candidate_aliases
            for entry in allow_from
        )

    def _is_dm_allowed(self, sender_id: str) -> bool:
        """Strict DM authorization — pairing does not imply access."""
        if self._dm_policy == "allowlist":
            return self._matches_whatsapp_allowlist(sender_id, self._live_dm_allow_from())
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        """Whether a DM may reach the gateway intake (pairing handshake path)."""
        principal = str(sender_id or "").strip()
        if not principal:
            return False
        if self._dm_policy == "allowlist":
            return self._matches_whatsapp_allowlist(principal, self._live_dm_allow_from())
        if self._dm_policy == "pairing":
            return True
        return self._dm_policy == "open" and self._open_dm_opted_in()

    def _is_group_allowed(self, chat_id: str) -> bool:
        """Check whether a group chat should be processed."""
        if self._group_policy == "allowlist":
            return self._matches_whatsapp_allowlist(chat_id, self._group_allow_from)
        return self._group_policy == "open"

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = (_get_wsecret("WHATSAPP_MENTION_PATTERNS", default="") or "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    # Plain text: one pattern per line, else comma-separated.
                    patterns = [p.strip() for p in raw.splitlines() if p.strip()]
                    patterns = patterns or [p.strip() for p in raw.split(",") if p.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning("[%s] whatsapp mention_patterns must be a list or string; got %s", self.name, type(patterns).__name__)
            return []
        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid WhatsApp mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d WhatsApp mention pattern(s)", self.name, len(compiled))
        return compiled

    def _bot_ids_from_message(self, data: Dict[str, Any]) -> set[str]:
        return {nid for c in (data.get("botIds") or []) if (nid := self._normalize_whatsapp_id(c))}

    def _message_is_reply_to_bot(self, data: Dict[str, Any]) -> bool:
        quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
        return bool(quoted_participant) and quoted_participant in self._bot_ids_from_message(data)

    def _message_mentions_bot(self, data: Dict[str, Any]) -> bool:
        bot_ids = self._bot_ids_from_message(data)
        if not bot_ids:
            return False
        mentioned = {nid for c in (data.get("mentionedIds") or []) if (nid := self._normalize_whatsapp_id(c))}
        if mentioned & bot_ids:
            return True
        lower_body = str(data.get("body") or "").lower()
        return any(
            bare and (f"@{bare}" in lower_body or bare in lower_body)
            for bare in (bot_id.split("@", 1)[0].lower() for bot_id in bot_ids)
        )

    def _message_matches_mention_patterns(self, data: Dict[str, Any]) -> bool:
        body = str(data.get("body") or "")
        return any(pattern.search(body) for pattern in self._mention_patterns or ())

    def _clean_bot_mention_text(self, text: str, data: Dict[str, Any]) -> str:
        if not text:
            return text
        cleaned = text
        for bot_id in self._bot_ids_from_message(data):
            bare_id = bot_id.split("@", 1)[0]
            if bare_id:
                cleaned = re.sub(rf"@{re.escape(bare_id)}\b[,:\-]*\s*", "", cleaned)
        return cleaned.strip() or text

    def _should_process_message(self, data: Dict[str, Any]) -> bool:
        chat_id = str(data.get("chatId") or "")
        # Broadcast pseudo-chats are filtered even in self-chat mode (fromMe events).
        if self._is_broadcast_chat(chat_id):
            return False
        if not data.get("isGroup", False):
            # DMs that pass the policy gate are always processed
            return self._is_dm_intake_allowed(str(data.get("senderId") or data.get("from") or ""))
        if not self._is_group_allowed(chat_id):
            return False
        # Group messages: check mention / free-response settings
        if chat_id in self._whatsapp_free_response_chats() or not self._whatsapp_require_mention():
            return True
        return (
            str(data.get("body") or "").strip().startswith("/")
            or self._message_is_reply_to_bot(data)
            or self._message_mentions_bot(data)
            or self._message_matches_mention_patterns(data)
        )

    # ------------------------------------------------------------------ formatting
    def format_message(self, content: str) -> str:
        """Convert markdown to WhatsApp syntax (*bold*, _italic_, ~strike~); fenced and
        inline code are protected via placeholder substitution."""
        if not content:
            return content
        result, fences = _stash(r"```[\s\S]*?```", self._sanitize_outbound_text(content), "FENCE")
        result, codes = _stash(r"`[^`\n]+`", result, "CODE")
        # Italic *text* → _text_ BEFORE bold so **bold** doesn't become italic;
        # lookarounds skip list bullets and bold delimiters.
        result = re.sub(r"(?<!\*)\*(?!\s|\*)([^*\n]*?\S[^*\n]*?)\*(?!\*)", r"_\1_", result)
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
        result = re.sub(r"__(.+?)__", r"*\1*", result)
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)
        result = re.sub(r"^#{1,6}\s+(.+)$", _header_to_bold, result, flags=re.MULTILINE)
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)  # [text](url) → text (url)
        for tag, saved in (("FENCE", fences), ("CODE", codes)):
            for i, original in enumerate(saved):
                result = result.replace(f"\x00{tag}{i}\x00", original)
        return result


def resolve_whatsapp_bridge_dir() -> Path:
    """Bridge directory for CLI and adapter. A read-only install tree (e.g. Docker
    /opt/hermes) is mirrored to HERMES_HOME so npm install works."""
    import shutil
    from pathlib import Path as _Path
    from hermes_constants import get_hermes_home
    install_bridge = _Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge"
    hermes_home_bridge = get_hermes_home() / "scripts" / "whatsapp-bridge"
    try:
        (install_bridge / ".write_test").touch()
        (install_bridge / ".write_test").unlink()
        return install_bridge
    except OSError:
        pass
    if hermes_home_bridge.exists():
        return hermes_home_bridge
    try:
        hermes_home_bridge.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(install_bridge, hermes_home_bridge, dirs_exist_ok=False)
        return hermes_home_bridge
    except Exception:
        return install_bridge
