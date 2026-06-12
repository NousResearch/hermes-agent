"""LFDM wiki password gateway shortcut.

This plugin handles the specific operational request “what/give/send me the wiki
password” without routing the message through an LLM provider. That avoids two
observed failure modes:

* Codex/OpenAI safety filters treating an authorized local wiki-password lookup
  as generic credential exfiltration.
* Fallback replay into DeepSeek V4 thinking mode tripping strict
  ``reasoning_content`` validation before the user receives the password.

The password is still read from the existing local-only source of truth
(``~/.config/sassy-wiki/password`` by default), is only sent to an already
authorized gateway user, and is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PASSWORD_PATH_ENV = "SASSY_WIKI_PASSWORD_FILE"
_ALLOWED_CHATS_ENV = "LFDM_WIKI_PASSWORD_ALLOWED_CHATS"
_DEFAULT_PASSWORD_PATH = "~/.config/sassy-wiki/password"

_PASSWORD_TERMS_RE = re.compile(r"\b(pass(?:word)?|pw)\b")
_WIKI_TERM_RE = re.compile(r"\bwiki\b")
_INTENT_RE = re.compile(
    r"\b(what(?:'s|\s+is)?|give|send|show|tell|share|get|need|current|latest)\b"
)
_DIAGNOSTIC_TERMS_RE = re.compile(
    r"\b(fix|workaround|debug|error|failed|failure|route|handler|why|postmortem|refus(?:e|al))\b"
)
_MENTION_RE = re.compile(r"<@[!&]?\d+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")


def _password_path() -> Path:
    raw = os.environ.get(_PASSWORD_PATH_ENV, _DEFAULT_PASSWORD_PATH)
    return Path(raw).expanduser()


def _normalized_text(text: str) -> str:
    lowered = _MENTION_RE.sub(" ", (text or "").lower())
    lowered = _NON_WORD_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _is_wiki_password_request(text: str) -> bool:
    """Return True for direct requests for the current wiki password."""
    normalized = _normalized_text(text)
    if not normalized:
        return False
    if not _WIKI_TERM_RE.search(normalized) or not _PASSWORD_TERMS_RE.search(normalized):
        return False

    # Short shorthands like “wiki password” / “wiki pw please” are intentional.
    short_tokens = normalized.split()
    if len(short_tokens) <= 6 and not _DIAGNOSTIC_TERMS_RE.search(normalized):
        return True

    # Longer messages need explicit request intent. This avoids catching
    # debugging/process messages like “fix the wiki password route”.
    return bool(_INTENT_RE.search(normalized))


def _allowed_in_chat(source: Any) -> bool:
    allowed = {
        item.strip()
        for item in os.environ.get(_ALLOWED_CHATS_ENV, "").split(",")
        if item.strip()
    }
    if not allowed:
        return True
    chat_id = str(getattr(source, "chat_id", "") or "")
    thread_id = str(getattr(source, "thread_id", "") or "")
    return chat_id in allowed or bool(thread_id and f"{chat_id}:{thread_id}" in allowed)


def _source_is_authorized(gateway: Any, source: Any) -> bool:
    if gateway is None or source is None:
        return False
    if not _allowed_in_chat(source):
        return False
    checker = getattr(gateway, "_is_user_authorized", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(source))
    except Exception as exc:  # pragma: no cover - defensive against gateway drift
        logger.warning("lfdm-wiki-password: auth check failed: %s", exc)
        return False


def _read_password() -> str:
    path = _password_path()
    if not path.exists():
        raise FileNotFoundError(f"wiki password file not found: {path}")
    if not path.is_file():
        raise OSError(f"wiki password path is not a file: {path}")
    password = path.read_text(encoding="utf-8").strip()
    if not password:
        raise ValueError(f"wiki password file is empty: {path}")
    return password


async def _send_direct(event: Any, gateway: Any, content: str) -> None:
    source = getattr(event, "source", None)
    adapter = getattr(gateway, "adapters", {}).get(getattr(source, "platform", None))
    if adapter is None:
        logger.warning("lfdm-wiki-password: no adapter for platform=%r", getattr(source, "platform", None))
        return

    reply_to: Optional[str] = None
    metadata = None
    try:
        anchor_fn = getattr(gateway, "_reply_anchor_for_event", None)
        if callable(anchor_fn):
            anchor = anchor_fn(event)
            reply_to = str(anchor) if anchor is not None else None
    except Exception:
        reply_to = None
    try:
        meta_fn = getattr(gateway, "_thread_metadata_for_source", None)
        if callable(meta_fn):
            metadata = meta_fn(source, reply_to)
    except Exception:
        metadata = None

    result = await adapter.send(
        getattr(source, "chat_id", None),
        content,
        reply_to=reply_to,
        metadata=metadata,
    )
    if result is not None and getattr(result, "success", True) is False:
        logger.warning("lfdm-wiki-password: send failed: %s", getattr(result, "error", "unknown"))


def _track_task(gateway: Any, task: "asyncio.Task[None]") -> None:
    task_set = getattr(gateway, "_background_tasks", None)
    if isinstance(task_set, set):
        task_set.add(task)

    def _done(done: "asyncio.Task[None]") -> None:
        if isinstance(task_set, set):
            task_set.discard(done)
        if done.cancelled():
            return
        try:
            done.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("lfdm-wiki-password: direct send task failed: %s", exc)

    task.add_done_callback(_done)


def _on_pre_gateway_dispatch(event: Any = None, gateway: Any = None, session_store: Any = None) -> Optional[dict]:
    del session_store
    text = getattr(event, "text", "") or ""
    if not _is_wiki_password_request(text):
        return None

    source = getattr(event, "source", None)
    if not _source_is_authorized(gateway, source):
        logger.info("lfdm-wiki-password: matched request but source is not authorized; falling through")
        return {"action": "allow"}

    try:
        content = _read_password()
    except Exception as exc:
        # This is an operational error; do not let the LLM hallucinate a secret.
        content = f"⚠️ Wiki password lookup failed: {exc}"

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("lfdm-wiki-password: no running event loop; falling through")
        return {"action": "allow"}

    task = loop.create_task(_send_direct(event, gateway, content))
    _track_task(gateway, task)
    return {"action": "skip", "reason": "lfdm_wiki_password_direct_response"}


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
