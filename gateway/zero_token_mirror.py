"""Zero-token cross-platform mirroring for gateway conversations.

This module is deliberately deterministic: it only reads config and calls
platform adapters.  It must never instantiate AIAgent or call an LLM.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

ANY_THREAD = "*"
_LABELS_FILE = "zero-token-mirror-labels.json"
_LABEL_LOCK = threading.Lock()
_MAX_TITLE_CHARS = 12


@dataclass(frozen=True)
class MirrorEndpoint:
    platform: str
    chat_id: str
    thread_id: str | None = None


@dataclass(frozen=True)
class MirrorPair:
    name: str
    endpoints: tuple[MirrorEndpoint, ...]
    mirror_user_messages: bool = True
    mirror_assistant_messages: bool = True


def load_mirror_pairs(config: dict[str, Any] | None) -> tuple[MirrorPair, ...]:
    """Parse ``gateway.deterministic_mirrors`` from raw config.yaml data.

    Accepts either the full config dict (``{"gateway": ...}``) or the
    already-extracted ``deterministic_mirrors`` mapping from ``GatewayConfig``.
    """
    if not isinstance(config, dict):
        return ()
    if "gateway" in config:
        gateway_cfg = config.get("gateway") or {}
        if not isinstance(gateway_cfg, dict):
            return ()
        mirror_cfg = gateway_cfg.get("deterministic_mirrors") or {}
    else:
        mirror_cfg = config
    if not isinstance(mirror_cfg, dict) or not _coerce_bool(mirror_cfg.get("enabled"), False):
        return ()

    pairs: list[MirrorPair] = []
    for raw_pair in mirror_cfg.get("pairs") or []:
        if not isinstance(raw_pair, dict):
            continue
        endpoints: list[MirrorEndpoint] = []
        for raw_ep in raw_pair.get("endpoints") or []:
            if not isinstance(raw_ep, dict):
                continue
            platform = str(raw_ep.get("platform") or "").strip().lower()
            chat_id = str(raw_ep.get("chat_id") or "").strip()
            if not platform or not chat_id:
                continue
            thread_id_raw = raw_ep.get("thread_id")
            thread_id = str(thread_id_raw).strip() if thread_id_raw not in (None, "") else None
            endpoints.append(MirrorEndpoint(platform=platform, chat_id=chat_id, thread_id=thread_id))
        if len(endpoints) < 2:
            continue
        pairs.append(
            MirrorPair(
                name=str(raw_pair.get("name") or "mirror"),
                endpoints=tuple(endpoints),
                mirror_user_messages=_coerce_bool(raw_pair.get("mirror_user_messages"), True),
                mirror_assistant_messages=_coerce_bool(raw_pair.get("mirror_assistant_messages"), True),
            )
        )
    return tuple(pairs)


def targets_for_source(
    config: dict[str, Any] | None,
    source: SessionSource,
    *,
    assistant: bool = False,
) -> tuple[MirrorEndpoint, ...]:
    """Return configured endpoints that should receive a mirror of ``source``."""
    source_endpoint = _endpoint_from_source(source)
    if source_endpoint is None:
        return ()
    targets: list[MirrorEndpoint] = []
    for pair in load_mirror_pairs(config):
        if assistant and not pair.mirror_assistant_messages:
            continue
        if not assistant and not pair.mirror_user_messages:
            continue
        matched = [ep for ep in pair.endpoints if _endpoint_matches(ep, source_endpoint)]
        if not matched:
            continue
        for endpoint in pair.endpoints:
            if any(_same_endpoint(endpoint, m) for m in matched):
                continue
            # A wildcard thread endpoint represents "all concrete threads in
            # this chat" as sources.  It is not a concrete destination: a
            # message from the paired non-thread endpoint cannot know which
            # Slack/Telegram thread to target without an explicit mapping.
            if endpoint.thread_id == ANY_THREAD:
                continue
            targets.append(endpoint)
    return tuple(targets)


def canonical_mirror_session_key(config: dict[str, Any] | None, source: SessionSource) -> str | None:
    """Return a shared session key for a configured cross-platform mirror lane.

    This is the token-saving part of mirroring: paired Slack threads and
    Telegram topics/DM lanes must resolve to the same Hermes session, otherwise
    identical mirrored context is stored twice and later re-read twice.
    """
    source_endpoint = _endpoint_from_source(source)
    if source_endpoint is None:
        return None
    for pair in load_mirror_pairs(config):
        matched = [endpoint for endpoint in pair.endpoints if _endpoint_matches(endpoint, source_endpoint)]
        if matched:
            # A wildcard pair can only be canonicalized from the concrete
            # wildcard-side source thread.  The paired non-thread endpoint has
            # no way to identify which concrete thread it should share.
            if any(endpoint.thread_id == ANY_THREAD for endpoint in pair.endpoints) and not any(
                endpoint.thread_id == ANY_THREAD for endpoint in matched
            ):
                return None
            safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pair.name.strip())
            if not safe_name:
                safe_name = "mirror"
            if any(endpoint.thread_id == ANY_THREAD for endpoint in matched):
                source_key = _safe_session_fragment(
                    f"{source_endpoint.platform}:{source_endpoint.chat_id}:{source_endpoint.thread_id or 'main'}"
                )
                return f"agent:main:mirror:{safe_name}:{source_key}"
            return f"agent:main:mirror:{safe_name}"
    return None


def resolve_gateway_session_key(
    config: dict[str, Any] | None,
    source: SessionSource,
    *,
    group_sessions_per_user: bool = True,
    thread_sessions_per_user: bool = False,
) -> str:
    """Return the canonical gateway session key, including mirror aliases."""
    mirror_key = canonical_mirror_session_key(config, source)
    if mirror_key:
        return mirror_key
    return build_session_key(
        source,
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


async def mirror_user_message(gateway: Any, source: SessionSource, text: str) -> None:
    """Mirror a user-originated inbound message to the paired endpoint(s)."""
    if not text:
        return
    try:
        config = gateway._load_config() if hasattr(gateway, "_load_config") else None
    except Exception:
        config = None
    for target in targets_for_source(config, source, assistant=False):
        label = _label_for_source(config, source, seed_text=text)
        content = _format_mirror("사용자", source, text, label=label)
        await _send_to_endpoint(gateway, target, content)


async def mirror_assistant_message(gateway: Any, source: SessionSource, text: str) -> None:
    """Mirror an assistant response to the paired endpoint(s)."""
    if not text:
        return
    try:
        config = gateway._load_config() if hasattr(gateway, "_load_config") else None
    except Exception:
        config = None
    for target in targets_for_source(config, source, assistant=True):
        label = _label_for_source(config, source)
        content = _format_mirror("Leo", source, text, label=label)
        await _send_to_endpoint(gateway, target, content)


async def _send_to_endpoint(gateway: Any, endpoint: MirrorEndpoint, content: str) -> None:
    try:
        platform = Platform(endpoint.platform)
        adapter = gateway.adapters.get(platform)
        if adapter is None:
            logger.debug("mirror target adapter unavailable: %s", endpoint.platform)
            return
        metadata: dict[str, Any] = {"mirror": True, "notify": False}
        if endpoint.thread_id and endpoint.thread_id != ANY_THREAD:
            metadata["thread_id"] = endpoint.thread_id
        await adapter.send(endpoint.chat_id, content, metadata=metadata)
    except Exception:
        logger.debug("zero-token mirror send failed", exc_info=True)


def _endpoint_from_source(source: SessionSource | None) -> MirrorEndpoint | None:
    if source is None or source.platform is None or not source.chat_id:
        return None
    platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
    thread_id = str(source.thread_id) if source.thread_id not in (None, "") else None
    return MirrorEndpoint(platform=platform.lower(), chat_id=str(source.chat_id), thread_id=thread_id)


def _endpoint_matches(configured: MirrorEndpoint, actual: MirrorEndpoint) -> bool:
    if configured.platform != actual.platform or configured.chat_id != actual.chat_id:
        return False
    # If the config pins a thread/topic, require that exact lane.  If it does
    # not, the whole chat is mirrored.  A literal "*" means mirror every
    # thread in the chat while keeping each source thread as a separate session.
    if configured.thread_id == ANY_THREAD:
        return actual.thread_id not in (None, "")
    if configured.thread_id is not None:
        return configured.thread_id == actual.thread_id
    return True


def _same_endpoint(left: MirrorEndpoint, right: MirrorEndpoint) -> bool:
    return (
        left.platform == right.platform
        and left.chat_id == right.chat_id
        and left.thread_id == right.thread_id
    )


def _mirror_label_key(config: dict[str, Any] | None, source: SessionSource) -> str | None:
    key = canonical_mirror_session_key(config, source)
    if key:
        return key
    endpoint = _endpoint_from_source(source)
    if endpoint is None:
        return None
    return f"mirror:{endpoint.platform}:{endpoint.chat_id}:{endpoint.thread_id or 'main'}"


def _label_for_source(
    config: dict[str, Any] | None,
    source: SessionSource,
    *,
    seed_text: str | None = None,
) -> str | None:
    """Return a stable human-visible mirror label without using an LLM.

    The first user message creates a compact label like ``00001 미러링 설정``.
    Later user/assistant mirror copies reuse that label from disk so Slack and
    Telegram stay easy to correlate without adding any prompt tokens.
    """
    key = _mirror_label_key(config, source)
    if not key:
        return None
    try:
        with _LABEL_LOCK:
            labels = _read_labels_unlocked()
            existing = labels.get(key)
            if isinstance(existing, dict) and existing.get("label"):
                return str(existing["label"])

            index = _next_label_index(labels)
            title = _summarize_text(seed_text) or _fallback_title(source)
            label = f"{index:05d} {title}"
            labels[key] = {"label": label, "title": title, "index": index}
            _write_labels_unlocked(labels)
            return label
    except Exception:
        logger.debug("failed to resolve zero-token mirror label", exc_info=True)
        return None


def _labels_path():
    return get_hermes_home() / _LABELS_FILE


def _read_labels_unlocked() -> dict[str, Any]:
    path = _labels_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.debug("failed to read zero-token mirror labels", exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _write_labels_unlocked(labels: dict[str, Any]) -> None:
    path = _labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_replace(tmp_path, path)


def _next_label_index(labels: dict[str, Any]) -> int:
    max_index = 0
    for value in labels.values():
        if isinstance(value, dict):
            try:
                max_index = max(max_index, int(value.get("index") or 0))
            except (TypeError, ValueError):
                continue
    return max_index + 1


def _summarize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\[Replying to:.*?\]\s*", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"^\[[^\]]{1,20}\]\s*", "", cleaned.strip())
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[`*_~>#|]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -—:：,.!?。\n\t")
    if not cleaned:
        return ""
    first_sentence = re.split(r"[.!?。！？\n]", cleaned, maxsplit=1)[0].strip()
    if first_sentence:
        cleaned = first_sentence
    return cleaned[:_MAX_TITLE_CHARS]


def _fallback_title(source: SessionSource) -> str:
    if source.chat_name:
        return str(source.chat_name)[:_MAX_TITLE_CHARS]
    platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
    return f"{_platform_label(platform)} 세션"[:_MAX_TITLE_CHARS]


def _format_mirror(kind: str, source: SessionSource, text: str, *, label: str | None = None) -> str:
    platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
    session_line = f"🧭 세션: {label}\n" if label else ""
    label_platform = _platform_label(platform)
    channel = _format_channel(source)
    sender = source.user_name or source.user_id or kind

    if kind == "Leo":
        return (
            f"🤖 *Leo 답변 미러*\n"
            f"{session_line}"
            f"↪ 원본: {label_platform} · {channel}\n"
            f"⚙️ agent 재입력 없음 / 답변 1회 생성\n\n"
            f"{text}"
        )

    return (
        f"👤 *사용자 메시지 미러*\n"
        f"{session_line}"
        f"↪ 원본: {label_platform} · {channel}\n"
        f"🙋 작성자: {sender}\n"
        f"⚙️ LLM 재입력 없음\n\n"
        f"{text}"
    )


def _platform_label(platform: str) -> str:
    if platform == "slack":
        return "Slack"
    if platform == "telegram":
        return "Telegram"
    return platform


def _safe_session_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()) or "source"


def _format_channel(source: SessionSource) -> str:
    chat = str(source.chat_id or "unknown")
    thread = str(source.thread_id) if source.thread_id not in (None, "") else "main"
    return f"chat={chat} / thread={thread}"
