from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_constants import get_hermes_home


REQUIRED_TOPICS = (
    "operator",
    "briefings",
    "alerts",
    "input",
    "reviews",
    "content",
    "learning",
)


@dataclass(frozen=True)
class TelegramTopic:
    key: str
    title: str
    chat_id: str
    thread_id: str
    contract: Optional[str] = None


@dataclass(frozen=True)
class TelegramTopology:
    chat_id: str
    topics: dict[str, TelegramTopic]
    general_thread_ids: tuple[Optional[str], ...]
    temporary_thread_ids: tuple[str, ...]

    def topic(self, key: str) -> Optional[TelegramTopic]:
        return self.topics.get(str(key).strip().lower())


def _topology_path(home: Optional[Path] = None) -> Path:
    return Path(home or get_hermes_home()) / "telegram_topology.yaml"


def _coerce_thread_id(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ValueError(f"{field} must be a numeric string")
    return text


def validate_telegram_topology(data: dict[str, Any]) -> TelegramTopology:
    if not isinstance(data, dict):
        raise ValueError("telegram_topology.yaml must contain a mapping")
    if data.get("version") != 1:
        raise ValueError("telegram topology version must be 1")
    if str(data.get("platform", "")).strip().lower() != "telegram":
        raise ValueError("telegram topology platform must be telegram")

    chat_id = str(data.get("chat_id", "")).strip()
    if not chat_id:
        raise ValueError("telegram topology chat_id is required")

    raw_topics = data.get("topics")
    if not isinstance(raw_topics, dict):
        raise ValueError("telegram topology topics must be a mapping")

    missing = [key for key in REQUIRED_TOPICS if key not in raw_topics]
    extra = [key for key in raw_topics if key not in REQUIRED_TOPICS]
    if missing:
        raise ValueError(f"missing required telegram topics: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected telegram topics: {', '.join(extra)}")

    excluded = data.get("excluded") or {}
    general = tuple(
        None if item is None else str(item).strip()
        for item in (excluded.get("general") or [])
    )
    temporary = tuple(str(item).strip() for item in (excluded.get("temporary") or []))

    topics: dict[str, TelegramTopic] = {}
    seen_thread_ids: set[str] = set()
    forbidden = {tid for tid in general if tid is not None} | set(temporary)

    for key in REQUIRED_TOPICS:
        raw = raw_topics[key]
        if not isinstance(raw, dict):
            raise ValueError(f"topic {key} must be a mapping")
        thread_id = _coerce_thread_id(raw.get("thread_id"), field=f"topics.{key}.thread_id")
        if thread_id in forbidden:
            raise ValueError(f"topic {key} uses excluded thread_id {thread_id}")
        if thread_id in seen_thread_ids:
            raise ValueError(f"duplicate thread_id {thread_id}")
        seen_thread_ids.add(thread_id)
        title = str(raw.get("title", "")).strip()
        if not title:
            raise ValueError(f"topics.{key}.title is required")
        contract = str(raw.get("contract", "")).strip() or None
        topics[key] = TelegramTopic(
            key=key,
            title=title,
            chat_id=chat_id,
            thread_id=thread_id,
            contract=contract,
        )

    return TelegramTopology(
        chat_id=chat_id,
        topics=topics,
        general_thread_ids=general,
        temporary_thread_ids=temporary,
    )


def load_telegram_topology(home: Optional[Path] = None) -> Optional[TelegramTopology]:
    path = _topology_path(home)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return validate_telegram_topology(data)


def operator_target_for_command(source: Any, home: Optional[Path] = None) -> Optional[dict[str, str]]:
    topology = load_telegram_topology(home)
    if topology is None:
        return None

    platform = getattr(source, "platform", None)
    platform_value = getattr(platform, "value", platform)
    if str(platform_value).lower() != "telegram":
        return None
    if str(getattr(source, "chat_id", "")).strip() != topology.chat_id:
        return None

    thread_id = getattr(source, "thread_id", None)
    if thread_id is None:
        return None
    thread_id = str(thread_id).strip()

    main_thread_ids = {topic.thread_id for topic in topology.topics.values()}
    operator = topology.topic("operator")
    if operator is None:
        return None
    if thread_id not in main_thread_ids:
        return None
    if thread_id == operator.thread_id:
        return None

    return {"chat_id": operator.chat_id, "thread_id": operator.thread_id}


def topic_contract_for_source(source: Any, home: Optional[Path] = None) -> Optional[str]:
    """Return the configured behavioral contract for an exact Telegram topic."""
    topology = load_telegram_topology(home)
    if topology is None:
        return None

    platform = getattr(source, "platform", None)
    platform_value = getattr(platform, "value", platform)
    if str(platform_value).lower() != "telegram":
        return None
    if str(getattr(source, "chat_id", "")).strip() != topology.chat_id:
        return None

    thread_id = getattr(source, "thread_id", None)
    if thread_id is None:
        return None
    thread_id = str(thread_id).strip()
    for topic in topology.topics.values():
        if topic.thread_id == thread_id:
            return topic.contract
    return None
