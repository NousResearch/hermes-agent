"""Fail-closed important-contact matching for gateway message nudges.

This module deliberately matches only explicit identity fields from
``SessionSource``. Message content is never used to infer importance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalized_set(value: Any) -> set[str]:
    return {_clean(item).lower() for item in _as_list(value) if _clean(item)}


def _candidate_rules(config: Any, platform: str) -> list[Mapping[str, Any]]:
    """Return rule dictionaries relevant to *platform*.

    Accepted shapes are intentionally small and explicit:
    - {"contacts": [{"platform": "sms", "user_ids": ["+1..."]}]}
    - {"sms": {"user_ids": ["+1..."]}}
    - {"sms": [{"user_ids": ["+1..."]}]}
    - [{"platform": "sms", "user_ids": ["+1..."]}]
    """
    if not config:
        return []

    rules: list[Mapping[str, Any]] = []
    if isinstance(config, Mapping):
        for item in _as_list(config.get("contacts")):
            if isinstance(item, Mapping):
                rules.append(item)

        platform_rules = config.get(platform)
        if isinstance(platform_rules, Mapping):
            rules.append(platform_rules)
        else:
            for item in _as_list(platform_rules):
                if isinstance(item, Mapping):
                    rules.append(item)
    else:
        for item in _as_list(config):
            if isinstance(item, Mapping):
                rules.append(item)

    relevant: list[Mapping[str, Any]] = []
    for rule in rules:
        rule_platforms = _normalized_set(
            rule.get("platforms", rule.get("platform"))
        )
        if rule_platforms and platform not in rule_platforms:
            continue
        relevant.append(rule)
    return relevant


def _has_stable_identity(source: Any) -> bool:
    return any(
        _clean(getattr(source, attr, None))
        for attr in ("user_id", "user_id_alt", "chat_id", "chat_id_alt")
    )


def is_important_sender(source: Any, config: Any) -> bool:
    """Return True only when *source* explicitly matches an important-contact rule.

    The matcher fails closed when the source has no stable identity. Display
    names are intentionally not positive match keys: they can be duplicated or
    spoofed across contacts, so only stable sender/chat identity fields can make
    a message important.
    """
    if source is None:
        return False

    platform = _platform_value(source)
    if not platform or not _has_stable_identity(source):
        return False

    identities = {
        "user_ids": _clean(getattr(source, "user_id", None)).lower(),
        "user_id_alts": _clean(getattr(source, "user_id_alt", None)).lower(),
        "chat_ids": _clean(getattr(source, "chat_id", None)).lower(),
        "chat_id_alts": _clean(getattr(source, "chat_id_alt", None)).lower(),
    }

    for rule in _candidate_rules(config, platform):
        for field, actual in identities.items():
            if actual and actual in _normalized_set(rule.get(field)):
                return True

    return False
