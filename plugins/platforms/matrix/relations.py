"""Interpret Matrix reply, thread and edit relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatrixRelation:
    thread_root: str | None = None
    reply_target: str | None = None
    thread_fallback_target: str | None = None
    is_edit: bool = False

    @classmethod
    def from_content(cls, relates_to: Any) -> MatrixRelation:
        if not isinstance(relates_to, dict):
            return cls()

        relation_type = relates_to.get("rel_type")
        if relation_type == "m.replace":
            return cls(is_edit=True)

        in_reply_to = relates_to.get("m.in_reply_to")
        target = in_reply_to.get("event_id") if isinstance(in_reply_to, dict) else None
        target = target if isinstance(target, str) and target else None
        if relation_type == "m.thread":
            root = relates_to.get("event_id")
            root = root if isinstance(root, str) and root else None
            if relates_to.get("is_falling_back"):
                return cls(thread_root=root, thread_fallback_target=target)
            return cls(thread_root=root, reply_target=target)

        return cls(reply_target=target)
