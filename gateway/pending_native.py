"""Data-only native input identity and cached attachment fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, TYPE_CHECKING

from gateway.config import Platform

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PendingAttachment:
    path: str
    digest: str | None


@dataclass(frozen=True)
class PendingNativeInput:
    platform: Platform
    content: dict[str, Any]
    attachments: tuple[PendingAttachment, ...]

    @classmethod
    def capture(
        cls, event: MessageEvent, content: dict[str, Any]
    ) -> PendingNativeInput:
        from gateway.shutdown_pending_codec import _file_digest

        return cls(
            event.source.platform,
            json.loads(json.dumps(content, allow_nan=False)),
            tuple(
                PendingAttachment(path, _file_digest(path)) for path in event.media_urls
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "content": self.content,
            "attachments": [
                {"path": item.path, "digest": item.digest} for item in self.attachments
            ],
        }

    @classmethod
    def from_payload(cls, value: Any, event: MessageEvent) -> PendingNativeInput:
        if (
            not isinstance(value, dict)
            or set(value) != {"platform", "content", "attachments"}
            or value["platform"] != event.source.platform.value
            or not isinstance(value["content"], dict)
            or not isinstance(value["attachments"], list)
        ):
            raise ValueError("pending native identity differs from its event")
        attachments = []
        for item in value["attachments"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "digest"}
                or not isinstance(item["path"], str)
                or item["digest"] is not None
                and (
                    not isinstance(item["digest"], str)
                    or not _DIGEST.fullmatch(item["digest"])
                )
            ):
                raise ValueError("pending native attachment fingerprint is invalid")
            attachments.append(PendingAttachment(item["path"], item["digest"]))
        if [item.path for item in attachments] != event.media_urls:
            raise ValueError("pending native attachment paths differ from their event")
        return cls(event.source.platform, value["content"], tuple(attachments))

    def attachments_available(self, paths: list[str]) -> bool:
        from gateway.shutdown_pending_codec import _file_digest

        return [item.path for item in self.attachments] == paths and all(
            item.digest is not None and _file_digest(item.path) == item.digest
            for item in self.attachments
        )
