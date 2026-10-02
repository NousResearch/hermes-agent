"""Data-only native input identity and cached attachment fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    origin_path: str | None = None


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
                {
                    "path": item.path,
                    "digest": item.digest,
                    **(
                        {"origin_path": item.origin_path}
                        if item.origin_path is not None
                        else {}
                    ),
                }
                for item in self.attachments
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
                or set(item)
                not in ({"path", "digest"}, {"path", "digest", "origin_path"})
                or not isinstance(item["path"], str)
                or "origin_path" in item
                and (
                    not isinstance(item["origin_path"], str) or not item["origin_path"]
                )
                or item["digest"] is not None
                and (
                    not isinstance(item["digest"], str)
                    or not _DIGEST.fullmatch(item["digest"])
                )
            ):
                raise ValueError("pending native attachment fingerprint is invalid")
            attachments.append(
                PendingAttachment(item["path"], item["digest"], item.get("origin_path"))
            )
        if [item.path for item in attachments] != event.media_urls:
            raise ValueError("pending native attachment paths differ from their event")
        return cls(event.source.platform, value["content"], tuple(attachments))

    def rehome_attachments(self, paths: dict[str, str]) -> PendingNativeInput:
        return replace(
            self,
            attachments=tuple(
                replace(
                    item,
                    path=paths[item.path],
                    origin_path=item.origin_path or item.path,
                )
                if item.path in paths and paths[item.path] != item.path
                else item
                for item in self.attachments
            ),
        )

    def attachment_path(self, original_path: str) -> str | None:
        return next(
            (
                item.path
                for item in self.attachments
                if original_path in (item.path, item.origin_path)
            ),
            None,
        )

    def attachments_available(self, paths: list[str]) -> bool:
        from gateway.shutdown_pending_codec import _file_digest

        return [item.path for item in self.attachments] == paths and all(
            item.digest is not None and _file_digest(item.path) == item.digest
            for item in self.attachments
        )
