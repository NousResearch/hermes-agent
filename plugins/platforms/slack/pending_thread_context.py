"""Original Slack thread dependencies for durable pending input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def thread_message_content(message: dict[str, Any]) -> dict[str, Any]:
    files = message.get("files") or []
    return {
        key: message.get(key)
        for key in (
            "ts",
            "user",
            "bot_id",
            "username",
            "subtype",
            "text",
            "blocks",
            "attachments",
        )
    } | {
        "files": [
            {key: file.get(key) for key in ("id", "name", "size", "mimetype")}
            for file in files
        ]
    }


@dataclass(frozen=True)
class SlackThreadMessage:
    message_id: str
    content: dict[str, Any]


@dataclass(frozen=True)
class SlackThreadFile:
    message_id: str
    file_id: str
    path: str


@dataclass(frozen=True)
class SlackThreadContext:
    text: str | None
    messages: tuple[SlackThreadMessage, ...]
    files: tuple[SlackThreadFile, ...]

    @classmethod
    def capture(
        cls, text: str | None, messages: list[dict], paths: list[str], thread_id: str
    ) -> SlackThreadContext:
        entries = tuple(
            SlackThreadMessage(message["ts"], thread_message_content(message))
            for message in messages
        )
        eligible = [
            (message["ts"], file["id"])
            for message in messages
            if message["ts"] == thread_id
            for file in message.get("files") or []
            if str(file.get("mimetype") or "").startswith("image/")
            and (file.get("url_private_download") or file.get("url_private"))
        ]
        files = tuple(
            SlackThreadFile(message_id, file_id, path)
            for (message_id, file_id), path in zip(eligible, paths)
        )
        if len(files) != len(paths):
            raise ValueError("Slack thread file identity is unavailable")
        return cls.from_payload(cls(text, entries, files).to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "messages": [
                {"message_id": entry.message_id, "content": entry.content}
                for entry in self.messages
            ],
            "files": [
                {
                    "message_id": file.message_id,
                    "file_id": file.file_id,
                    "path": file.path,
                }
                for file in self.files
            ],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> SlackThreadContext:
        from plugins.platforms.slack.adapter import _THREAD_ROOT_IMAGE_MAX

        if (
            not isinstance(payload, dict)
            or set(payload) != {"text", "messages", "files"}
            or payload["text"] is not None
            and not isinstance(payload["text"], str)
            or not isinstance(payload["messages"], list)
            or len(payload["messages"]) > 31
            or not isinstance(payload["files"], list)
            or len(payload["files"]) > _THREAD_ROOT_IMAGE_MAX
        ):
            raise ValueError("Slack pending thread context is invalid")
        messages = []
        for value in payload["messages"]:
            if (
                not isinstance(value, dict)
                or set(value) != {"message_id", "content"}
                or not isinstance(value["message_id"], str)
                or not value["message_id"]
                or not isinstance(value["content"], dict)
                or value["content"].get("ts") != value["message_id"]
            ):
                raise ValueError("Slack pending thread message identity is invalid")
            messages.append(SlackThreadMessage(value["message_id"], value["content"]))
        if payload["text"] and not messages:
            raise ValueError("Slack pending thread quotation has no native identities")
        if len({value.message_id for value in messages}) != len(messages):
            raise ValueError("Slack pending thread message identity is repeated")
        files = []
        for value in payload["files"]:
            if (
                not isinstance(value, dict)
                or set(value) != {"message_id", "file_id", "path"}
                or not all(isinstance(item, str) and item for item in value.values())
                or value["message_id"]
                not in {message.message_id for message in messages}
            ):
                raise ValueError("Slack pending thread file identity is invalid")
            message = next(
                message
                for message in messages
                if message.message_id == value["message_id"]
            )
            native_files = message.content.get("files")
            if not isinstance(native_files, list) or not any(
                isinstance(file, dict) and file.get("id") == value["file_id"]
                for file in native_files
            ):
                raise ValueError(
                    "Slack pending thread file differs from its native message"
                )
            files.append(
                SlackThreadFile(value["message_id"], value["file_id"], value["path"])
            )
        return cls(payload["text"], tuple(messages), tuple(files))
