"""Pure helpers for Slack shared-message (forwarded-message) payloads.

Slack puts a shared message in ``attachments`` and reuses the same flags for
an automatic preview of a message that was already written by the bot.  Keep
that distinction and the nested-file merging in one place so live messages,
thread history, and thread-root media use identical rules.
"""

from collections.abc import Iterable, Mapping
from typing import Any


_MESSAGE_UNFURL_FLAGS = ("is_share", "is_msg_unfurl", "is_reply_unfurl")
_FILE_ID_FIELDS = ("id", "file_id")
_FILE_URL_FIELDS = ("url_private_download", "url_private", "permalink")


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _message_sources(attachment: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield an attachment and any nested message objects once each."""
    pending = [attachment]
    seen: set[int] = set()
    while pending:
        source = pending.pop(0)
        marker = id(source)
        if marker in seen:
            continue
        seen.add(marker)
        yield source
        for key in ("message", "original_message"):
            nested = _as_mapping(source.get(key))
            if nested is not None:
                pending.append(nested)


def is_message_unfurl_attachment(attachment: Any) -> bool:
    """Return whether Slack marked *attachment* as a shared/message unfurl."""
    mapping = _as_mapping(attachment)
    return bool(mapping and any(mapping.get(flag) for flag in _MESSAGE_UNFURL_FLAGS))


def is_explicit_share_attachment(attachment: Any) -> bool:
    """Return whether Slack explicitly marked the attachment as a user share."""
    mapping = _as_mapping(attachment)
    return bool(mapping and mapping.get("is_share"))


def forwarded_author_id(attachment: Any) -> str:
    """Return the original author's Slack user id, when the payload includes it."""
    mapping = _as_mapping(attachment)
    if mapping is None:
        return ""
    for source in _message_sources(mapping):
        value = source.get("author_id")
        if value:
            return str(value)
    return ""


def is_self_echo_attachment(attachment: Any, bot_uid: str = "") -> bool:
    """Return whether a message-preview attachment repeats this bot's own message.

    ``is_share`` is an explicit user action.  Keep it even when the original
    author happens to be this bot: the user deliberately forwarded that
    message and the agent must see the quote.  The author check is only used
    for automatic/reply unfurls.  Missing author data fails open so a real
    forwarded message is not silently discarded.
    """
    return bool(
        is_message_unfurl_attachment(attachment)
        and not is_explicit_share_attachment(attachment)
        and bot_uid
        and forwarded_author_id(attachment) == str(bot_uid)
    )


def iter_forwarded_attachments(
    attachments: Any, bot_uid: str = ""
) -> Iterable[Mapping[str, Any]]:
    """Yield real shared-message attachments, excluding known self echoes."""
    if not isinstance(attachments, (list, tuple)):
        return
    for attachment in attachments:
        mapping = _as_mapping(attachment)
        if mapping is None:
            continue
        if is_message_unfurl_attachment(mapping) and not is_self_echo_attachment(mapping, bot_uid):
            yield mapping


def forwarded_attachment_text_values(attachment: Any) -> list[str]:
    """Return unique readable text fields from an attachment and nested message."""
    mapping = _as_mapping(attachment)
    if mapping is None:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for source in _message_sources(mapping):
        for key in ("pretext", "title", "text", "fallback"):
            value = source.get(key)
            if not value:
                continue
            rendered = str(value).strip()
            if rendered and rendered not in seen:
                seen.add(rendered)
                values.append(rendered)
        fields = source.get("fields")
        if isinstance(fields, (list, tuple)):
            for field in fields:
                field_mapping = _as_mapping(field)
                if field_mapping is None:
                    continue
                for key in ("title", "value"):
                    value = field_mapping.get(key)
                    if not value:
                        continue
                    rendered = str(value).strip()
                    if rendered and rendered not in seen:
                        seen.add(rendered)
                        values.append(rendered)
    return values


def forwarded_attachment_title(attachment: Any) -> str:
    mapping = _as_mapping(attachment)
    if mapping is None:
        return ""
    for source in _message_sources(mapping):
        value = source.get("title")
        if value:
            return str(value).strip()
    return ""


def forwarded_attachment_link(attachment: Any) -> str:
    mapping = _as_mapping(attachment)
    if mapping is None:
        return ""
    for source in _message_sources(mapping):
        for key in ("from_url", "title_link", "url"):
            value = source.get(key)
            if value:
                return str(value).strip()
    return ""


def forwarded_attachment_author(attachment: Any) -> str:
    mapping = _as_mapping(attachment)
    if mapping is None:
        return ""
    for source in _message_sources(mapping):
        for key in ("author_name", "username", "user_name"):
            value = source.get(key)
            if value:
                return str(value).strip()
        author = _as_mapping(source.get("author"))
        if author:
            for key in ("name", "display_name", "id"):
                value = author.get(key)
                if value:
                    return str(value).strip()
        author_id = source.get("author_id")
        if author_id:
            return str(author_id).strip()
    return ""


def forwarded_attachment_block_sets(attachment: Any) -> list[list[Mapping[str, Any]]]:
    """Return ``blocks``/``message_blocks`` payloads from all message objects."""
    mapping = _as_mapping(attachment)
    if mapping is None:
        return []
    result: list[list[Mapping[str, Any]]] = []
    seen: set[str] = set()
    for source in _message_sources(mapping):
        for key in ("message_blocks", "blocks"):
            value = source.get(key)
            if isinstance(value, Mapping) and isinstance(value.get("blocks"), list):
                value = value["blocks"]
            elif isinstance(value, Mapping):
                value = [value]
            if not isinstance(value, (list, tuple)):
                continue
            blocks = [block for block in value if isinstance(block, Mapping)]
            if not blocks:
                continue
            # Slack can include both ``message_blocks`` and a copied ``blocks``
            # field. Avoid rendering an identical payload twice.
            marker = repr(blocks)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(blocks)
    return result


def iter_forwarded_files(attachments: Any, bot_uid: str = "") -> Iterable[Mapping[str, Any]]:
    """Yield files nested in real forwarded attachments only."""
    for attachment in iter_forwarded_attachments(attachments, bot_uid=bot_uid):
        for source in _message_sources(attachment):
            files = source.get("files")
            if not isinstance(files, (list, tuple)):
                continue
            for file_obj in files:
                mapping = _as_mapping(file_obj)
                if mapping is not None:
                    yield mapping


def _file_identity(file_obj: Mapping[str, Any]) -> tuple[str, str] | None:
    for key in _FILE_ID_FIELDS:
        value = file_obj.get(key)
        if value:
            return "id", str(value)
    for key in _FILE_URL_FIELDS:
        value = file_obj.get(key)
        if value:
            return "url", str(value)
    return None


def _file_completeness(file_obj: Mapping[str, Any]) -> int:
    """Score records so a full file wins over a Slack Connect stub."""
    return sum(
        bool(file_obj.get(key))
        for key in (
            "id", "name", "title", "mimetype", "filetype", "size",
            "url_private_download", "url_private", "permalink", "file_access",
        )
    )


def merge_slack_files(
    event_files: Any, attachments: Any, bot_uid: str = ""
) -> list[dict[str, Any]]:
    """Merge direct and forwarded files, preserving order and removing duplicates.

    A complete record replaces a short record with the same Slack file id or
    URL. Files without a stable identity are retained rather than guessed to
    be duplicates.
    """
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}

    def add(file_obj: Mapping[str, Any]) -> None:
        record = dict(file_obj)
        identity = _file_identity(record)
        if identity is None:
            merged.append(record)
            return
        existing_position = positions.get(identity)
        if existing_position is None:
            positions[identity] = len(merged)
            merged.append(record)
            return
        if _file_completeness(record) > _file_completeness(merged[existing_position]):
            merged[existing_position] = record

    if isinstance(event_files, (list, tuple)):
        for file_obj in event_files:
            mapping = _as_mapping(file_obj)
            if mapping is not None:
                add(mapping)
    for file_obj in iter_forwarded_files(attachments, bot_uid=bot_uid):
        add(file_obj)
    return merged


def forwarded_attachment_has_file_reference(attachment: Any) -> bool:
    """Return whether a forwarded attachment has a URL or resolvable file id."""
    mapping = _as_mapping(attachment)
    if mapping is None:
        return False
    for source in _message_sources(mapping):
        files = source.get("files")
        if not isinstance(files, (list, tuple)):
            continue
        for file_obj in files:
            file_mapping = _as_mapping(file_obj)
            if file_mapping is None:
                continue
            has_url = any(file_mapping.get(key) for key in _FILE_URL_FIELDS)
            can_resolve = (
                file_mapping.get("file_access") == "check_file_info"
                and any(file_mapping.get(key) for key in _FILE_ID_FIELDS)
            )
            if has_url or can_resolve:
                return True
    return False
