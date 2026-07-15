"""Strict reader contract for the public Support Ops alias projections.

The privileged Canonical writer export is intentionally *not* readable by the
gateway.  A separate ``muncho-projector`` process derives this small document
from the append-only log and publishes only normalized alias-to-member pairs
plus non-semantic integrity receipts.

This module knows only the safe projection schema.  It has no database client,
writer socket, global export path, or event-folding code.
"""

from __future__ import annotations

import datetime as dt
import copy
import hashlib
import json
import os
import pathlib
import re
import stat
from collections.abc import Callable, Iterable, Mapping
from typing import Any


ALIAS_PROJECTION_SCHEMA = "canonical_brain.projection.support_ops_aliases.v2"
ALIAS_PROJECTION_RECEIPT_SCHEMA = (
    "canonical_brain.projection.support_ops_aliases_receipt.v2"
)
DEFAULT_PUBLIC_ALIAS_PROJECTION_PATH = pathlib.Path(
    "/var/lib/muncho-projector/public/team-member-aliases.json"
)
MAX_ALIAS_PROJECTION_BYTES = 1_048_576
MAX_PROJECTED_ALIASES = 10_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_TOP_LEVEL_KEYS = frozenset(
    {"schema", "aliases", "channel_aliases", "receipt"}
)
_CHANNEL_ALIAS_ROOT_KEYS = frozenset(
    {"guild_id", "target_type", "channel_id"}
)
_CHANNEL_ALIAS_THREAD_KEYS = _CHANNEL_ALIAS_ROOT_KEYS | {"parent_channel_id"}
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{16,19}$")
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "source_export_sha256",
        "source_event_count",
        "alias_event_count",
        "alias_count",
        "last_alias_event_id",
        "last_alias_event_at",
        "accepted_event_ids_sha256",
        "projection_sha256",
    }
)


class AliasProjectionError(ValueError):
    """The safe alias projection failed its exact mechanical contract."""


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or not key or key in result:
            raise AliasProjectionError("alias_projection_json_keys_invalid")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def projection_payload_sha256(
    aliases: Mapping[str, str],
    channel_aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Digest the exact public payload, excluding its self-describing receipt."""

    payload = {
        "schema": ALIAS_PROJECTION_SCHEMA,
        "aliases": dict(aliases),
        "channel_aliases": {
            alias: dict(target)
            for alias, target in (channel_aliases or {}).items()
        },
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def accepted_event_ids_sha256(event_ids: Iterable[str]) -> str:
    """Return a deterministic receipt without exposing individual event ids."""

    return hashlib.sha256(_canonical_json_bytes(list(event_ids))).hexdigest()


def validate_alias_projection_document(
    value: Any,
    *,
    normalize_alias: Callable[[str], str],
    valid_member_keys: Iterable[str],
    static_alias_member_keys: Mapping[str, str],
    expected_channel_guild_id: str | None = None,
    static_channel_alias_ids: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate and return the minimal alias mapping.

    All keys are exact, all counters use real integers (never booleans), and
    every alias is already normalized.  Invalid projections are rejected as a
    whole so a partial/tampered document cannot influence routing.
    """

    valid_keys = frozenset(str(key) for key in valid_member_keys)
    static_channels = dict(static_channel_alias_ids or {})
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise AliasProjectionError("alias_projection_envelope_invalid")
    if value.get("schema") != ALIAS_PROJECTION_SCHEMA:
        raise AliasProjectionError("alias_projection_schema_invalid")

    aliases_raw = value.get("aliases")
    if not isinstance(aliases_raw, Mapping) or len(aliases_raw) > MAX_PROJECTED_ALIASES:
        raise AliasProjectionError("alias_projection_aliases_invalid")
    aliases: dict[str, str] = {}
    for raw_alias, raw_member_key in aliases_raw.items():
        if not isinstance(raw_alias, str) or not isinstance(raw_member_key, str):
            raise AliasProjectionError("alias_projection_entry_invalid")
        if (
            not raw_alias
            or len(raw_alias) > 200
            or len(raw_alias.encode("utf-8")) > 512
            or normalize_alias(raw_alias) != raw_alias
            or raw_member_key not in valid_keys
            or raw_alias in static_channels
            or (
                raw_alias in static_alias_member_keys
                and static_alias_member_keys[raw_alias] != raw_member_key
            )
        ):
            raise AliasProjectionError("alias_projection_entry_invalid")
        aliases[raw_alias] = raw_member_key

    channel_aliases_raw = value.get("channel_aliases")
    if (
        not isinstance(channel_aliases_raw, Mapping)
        or len(channel_aliases_raw) > MAX_PROJECTED_ALIASES
    ):
        raise AliasProjectionError("alias_projection_channel_aliases_invalid")
    expected_guild_id = str(expected_channel_guild_id or "").strip()
    channel_aliases: dict[str, dict[str, str]] = {}
    for raw_alias, raw_target in channel_aliases_raw.items():
        if (
            not isinstance(raw_alias, str)
            or not isinstance(raw_target, Mapping)
        ):
            raise AliasProjectionError("alias_projection_channel_entry_invalid")
        guild_id = raw_target.get("guild_id")
        target_type = raw_target.get("target_type")
        channel_id = raw_target.get("channel_id")
        expected_target_keys = (
            _CHANNEL_ALIAS_THREAD_KEYS
            if target_type == "guild_thread"
            else _CHANNEL_ALIAS_ROOT_KEYS
        )
        parent_channel_id = raw_target.get("parent_channel_id")
        if (
            set(raw_target) != expected_target_keys
            or not raw_alias
            or len(raw_alias) > 200
            or len(raw_alias.encode("utf-8")) > 512
            or normalize_alias(raw_alias) != raw_alias
            or raw_alias in aliases
            or not isinstance(guild_id, str)
            or target_type not in {"guild_channel", "guild_thread"}
            or not isinstance(channel_id, str)
            or _SNOWFLAKE_RE.fullmatch(guild_id) is None
            or _SNOWFLAKE_RE.fullmatch(channel_id) is None
            or (expected_guild_id and guild_id != expected_guild_id)
            or (
                target_type == "guild_thread"
                and (
                    not isinstance(parent_channel_id, str)
                    or _SNOWFLAKE_RE.fullmatch(parent_channel_id) is None
                    or parent_channel_id == channel_id
                )
            )
            or (
                raw_alias in static_alias_member_keys
                or raw_alias in static_channels
                and (
                    static_channels[raw_alias] != channel_id
                    or target_type != "guild_channel"
                )
            )
        ):
            raise AliasProjectionError("alias_projection_channel_entry_invalid")
        channel_aliases[raw_alias] = {
            "guild_id": guild_id,
            "target_type": target_type,
            "channel_id": channel_id,
            **(
                {"parent_channel_id": parent_channel_id}
                if target_type == "guild_thread"
                else {}
            ),
        }

    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise AliasProjectionError("alias_projection_receipt_invalid")
    if receipt.get("schema") != ALIAS_PROJECTION_RECEIPT_SCHEMA:
        raise AliasProjectionError("alias_projection_receipt_schema_invalid")
    for key in (
        "source_export_sha256",
        "accepted_event_ids_sha256",
        "projection_sha256",
    ):
        if not isinstance(receipt.get(key), str) or not _SHA256_RE.fullmatch(
            receipt[key]
        ):
            raise AliasProjectionError("alias_projection_receipt_digest_invalid")

    counters: dict[str, int] = {}
    for key in ("source_event_count", "alias_event_count", "alias_count"):
        raw = receipt.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise AliasProjectionError("alias_projection_receipt_count_invalid")
        counters[key] = raw
    if not (
        counters["source_event_count"] >= counters["alias_event_count"]
        >= counters["alias_count"] == len(aliases) + len(channel_aliases)
    ):
        raise AliasProjectionError("alias_projection_receipt_count_invalid")

    last_id = receipt.get("last_alias_event_id")
    last_at = receipt.get("last_alias_event_at")
    if not isinstance(last_id, str) or not isinstance(last_at, str):
        raise AliasProjectionError("alias_projection_receipt_cursor_invalid")
    if counters["alias_event_count"] == 0:
        if last_id or last_at:
            raise AliasProjectionError("alias_projection_receipt_cursor_invalid")
    elif not _UUID_RE.fullmatch(last_id) or not last_at:
        raise AliasProjectionError("alias_projection_receipt_cursor_invalid")
    else:
        try:
            parsed_last_at = dt.datetime.fromisoformat(
                last_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise AliasProjectionError(
                "alias_projection_receipt_cursor_invalid"
            ) from exc
        if (
            parsed_last_at.tzinfo is None
            or parsed_last_at.utcoffset() != dt.timedelta(0)
        ):
            raise AliasProjectionError("alias_projection_receipt_cursor_invalid")

    if receipt["projection_sha256"] != projection_payload_sha256(
        aliases,
        channel_aliases,
    ):
        raise AliasProjectionError("alias_projection_payload_digest_mismatch")
    return aliases


def _read_stable_regular_file(path: pathlib.Path) -> bytes:
    path = pathlib.Path(path)
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
        or not 0 < before.st_size <= MAX_ALIAS_PROJECTION_BYTES
    ):
        raise AliasProjectionError("alias_projection_file_untrusted")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise AliasProjectionError("alias_projection_file_changed")
        chunks: list[bytes] = []
        remaining = MAX_ALIAS_PROJECTION_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or _file_identity(after) != _file_identity(before)
    ):
        raise AliasProjectionError("alias_projection_file_changed")
    return raw


def load_alias_projection_document(
    path: pathlib.Path,
    *,
    normalize_alias: Callable[[str], str],
    valid_member_keys: Iterable[str],
    static_alias_member_keys: Mapping[str, str],
    expected_channel_guild_id: str | None = None,
    static_channel_alias_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read one complete protected projection without writer/export access.

    Returning the validated envelope lets the isolated projector enforce
    append-only progress and perform an exact post-rename readback.  Gateway
    callers should normally use :func:`load_alias_projection`, which exposes
    only the minimal alias mapping.
    """

    raw = _read_stable_regular_file(pathlib.Path(path))
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AliasProjectionError("alias_projection_json_constant_invalid")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AliasProjectionError("alias_projection_json_invalid") from exc
    validate_alias_projection_document(
        value,
        normalize_alias=normalize_alias,
        valid_member_keys=valid_member_keys,
        static_alias_member_keys=static_alias_member_keys,
        expected_channel_guild_id=expected_channel_guild_id,
        static_channel_alias_ids=static_channel_alias_ids,
    )
    return copy.deepcopy(dict(value))


def load_alias_projection(
    path: pathlib.Path,
    *,
    normalize_alias: Callable[[str], str],
    valid_member_keys: Iterable[str],
    static_alias_member_keys: Mapping[str, str],
    expected_channel_guild_id: str | None = None,
    static_channel_alias_ids: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read one protected public projection without any writer/export access."""

    value = load_alias_projection_document(
        path,
        normalize_alias=normalize_alias,
        valid_member_keys=valid_member_keys,
        static_alias_member_keys=static_alias_member_keys,
        expected_channel_guild_id=expected_channel_guild_id,
        static_channel_alias_ids=static_channel_alias_ids,
    )
    return dict(value["aliases"])


def load_channel_alias_projection(
    path: pathlib.Path,
    *,
    normalize_alias: Callable[[str], str],
    valid_member_keys: Iterable[str],
    static_alias_member_keys: Mapping[str, str],
    expected_channel_guild_id: str,
    static_channel_alias_ids: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Read only exact Canonical alias-to-guild-channel mappings."""

    value = load_alias_projection_document(
        path,
        normalize_alias=normalize_alias,
        valid_member_keys=valid_member_keys,
        static_alias_member_keys=static_alias_member_keys,
        expected_channel_guild_id=expected_channel_guild_id,
        static_channel_alias_ids=static_channel_alias_ids,
    )
    return {
        alias: dict(target)
        for alias, target in value["channel_aliases"].items()
    }


__all__ = [
    "ALIAS_PROJECTION_RECEIPT_SCHEMA",
    "ALIAS_PROJECTION_SCHEMA",
    "AliasProjectionError",
    "DEFAULT_PUBLIC_ALIAS_PROJECTION_PATH",
    "accepted_event_ids_sha256",
    "load_alias_projection",
    "load_alias_projection_document",
    "load_channel_alias_projection",
    "projection_payload_sha256",
    "validate_alias_projection_document",
]
