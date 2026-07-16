#!/usr/bin/env python3
"""Derive safe person/channel alias projections from Canonical Brain events.

This process is deliberately mechanical and is intended to run as the isolated
``muncho-projector`` identity.  It reads the protected writer export, accepts
only exact typed alias envelopes, and emits no summaries,
source references, actors, payload extras, or unrelated event content.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import pwd
import grp
import re
import stat
import uuid
from collections.abc import Mapping
from typing import Any

from gateway.support_ops_alias_projection import (
    ALIAS_PROJECTION_RECEIPT_SCHEMA,
    ALIAS_PROJECTION_SCHEMA,
    accepted_event_ids_sha256,
    load_alias_projection_document,
    projection_payload_sha256,
    validate_alias_projection_document,
)
from gateway.canonical_projection_export import (
    ProjectionExportError,
    validate_projection_export,
)
from gateway.support_ops_team_registry import (
    ALIAS_ENTRIES,
    APPROVED_LANE_ALIAS_ENTRIES,
    APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID,
    SKYVISION_GUILD_ID,
    STATIC_ALIAS_MEMBER_KEYS,
    STATIC_ALIAS_CHANNEL_IDS,
    TEAM_MEMBERS_BY_KEY,
    normalize_team_member_alias,
)


ALIAS_EVENT_TYPE = "person.alias.learned"
CHANNEL_ALIAS_EVENT_TYPE = "channel.alias.learned"
ALIAS_EVENT_TYPES = frozenset({ALIAS_EVENT_TYPE, CHANNEL_ALIAS_EVENT_TYPE})
ALIAS_PROJECTION_RUN_RECEIPT_SCHEMA = (
    "canonical_brain.projection.support_ops_aliases_run_receipt.v2"
)
PRODUCTION_WRITER_USER = "muncho-canonical-writer"
PRODUCTION_PROJECTOR_USER = "muncho-projector"
PRODUCTION_PROJECTOR_GROUP = "muncho-projector"
PRODUCTION_GATEWAY_GROUP = "ai-platform-brain"
PRODUCTION_WRITER_EXPORT_PATH = pathlib.Path(
    "/var/lib/muncho-canonical-writer/projection/canonical-events.json"
)
PRODUCTION_PUBLIC_PROJECTION_PATH = pathlib.Path(
    "/var/lib/muncho-projector/public/team-member-aliases.json"
)
PRODUCTION_RUN_RECEIPT_PATH = pathlib.Path(
    "/var/lib/muncho-projector/public/team-member-aliases.receipt.json"
)
MAX_WRITER_EXPORT_BYTES = 256 * 1024 * 1024
MAX_WRITER_EXPORT_EVENTS = 1_000_000
_CASE_ID_RE = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEYS = frozenset(
    {
        "event_id",
        "schema_version",
        "event_type",
        "occurred_at",
        "case_id",
        "source",
        "actor",
        "subject",
        "evidence",
        "decision",
        "status",
        "next_action",
        "safety",
        "payload",
    }
)
_PERSON_PAYLOAD_KEYS = frozenset(
    {
        "alias",
        "member_key",
        "idempotency_key",
        "summary",
        "canonical_content_sha256",
    }
)
_CHANNEL_PAYLOAD_KEYS = frozenset(
    {
        "alias",
        "guild_id",
        "target_type",
        "channel_id",
        "idempotency_key",
        "summary",
        "canonical_content_sha256",
    }
)
_SOURCE_KEYS = frozenset(
    {"system", "component", "source_refs", "observed_session"}
)
_DECISION_KEYS = frozenset(
    {"kind", "decided_by", "keyword_authority", "attestation"}
)
_STATUS_KEYS = frozenset({"state", "event_type", "summary"})
_SAFETY_KEYS = frozenset(
    {"secret_value_recorded", "payment_credential_recorded", "business_mutation"}
)


class AliasProjectorError(ValueError):
    """Writer export or alias event failed the exact projector contract."""


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
            raise AliasProjectorError("writer_export_json_keys_invalid")
        result[key] = value
    return result


def _canonical_event_id(value: Any) -> str:
    if not isinstance(value, str):
        raise AliasProjectorError("writer_export_event_identity_invalid")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise AliasProjectorError("writer_export_event_identity_invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise AliasProjectorError("writer_export_event_identity_invalid")
    return value


def _utc_timestamp(value: Any) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise AliasProjectorError("writer_export_event_timestamp_invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AliasProjectorError("writer_export_event_timestamp_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise AliasProjectorError("writer_export_event_timestamp_invalid")
    return parsed


def _read_stable_writer_export(
    path: pathlib.Path,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> tuple[list[Any], str, os.stat_result]:
    path = pathlib.Path(path)
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
        or (expected_uid is not None and before.st_uid != expected_uid)
        or (expected_gid is not None and before.st_gid != expected_gid)
        or (
            expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode
        )
        or not 0 < before.st_size <= MAX_WRITER_EXPORT_BYTES
    ):
        raise AliasProjectorError("writer_export_file_untrusted")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise AliasProjectorError("writer_export_file_changed")
        chunks: list[bytes] = []
        remaining = MAX_WRITER_EXPORT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise AliasProjectorError("writer_export_file_changed") from exc
    if (
        len(raw) != before.st_size
        or _file_identity(after) != _file_identity(before)
        or _file_identity(path_after) != _file_identity(before)
    ):
        raise AliasProjectorError("writer_export_file_changed")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AliasProjectorError("writer_export_json_constant_invalid")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AliasProjectorError("writer_export_json_invalid") from exc
    try:
        rows, _provenance = validate_projection_export(
            value,
            maximum_events=MAX_WRITER_EXPORT_EVENTS,
        )
    except ProjectionExportError as exc:
        raise AliasProjectorError("writer_export_envelope_invalid") from exc
    return rows, hashlib.sha256(raw).hexdigest(), before


def _validate_alias_event(
    row: Mapping[str, Any],
) -> tuple[str, str, str | dict[str, str]]:
    if set(row) != _EVENT_KEYS:
        raise AliasProjectorError("alias_event_envelope_invalid")
    if row.get("schema_version") != "canonical_event.v1":
        raise AliasProjectorError("alias_event_schema_invalid")
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise AliasProjectorError("alias_event_case_invalid")

    source = row.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != _SOURCE_KEYS
        or source.get("system") != "hermes_agent"
        or source.get("component") != "canonical_writer"
        or not isinstance(source.get("source_refs"), Mapping)
        or not isinstance(source.get("observed_session"), Mapping)
        or not isinstance(row.get("actor"), Mapping)
        or not isinstance(row.get("subject"), Mapping)
        or row.get("evidence") != []
    ):
        raise AliasProjectorError("alias_event_provenance_invalid")

    decision = row.get("decision")
    if (
        not isinstance(decision, Mapping)
        or set(decision) != _DECISION_KEYS
        or decision.get("kind") != "typed_canonical_writer_operation"
        or decision.get("keyword_authority") is not False
        or decision.get("attestation") != "model_authored"
        or not isinstance(decision.get("decided_by"), str)
        or not decision.get("decided_by")
    ):
        raise AliasProjectorError("alias_event_decision_invalid")

    payload = row.get("payload")
    status = row.get("status")
    safety = row.get("safety")
    event_type = row.get("event_type")
    if not isinstance(payload, Mapping):
        raise AliasProjectorError("alias_event_payload_invalid")
    expected_payload_keys = _PERSON_PAYLOAD_KEYS
    if event_type == CHANNEL_ALIAS_EVENT_TYPE:
        expected_payload_keys = _CHANNEL_PAYLOAD_KEYS
        if payload.get("target_type") == "guild_thread":
            expected_payload_keys = expected_payload_keys | {"parent_channel_id"}
    if set(payload) != expected_payload_keys:
        raise AliasProjectorError("alias_event_payload_invalid")
    if (
        not isinstance(status, Mapping)
        or set(status) != _STATUS_KEYS
        or status.get("state") != event_type
        or status.get("event_type") != event_type
        or not isinstance(status.get("summary"), str)
        or not isinstance(payload.get("summary"), str)
        or status.get("summary") != payload.get("summary")[:500]
        or not payload.get("summary")
    ):
        raise AliasProjectorError("alias_event_status_invalid")
    if (
        row.get("next_action") != {}
        or not isinstance(safety, Mapping)
        or set(safety) != _SAFETY_KEYS
        or any(safety.get(key) is not False for key in _SAFETY_KEYS)
    ):
        raise AliasProjectorError("alias_event_safety_invalid")

    content_sha256 = payload.get("canonical_content_sha256")
    idempotency_key = payload.get("idempotency_key")
    alias = payload.get("alias")
    if (
        not isinstance(content_sha256, str)
        or not _SHA256_RE.fullmatch(content_sha256)
        or not isinstance(idempotency_key, str)
        or not 0 < len(idempotency_key.encode("utf-8")) <= 256
        or not isinstance(alias, str)
    ):
        raise AliasProjectorError("alias_event_payload_invalid")
    normalized = normalize_team_member_alias(alias)
    if (
        not normalized
        or len(normalized) > 200
        or len(normalized.encode("utf-8")) > 512
    ):
        raise AliasProjectorError("alias_event_alias_invalid")

    if event_type == ALIAS_EVENT_TYPE:
        member_key = payload.get("member_key")
        if (
            not isinstance(member_key, str)
            or member_key not in TEAM_MEMBERS_BY_KEY
            or normalized in STATIC_ALIAS_CHANNEL_IDS
        ):
            raise AliasProjectorError("alias_event_payload_invalid")
        static_matches = {
            member.key
            for static_alias, member in ALIAS_ENTRIES
            if static_alias == normalized
        }
        if static_matches and static_matches != {member_key}:
            raise AliasProjectorError("alias_event_conflicts_with_static_registry")
        return "person", normalized, member_key

    guild_id = payload.get("guild_id")
    target_type = payload.get("target_type")
    channel_id = payload.get("channel_id")
    parent_channel_id = payload.get("parent_channel_id")
    if (
        not isinstance(guild_id, str)
        or guild_id != SKYVISION_GUILD_ID
        or not isinstance(channel_id, str)
        or re.fullmatch(r"[1-9][0-9]{16,19}", channel_id) is None
        or target_type not in {"guild_channel", "guild_thread"}
        or (
            target_type == "guild_thread"
            and (
                not isinstance(parent_channel_id, str)
                or re.fullmatch(r"[1-9][0-9]{16,19}", parent_channel_id) is None
                or parent_channel_id == channel_id
            )
        )
        or normalized in STATIC_ALIAS_MEMBER_KEYS
    ):
        raise AliasProjectorError("channel_alias_event_target_invalid")
    if (
        target_type == "guild_channel"
        and channel_id not in APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID
    ) or (
        target_type == "guild_thread"
        and parent_channel_id
        not in APPROVED_OPERATIONAL_GUILD_LANES_BY_CHANNEL_ID
    ):
        raise AliasProjectorError("channel_alias_event_target_not_owner_approved")
    static_channel_ids = {
        lane.channel_id
        for static_alias, lane in APPROVED_LANE_ALIAS_ENTRIES
        if static_alias == normalized
    }
    if static_channel_ids and (
        static_channel_ids != {channel_id} or target_type != "guild_channel"
    ):
        raise AliasProjectorError("alias_event_conflicts_with_static_registry")
    return "channel", normalized, {
        "guild_id": guild_id,
        "target_type": target_type,
        "channel_id": channel_id,
        **(
            {"parent_channel_id": parent_channel_id}
            if target_type == "guild_thread"
            else {}
        ),
    }


def build_alias_projection_document(
    rows: list[Any],
    *,
    source_export_sha256: str,
) -> dict[str, Any]:
    """Fold exact typed alias events; never inspect any free-form text."""

    if not _SHA256_RE.fullmatch(str(source_export_sha256 or "")):
        raise AliasProjectorError("writer_export_digest_invalid")
    aliases: dict[str, str] = {}
    channel_aliases: dict[str, dict[str, str]] = {}
    seen_event_ids: set[str] = set()
    alias_event_ids: list[str] = []
    last_sort_key: tuple[dt.datetime, str] | None = None
    last_alias_event_id = ""
    last_alias_event_at = ""
    for row in rows:
        if not isinstance(row, Mapping):
            raise AliasProjectorError("writer_export_event_invalid")
        event_id = _canonical_event_id(row.get("event_id"))
        occurred_at = _utc_timestamp(row.get("occurred_at"))
        sort_key = (occurred_at, event_id)
        if event_id in seen_event_ids or (
            last_sort_key is not None and sort_key <= last_sort_key
        ):
            raise AliasProjectorError("writer_export_event_order_invalid")
        seen_event_ids.add(event_id)
        last_sort_key = sort_key

        event_type = row.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise AliasProjectorError("writer_export_event_type_invalid")
        if event_type not in ALIAS_EVENT_TYPES:
            continue
        alias_kind, alias, target = _validate_alias_event(row)
        if alias_kind == "person":
            if alias in channel_aliases:
                raise AliasProjectorError("alias_event_mapping_conflict")
            member_key = str(target)
            existing = aliases.get(alias)
            if existing is not None and existing != member_key:
                raise AliasProjectorError("alias_event_mapping_conflict")
            aliases[alias] = member_key
        else:
            if alias in aliases:
                raise AliasProjectorError("alias_event_mapping_conflict")
            assert isinstance(target, dict)
            existing_channel = channel_aliases.get(alias)
            if existing_channel is not None and existing_channel != target:
                raise AliasProjectorError("alias_event_mapping_conflict")
            channel_aliases[alias] = dict(target)
        alias_event_ids.append(event_id)
        last_alias_event_id = event_id
        last_alias_event_at = str(row["occurred_at"])

    aliases = dict(sorted(aliases.items()))
    channel_aliases = dict(sorted(channel_aliases.items()))
    document: dict[str, Any] = {
        "schema": ALIAS_PROJECTION_SCHEMA,
        "aliases": aliases,
        "channel_aliases": channel_aliases,
        "receipt": {
            "schema": ALIAS_PROJECTION_RECEIPT_SCHEMA,
            "source_export_sha256": source_export_sha256,
            "source_event_count": len(rows),
            "alias_event_count": len(alias_event_ids),
            "alias_count": len(aliases) + len(channel_aliases),
            "last_alias_event_id": last_alias_event_id,
            "last_alias_event_at": last_alias_event_at,
            "accepted_event_ids_sha256": accepted_event_ids_sha256(
                alias_event_ids
            ),
            "projection_sha256": projection_payload_sha256(
                aliases,
                channel_aliases,
            ),
        },
    }
    validate_alias_projection_document(
        document,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    )
    return document


def project_aliases_from_writer_export(
    events_path: pathlib.Path,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    rows, source_sha256, _metadata = _read_stable_writer_export(
        events_path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_mode=expected_mode,
    )
    return build_alias_projection_document(
        rows,
        source_export_sha256=source_sha256,
    )


def validate_projection_progress(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Reject a stale, rewritten, or regressing append-only projection."""

    validate_alias_projection_document(
        previous,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    )
    validate_alias_projection_document(
        candidate,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    )
    old_receipt = previous["receipt"]
    new_receipt = candidate["receipt"]
    old_source_count = old_receipt["source_event_count"]
    new_source_count = new_receipt["source_event_count"]
    old_alias_count = old_receipt["alias_event_count"]
    new_alias_count = new_receipt["alias_event_count"]
    if new_source_count < old_source_count or new_alias_count < old_alias_count:
        raise AliasProjectorError("alias_projection_progress_regressed")
    if (
        new_source_count == old_source_count
        and new_receipt["source_export_sha256"]
        != old_receipt["source_export_sha256"]
    ):
        raise AliasProjectorError("alias_projection_source_rewritten")

    old_aliases = previous["aliases"]
    new_aliases = candidate["aliases"]
    if any(new_aliases.get(alias) != member for alias, member in old_aliases.items()):
        raise AliasProjectorError("alias_projection_mapping_regressed")
    old_channels = previous["channel_aliases"]
    new_channels = candidate["channel_aliases"]
    if any(
        new_channels.get(alias) != target
        for alias, target in old_channels.items()
    ):
        raise AliasProjectorError("alias_projection_mapping_regressed")

    cursor_fields = (
        "last_alias_event_id",
        "last_alias_event_at",
        "accepted_event_ids_sha256",
    )
    if new_alias_count == old_alias_count and any(
        new_receipt[field] != old_receipt[field] for field in cursor_fields
    ):
        raise AliasProjectorError("alias_projection_cursor_rewritten")
    if old_alias_count:
        old_cursor = (
            _utc_timestamp(old_receipt["last_alias_event_at"]),
            old_receipt["last_alias_event_id"],
        )
        new_cursor = (
            _utc_timestamp(new_receipt["last_alias_event_at"]),
            new_receipt["last_alias_event_id"],
        )
        if new_cursor < old_cursor:
            raise AliasProjectorError("alias_projection_cursor_regressed")


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_regular_bytes(
    path: pathlib.Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    maximum: int = 2 * 1024 * 1024,
) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or before.st_gid != expected_gid
        or stat.S_IMODE(before.st_mode) != expected_mode
        or not 0 < before.st_size <= maximum
    ):
        raise AliasProjectorError("alias_projection_readback_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise AliasProjectorError("alias_projection_readback_changed")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise AliasProjectorError("alias_projection_readback_changed") from exc
    if (
        len(raw) != before.st_size
        or _file_identity(after) != _file_identity(before)
        or _file_identity(path_after) != _file_identity(before)
    ):
        raise AliasProjectorError("alias_projection_readback_changed")
    return raw, before


def _write_json_file(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Use the projection writer's pinned no-follow atomic replacement path."""

    write_alias_projection(path, value, validate_document=False)


def validate_run_receipt(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "source_export_sha256",
        "projection_sha256",
        "projection_file_sha256",
        "source_event_count",
        "alias_event_count",
        "alias_count",
        "last_alias_event_id",
        "last_alias_event_at",
        "previous_projection_sha256",
        "projection_path",
        "source_export_path",
        "source_export_uid",
        "source_export_gid",
        "source_export_mode",
        "projection_uid",
        "projection_gid",
        "projection_mode",
        "replaced_existing",
        "secret_material_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AliasProjectorError("alias_projection_run_receipt_invalid")
    receipt = dict(value)
    digest_fields = (
        "source_export_sha256",
        "projection_sha256",
        "projection_file_sha256",
        "receipt_sha256",
    )
    previous = receipt["previous_projection_sha256"]
    source_path = receipt["source_export_path"]
    projection_path = receipt["projection_path"]
    alias_event_count = receipt["alias_event_count"]
    last_id = receipt["last_alias_event_id"]
    last_at = receipt["last_alias_event_at"]
    cursor_valid = True
    if (
        type(alias_event_count) is int
        and alias_event_count > 0
        and isinstance(last_id, str)
        and isinstance(last_at, str)
    ):
        try:
            _canonical_event_id(last_id)
            _utc_timestamp(last_at)
        except AliasProjectorError:
            cursor_valid = False
    if (
        receipt["schema"] != ALIAS_PROJECTION_RUN_RECEIPT_SCHEMA
        or any(_SHA256_RE.fullmatch(str(receipt[field])) is None for field in digest_fields)
        or (previous is not None and _SHA256_RE.fullmatch(str(previous)) is None)
        or any(
            isinstance(receipt[field], bool)
            or not isinstance(receipt[field], int)
            or receipt[field] < 0
            for field in ("source_event_count", "alias_event_count", "alias_count")
        )
        or receipt["source_event_count"] < receipt["alias_event_count"]
        or receipt["alias_event_count"] < receipt["alias_count"]
        or not isinstance(last_id, str)
        or not isinstance(last_at, str)
        or (
            alias_event_count == 0
            and (last_id != "" or last_at != "")
        )
        or (
            alias_event_count > 0
            and not cursor_valid
        )
        or any(
            isinstance(receipt[field], bool) or not isinstance(receipt[field], int)
            for field in (
                "source_export_uid",
                "source_export_gid",
                "source_export_mode",
                "projection_uid",
                "projection_gid",
                "projection_mode",
            )
        )
        or receipt["source_export_mode"] != 0o640
        or receipt["projection_mode"] != 0o640
        or type(receipt["replaced_existing"]) is not bool
        or receipt["secret_material_recorded"] is not False
        or not isinstance(projection_path, str)
        or not isinstance(source_path, str)
        or not pathlib.Path(projection_path).is_absolute()
        or not pathlib.Path(source_path).is_absolute()
        or ".." in pathlib.Path(projection_path).parts
        or ".." in pathlib.Path(source_path).parts
        or projection_path == source_path
        or hashlib.sha256(
            json.dumps(
                {key: item for key, item in receipt.items() if key != "receipt_sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        != receipt["receipt_sha256"]
    ):
        raise AliasProjectorError("alias_projection_run_receipt_invalid")
    return receipt


def publish_alias_projection(
    events_path: pathlib.Path,
    output_path: pathlib.Path,
    receipt_path: pathlib.Path,
    *,
    writer_uid: int,
    projector_uid: int,
    projector_gid: int,
    gateway_gid: int,
    public_directory_mode: int = 0o2750,
) -> dict[str, Any]:
    """Publish one monotonic projection and a deterministic readback receipt."""

    target = pathlib.Path(output_path)
    receipt_target = pathlib.Path(receipt_path)
    export = pathlib.Path(events_path)
    parent = target.parent.lstat()
    if (
        target.parent != receipt_target.parent
        or stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != projector_uid
        or parent.st_gid != gateway_gid
        or stat.S_IMODE(parent.st_mode) != public_directory_mode
    ):
        raise AliasProjectorError("alias_projection_output_directory_untrusted")
    rows, source_sha256, export_metadata = _read_stable_writer_export(
        export,
        expected_uid=writer_uid,
        expected_gid=projector_gid,
        expected_mode=0o640,
    )
    candidate = build_alias_projection_document(
        rows,
        source_export_sha256=source_sha256,
    )
    previous: Mapping[str, Any] | None = None
    if target.exists() or target.is_symlink():
        existing = target.lstat()
        if (
            stat.S_ISLNK(existing.st_mode)
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
            or existing.st_uid != projector_uid
            or existing.st_gid != gateway_gid
            or stat.S_IMODE(existing.st_mode) != 0o640
        ):
            raise AliasProjectorError("alias_projection_output_file_untrusted")
        previous = load_alias_projection_document(
            target,
            normalize_alias=normalize_team_member_alias,
            valid_member_keys=TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=SKYVISION_GUILD_ID,
            static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
        )
        validate_projection_progress(previous, candidate)
    write_alias_projection(target, candidate)
    readback = load_alias_projection_document(
        target,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    )
    if readback != candidate:
        raise AliasProjectorError("alias_projection_readback_mismatch")
    raw, output_metadata = _stable_regular_bytes(
        target,
        expected_uid=projector_uid,
        expected_gid=gateway_gid,
        expected_mode=0o640,
    )
    if raw != _canonical_json_line(candidate):
        raise AliasProjectorError("alias_projection_readback_mismatch")
    previous_digest = (
        None
        if previous is None
        else str(previous["receipt"]["projection_sha256"])
    )
    unsigned = {
        "schema": ALIAS_PROJECTION_RUN_RECEIPT_SCHEMA,
        "source_export_sha256": source_sha256,
        "projection_sha256": candidate["receipt"]["projection_sha256"],
        "projection_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_event_count": candidate["receipt"]["source_event_count"],
        "alias_event_count": candidate["receipt"]["alias_event_count"],
        "alias_count": candidate["receipt"]["alias_count"],
        "last_alias_event_id": candidate["receipt"]["last_alias_event_id"],
        "last_alias_event_at": candidate["receipt"]["last_alias_event_at"],
        "previous_projection_sha256": previous_digest,
        "projection_path": str(target),
        "source_export_path": str(export),
        "source_export_uid": export_metadata.st_uid,
        "source_export_gid": export_metadata.st_gid,
        "source_export_mode": stat.S_IMODE(export_metadata.st_mode),
        "projection_uid": output_metadata.st_uid,
        "projection_gid": output_metadata.st_gid,
        "projection_mode": stat.S_IMODE(output_metadata.st_mode),
        "replaced_existing": previous is not None,
        "secret_material_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    validate_run_receipt(receipt)
    if receipt_target.exists() or receipt_target.is_symlink():
        existing_receipt = receipt_target.lstat()
        if (
            stat.S_ISLNK(existing_receipt.st_mode)
            or not stat.S_ISREG(existing_receipt.st_mode)
            or existing_receipt.st_nlink != 1
            or existing_receipt.st_uid != projector_uid
            or existing_receipt.st_gid != gateway_gid
            or stat.S_IMODE(existing_receipt.st_mode) != 0o640
        ):
            raise AliasProjectorError("alias_projection_run_receipt_untrusted")
    _write_json_file(receipt_target, receipt)
    receipt_raw, _receipt_metadata = _stable_regular_bytes(
        receipt_target,
        expected_uid=projector_uid,
        expected_gid=gateway_gid,
        expected_mode=0o640,
    )
    try:
        receipt_readback = json.loads(receipt_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AliasProjectorError("alias_projection_run_receipt_invalid") from exc
    if receipt_raw != _canonical_json_line(receipt) or validate_run_receipt(
        receipt_readback
    ) != receipt:
        raise AliasProjectorError("alias_projection_run_receipt_readback_mismatch")
    return receipt


def write_alias_projection(
    path: pathlib.Path,
    document: Mapping[str, Any],
    *,
    validate_document: bool = True,
) -> None:
    """Atomically publish one digest-bound safe projection as mode 0640."""

    target = pathlib.Path(path)
    if not target.is_absolute() or ".." in target.parts or target.suffix != ".json":
        raise AliasProjectorError("alias_projection_output_path_invalid")
    if validate_document:
        validate_alias_projection_document(
            document,
            normalize_alias=normalize_team_member_alias,
            valid_member_keys=TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=SKYVISION_GUILD_ID,
            static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
        )
    parent_before = target.parent.lstat()
    if (
        stat.S_ISLNK(parent_before.st_mode)
        or not stat.S_ISDIR(parent_before.st_mode)
        or parent_before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX projector service boundary
        or stat.S_IMODE(parent_before.st_mode) & 0o002
    ):
        raise AliasProjectorError("alias_projection_output_directory_untrusted")
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    descriptor = -1
    try:
        if _file_identity(os.fstat(directory_fd)) != _file_identity(parent_before):
            raise AliasProjectorError("alias_projection_output_directory_changed")
        if target.exists() or target.is_symlink():
            current = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_uid != os.geteuid()  # windows-footgun: ok — POSIX projector service boundary
                or stat.S_IMODE(current.st_mode) != 0o640
            ):
                raise AliasProjectorError("alias_projection_output_file_untrusted")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o640,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o640)
        payload = (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AliasProjectorError("alias_projection_output_write_failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--receipt-json", type=pathlib.Path)
    parser.add_argument("--production-recurring", action="store_true")
    args = parser.parse_args()
    if args.production_recurring:
        if (
            args.events_json != PRODUCTION_WRITER_EXPORT_PATH
            or args.output_json != PRODUCTION_PUBLIC_PROJECTION_PATH
            or args.receipt_json != PRODUCTION_RUN_RECEIPT_PATH
        ):
            raise AliasProjectorError("alias_projection_production_path_invalid")
        try:
            writer = pwd.getpwnam(PRODUCTION_WRITER_USER)
            projector = pwd.getpwnam(PRODUCTION_PROJECTOR_USER)
            projector_group = grp.getgrnam(PRODUCTION_PROJECTOR_GROUP)
            gateway_group = grp.getgrnam(PRODUCTION_GATEWAY_GROUP)
        except KeyError as exc:
            raise AliasProjectorError("alias_projection_identity_unavailable") from exc
        if (
            writer.pw_uid == projector.pw_uid
            or writer.pw_gid == projector.pw_gid
            or projector.pw_gid != projector_group.gr_gid
            or os.geteuid() != projector.pw_uid  # windows-footgun: ok — POSIX projector service boundary
            or os.getegid() != projector.pw_gid  # windows-footgun: ok — POSIX projector service boundary
            or tuple(sorted(set(os.getgroups()) | {os.getegid()}))  # windows-footgun: ok — POSIX projector service boundary
            != (projector.pw_gid,)
        ):
            raise AliasProjectorError("alias_projection_identity_invalid")
        receipt = publish_alias_projection(
            args.events_json,
            args.output_json,
            args.receipt_json,
            writer_uid=writer.pw_uid,
            projector_uid=projector.pw_uid,
            projector_gid=projector_group.gr_gid,
            gateway_gid=gateway_group.gr_gid,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.receipt_json is not None:
        raise AliasProjectorError("alias_projection_receipt_requires_production_mode")
    document = project_aliases_from_writer_export(args.events_json)
    write_alias_projection(args.output_json, document)
    print(json.dumps(document["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AliasProjectorError",
    "ALIAS_PROJECTION_RUN_RECEIPT_SCHEMA",
    "build_alias_projection_document",
    "publish_alias_projection",
    "project_aliases_from_writer_export",
    "validate_projection_progress",
    "validate_run_receipt",
    "write_alias_projection",
]
