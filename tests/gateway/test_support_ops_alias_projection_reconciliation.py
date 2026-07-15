from __future__ import annotations

import json

import pytest

import gateway.support_ops_team_registry as registry
from gateway.support_ops_alias_projection import (
    ALIAS_PROJECTION_RECEIPT_SCHEMA,
    ALIAS_PROJECTION_SCHEMA,
    AliasProjectionError,
    accepted_event_ids_sha256,
    load_alias_projection,
    load_channel_alias_projection,
    projection_payload_sha256,
)


_EVENT_ID = "00000000-0000-5000-8000-000000000001"


def _projection_document() -> dict:
    aliases = {"niki": "alex"}
    return {
        "schema": ALIAS_PROJECTION_SCHEMA,
        "aliases": aliases,
        "channel_aliases": {},
        "receipt": {
            "schema": ALIAS_PROJECTION_RECEIPT_SCHEMA,
            "source_export_sha256": "f" * 64,
            "source_event_count": 1,
            "alias_event_count": 1,
            "alias_count": 1,
            "last_alias_event_id": _EVENT_ID,
            "last_alias_event_at": "2026-07-14T10:00:01+00:00",
            "accepted_event_ids_sha256": accepted_event_ids_sha256([_EVENT_ID]),
            "projection_sha256": projection_payload_sha256(aliases),
        },
    }


def _write_projection(path) -> None:
    path.write_text(
        json.dumps(_projection_document(), ensure_ascii=False),
        encoding="utf-8",
    )
    path.chmod(0o640)


def test_resolver_recovers_alias_from_canonical_projection_after_restart(
    tmp_path, monkeypatch
):
    projection = tmp_path / "team-member-aliases.json"
    local_cache = tmp_path / "fresh-home" / "state" / "team-member-aliases.json"
    _write_projection(projection)
    monkeypatch.setattr(registry, "_canonical_alias_projection_path", lambda: projection)
    monkeypatch.setattr(registry, "_learned_alias_path", lambda: local_cache)

    resolution = registry.resolve_team_member("NIKI")

    assert local_cache.exists() is False
    assert resolution.status == "resolved"
    assert resolution.member is not None
    assert resolution.member.key == "alex"


def test_present_tampered_projection_fails_closed_without_legacy_fallback(
    tmp_path, monkeypatch
):
    projection = tmp_path / "team-member-aliases.json"
    projection.write_text(
        json.dumps(
            {
                "schema": "canonical_brain.projection.team_member_aliases.v1",
                "aliases": {"niki": "alex"},
                "receipt": {},
            }
        ),
        encoding="utf-8",
    )
    projection.chmod(0o640)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "schema": "hermes.team_member_aliases.v1",
                "aliases": {"niki": "alex"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "_canonical_alias_projection_path", lambda: projection)
    monkeypatch.setattr(registry, "_learned_alias_path", lambda: legacy)

    assert registry.resolve_team_member("Niki").status == "unknown"


def test_missing_projection_preserves_verified_local_cache_transition(
    tmp_path, monkeypatch
):
    missing_projection = tmp_path / "missing-projection.json"
    local_cache = tmp_path / "state" / "team-member-aliases.json"
    monkeypatch.setattr(
        registry, "_canonical_alias_projection_path", lambda: missing_projection
    )
    monkeypatch.setattr(registry, "_learned_alias_path", lambda: local_cache)

    monkeypatch.setattr(
        "gateway.canonical_writer_boundary.writer_boundary_policy_required",
        lambda: False,
    )
    registry.learn_team_member_alias("Niki", "alex")

    resolution = registry.resolve_team_member("niki")
    assert resolution.status == "resolved"
    assert resolution.member is not None
    assert resolution.member.key == "alex"


def test_missing_projection_ignores_mutable_local_cache_in_production(
    tmp_path, monkeypatch
):
    missing_projection = tmp_path / "missing-projection.json"
    local_cache = tmp_path / "state" / "team-member-aliases.json"
    local_cache.parent.mkdir(parents=True)
    local_cache.write_text(
        json.dumps(
            {
                "schema": "hermes.team_member_aliases.v1",
                "aliases": {"niki": "alex"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        registry, "_canonical_alias_projection_path", lambda: missing_projection
    )
    monkeypatch.setattr(registry, "_learned_alias_path", lambda: local_cache)
    monkeypatch.setattr(
        "gateway.canonical_writer_boundary.writer_boundary_policy_required",
        lambda: True,
    )

    assert registry.resolve_team_member("Niki").status == "unknown"


def test_structured_canonical_thread_alias_round_trip(tmp_path, monkeypatch):
    projection = tmp_path / "team-member-aliases.json"
    document = _projection_document()
    document["channel_aliases"] = {
        "july incidents": {
            "guild_id": registry.SKYVISION_GUILD_ID,
            "target_type": "guild_thread",
            "channel_id": "1527000000000000002",
            "parent_channel_id": "1527000000000000001",
        }
    }
    document["receipt"]["alias_count"] = 2
    document["receipt"]["alias_event_count"] = 2
    document["receipt"]["source_event_count"] = 2
    document["receipt"]["projection_sha256"] = projection_payload_sha256(
        document["aliases"], document["channel_aliases"]
    )
    projection.write_text(json.dumps(document), encoding="utf-8")
    projection.chmod(0o640)
    monkeypatch.setattr(registry, "_canonical_alias_projection_path", lambda: projection)

    resolution = registry.resolve_approved_guild_lane("July Incidents")

    assert resolution.status == "resolved"
    assert resolution.lane is not None
    assert resolution.lane.target_type == "guild_thread"
    assert resolution.lane.channel_id == "1527000000000000002"
    assert resolution.lane.parent_channel_id == "1527000000000000001"
    assert load_channel_alias_projection(
        projection,
        normalize_alias=registry.normalize_team_member_alias,
        valid_member_keys=registry.TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=registry.STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=registry.SKYVISION_GUILD_ID,
        static_channel_alias_ids=registry.STATIC_ALIAS_CHANNEL_IDS,
    )["july incidents"]["target_type"] == "guild_thread"


def test_safe_reader_rejects_channel_alias_from_wrong_guild(tmp_path):
    projection = tmp_path / "team-member-aliases.json"
    document = _projection_document()
    document["channel_aliases"] = {
        "incidents": {
            "guild_id": "1282725267068157973",
            "target_type": "guild_channel",
            "channel_id": "1527000000000000001",
        }
    }
    document["receipt"]["alias_count"] = 2
    document["receipt"]["alias_event_count"] = 2
    document["receipt"]["source_event_count"] = 2
    document["receipt"]["projection_sha256"] = projection_payload_sha256(
        document["aliases"], document["channel_aliases"]
    )
    projection.write_text(json.dumps(document), encoding="utf-8")
    projection.chmod(0o640)

    with pytest.raises(AliasProjectionError, match="channel_entry_invalid"):
        load_channel_alias_projection(
            projection,
            normalize_alias=registry.normalize_team_member_alias,
            valid_member_keys=registry.TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=registry.STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=registry.SKYVISION_GUILD_ID,
            static_channel_alias_ids=registry.STATIC_ALIAS_CHANNEL_IDS,
        )


def test_gateway_registry_has_no_global_writer_export_or_database_path():
    source = __import__("inspect").getsource(registry)
    assert "canonical-events.json" not in source
    assert "projection_read_events" not in source
    assert "database" not in source.casefold()


def test_safe_reader_rejects_static_alias_rebinding_even_with_matching_digest(
    tmp_path,
):
    projection = tmp_path / "team-member-aliases.json"
    document = _projection_document()
    document["aliases"] = {"alex": "ivcho"}
    document["receipt"]["projection_sha256"] = projection_payload_sha256(
        document["aliases"]
    )
    projection.write_text(json.dumps(document), encoding="utf-8")
    projection.chmod(0o640)

    with pytest.raises(AliasProjectionError, match="entry_invalid"):
        load_alias_projection(
            projection,
            normalize_alias=registry.normalize_team_member_alias,
            valid_member_keys=registry.TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=registry.STATIC_ALIAS_MEMBER_KEYS,
        )


def test_safe_reader_rejects_person_alias_over_static_channel_name(tmp_path):
    projection = tmp_path / "team-member-aliases.json"
    document = _projection_document()
    document["aliases"] = {"backend": "alex"}
    document["receipt"]["projection_sha256"] = projection_payload_sha256(
        document["aliases"], document["channel_aliases"]
    )
    projection.write_text(json.dumps(document), encoding="utf-8")
    projection.chmod(0o640)

    with pytest.raises(AliasProjectionError, match="entry_invalid"):
        load_alias_projection(
            projection,
            normalize_alias=registry.normalize_team_member_alias,
            valid_member_keys=registry.TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=registry.STATIC_ALIAS_MEMBER_KEYS,
            expected_channel_guild_id=registry.SKYVISION_GUILD_ID,
            static_channel_alias_ids=registry.STATIC_ALIAS_CHANNEL_IDS,
        )
