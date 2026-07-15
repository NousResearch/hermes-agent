from __future__ import annotations

import hashlib
import json
import stat
import uuid

import pytest

from gateway.support_ops_alias_projection import (
    AliasProjectionError,
    load_alias_projection,
    load_channel_alias_projection,
)
from gateway.support_ops_team_registry import (
    SKYVISION_GUILD_ID,
    STATIC_ALIAS_CHANNEL_IDS,
    TEAM_MEMBERS_BY_KEY,
    STATIC_ALIAS_MEMBER_KEYS,
    normalize_team_member_alias,
)
from scripts.canonical_brain_alias_projector import (
    AliasProjectorError,
    build_alias_projection_document,
    project_aliases_from_writer_export,
    write_alias_projection,
)


def _event_id(number: int) -> str:
    return str(uuid.UUID(int=number, version=5))


def _alias_event(
    number: int,
    *,
    alias: str = "Niki",
    member_key: str = "alex",
    occurred_at: str | None = None,
) -> dict:
    summary = "Requester explicitly clarified the alias"
    return {
        "event_id": _event_id(number),
        "schema_version": "canonical_event.v1",
        "event_type": "person.alias.learned",
        "occurred_at": occurred_at or f"2026-07-14T10:00:{number:02d}+00:00",
        "case_id": "case:alias-learning",
        "source": {
            "system": "hermes_agent",
            "component": "canonical_writer",
            "source_refs": {"platform": "discord", "message_id": f"m-{number}"},
            "observed_session": {"request_id": f"r-{number}"},
        },
        "actor": {"type": "service", "id": "canonical_writer"},
        "subject": {"type": "case", "id": "case:alias-learning"},
        "evidence": [],
        "decision": {
            "kind": "typed_canonical_writer_operation",
            "decided_by": "model_event_append",
            "keyword_authority": False,
            "attestation": "model_authored",
        },
        "status": {
            "state": "person.alias.learned",
            "event_type": "person.alias.learned",
            "summary": summary,
        },
        "next_action": {},
        "safety": {
            "secret_value_recorded": False,
            "payment_credential_recorded": False,
            "business_mutation": False,
        },
        "payload": {
            "alias": alias,
            "member_key": member_key,
            "idempotency_key": f"alias:{number}",
            "summary": summary,
            "canonical_content_sha256": hashlib.sha256(
                f"alias:{number}".encode()
            ).hexdigest(),
        },
    }


def _channel_alias_event(
    number: int,
    *,
    alias: str = "Incidents",
    channel_id: str = "1527000000000000001",
    target_type: str = "guild_channel",
    parent_channel_id: str | None = None,
    guild_id: str = SKYVISION_GUILD_ID,
) -> dict:
    event = _alias_event(number, alias=alias)
    event["event_type"] = "channel.alias.learned"
    event["status"]["state"] = "channel.alias.learned"
    event["status"]["event_type"] = "channel.alias.learned"
    event["payload"].pop("member_key")
    event["payload"].update(
        {
            "guild_id": guild_id,
            "target_type": target_type,
            "channel_id": channel_id,
        }
    )
    if parent_channel_id is not None:
        event["payload"]["parent_channel_id"] = parent_channel_id
    return event


def _write_export(path, rows) -> None:
    path.write_text(
        json.dumps({"events": rows}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o640)


def test_projector_emits_only_safe_aliases_and_integrity_receipt(tmp_path):
    export = tmp_path / "canonical-events.json"
    unrelated = {
        "event_id": _event_id(1),
        "event_type": "case.note",
        "occurred_at": "2026-07-14T10:00:01+00:00",
        "payload": {"private_business_text": "must-not-escape"},
        "source": {"raw": "also-must-not-escape"},
    }
    _write_export(export, [unrelated, _alias_event(2, alias="  Ники  ")])

    document = project_aliases_from_writer_export(export)

    assert document["aliases"] == {"ники": "alex"}
    assert document["channel_aliases"] == {}
    assert document["receipt"]["source_event_count"] == 2
    assert document["receipt"]["alias_event_count"] == 1
    assert document["receipt"]["alias_count"] == 1
    rendered = json.dumps(document, ensure_ascii=False)
    assert "must-not-escape" not in rendered
    assert "Requester explicitly clarified" not in rendered
    assert "message_id" not in rendered


def test_projector_rejects_extra_alias_fields_and_conflicting_mappings():
    event = _alias_event(1)
    event["payload"]["extra"] = "not in the exact schema"
    with pytest.raises(AliasProjectorError, match="alias_event_payload_invalid"):
        build_alias_projection_document(
            [event], source_export_sha256="a" * 64
        )

    with pytest.raises(AliasProjectorError, match="alias_event_mapping_conflict"):
        build_alias_projection_document(
            [
                _alias_event(1, alias="Niki", member_key="alex"),
                _alias_event(2, alias="niki", member_key="ivcho"),
            ],
            source_export_sha256="b" * 64,
        )


def test_projector_rejects_static_registry_conflict_and_noncanonical_order():
    with pytest.raises(
        AliasProjectorError, match="alias_event_conflicts_with_static_registry"
    ):
        build_alias_projection_document(
            [_alias_event(1, alias="Alex", member_key="ivcho")],
            source_export_sha256="c" * 64,
        )


def test_channel_alias_root_and_thread_round_trip_and_replay():
    root = _channel_alias_event(1, alias="Incidents")
    thread = _channel_alias_event(
        2,
        alias="July incidents",
        channel_id="1527000000000000002",
        target_type="guild_thread",
        parent_channel_id="1527000000000000001",
    )
    replay = _channel_alias_event(
        3,
        alias=" incidents ",
    )

    document = build_alias_projection_document(
        [root, thread, replay], source_export_sha256="7" * 64
    )

    assert document["channel_aliases"] == {
        "incidents": {
            "guild_id": SKYVISION_GUILD_ID,
            "target_type": "guild_channel",
            "channel_id": "1527000000000000001",
        },
        "july incidents": {
            "guild_id": SKYVISION_GUILD_ID,
            "target_type": "guild_thread",
            "channel_id": "1527000000000000002",
            "parent_channel_id": "1527000000000000001",
        },
    }
    assert document["receipt"]["alias_event_count"] == 3
    assert document["receipt"]["alias_count"] == 2


@pytest.mark.parametrize(
    "event,error",
    [
        (
            _channel_alias_event(1, guild_id="1282725267068157973"),
            "channel_alias_event_target_invalid",
        ),
        (
            _channel_alias_event(
                1,
                target_type="guild_thread",
                channel_id="1527000000000000002",
            ),
            "alias_event_payload_invalid",
        ),
        (
            _channel_alias_event(
                1,
                target_type="guild_channel",
                parent_channel_id="1527000000000000002",
            ),
            "alias_event_payload_invalid",
        ),
    ],
)
def test_channel_alias_rejects_wrong_guild_and_invalid_parent_shape(event, error):
    with pytest.raises(AliasProjectorError, match=error):
        build_alias_projection_document([event], source_export_sha256="8" * 64)


def test_channel_alias_rejects_conflict_rebinding_and_cross_kind_alias():
    with pytest.raises(AliasProjectorError, match="alias_event_mapping_conflict"):
        build_alias_projection_document(
            [
                _channel_alias_event(1),
                _channel_alias_event(2, channel_id="1527000000000000009"),
            ],
            source_export_sha256="9" * 64,
        )
    with pytest.raises(AliasProjectorError, match="alias_event_mapping_conflict"):
        build_alias_projection_document(
            [_alias_event(1, alias="Incidents"), _channel_alias_event(2)],
            source_export_sha256="0" * 64,
        )

    with pytest.raises(AliasProjectorError, match="writer_export_event_order_invalid"):
        build_alias_projection_document(
            [_alias_event(2), _alias_event(1)],
            source_export_sha256="d" * 64,
        )


def test_writer_export_reader_rejects_duplicate_json_keys(tmp_path):
    export = tmp_path / "canonical-events.json"
    export.write_text('{"events":[],"events":[]}', encoding="utf-8")
    export.chmod(0o640)
    with pytest.raises(AliasProjectorError, match="writer_export_json_keys_invalid"):
        project_aliases_from_writer_export(export)


def test_atomic_projection_round_trip_and_tamper_rejection(tmp_path):
    document = build_alias_projection_document(
        [_alias_event(1)], source_export_sha256="e" * 64
    )
    output = tmp_path / "team-member-aliases.json"

    write_alias_projection(output, document)

    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert load_alias_projection(
        output,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
    ) == {"niki": "alex"}
    assert load_channel_alias_projection(
        output,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    ) == {}

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["aliases"]["niki"] = "ivcho"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    output.chmod(0o640)
    with pytest.raises(AliasProjectionError, match="payload_digest_mismatch"):
        load_alias_projection(
            output,
            normalize_alias=normalize_team_member_alias,
            valid_member_keys=TEAM_MEMBERS_BY_KEY,
            static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        )


def test_empty_projection_is_valid_and_input_symlinks_fail_closed(tmp_path):
    document = build_alias_projection_document(
        [], source_export_sha256="1" * 64
    )
    output = tmp_path / "team-member-aliases.json"
    write_alias_projection(output, document)
    assert load_alias_projection(
        output,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
    ) == {}

    export = tmp_path / "canonical-events.json"
    real_export = tmp_path / "real-export.json"
    _write_export(real_export, [])
    export.symlink_to(real_export)
    with pytest.raises(AliasProjectorError, match="writer_export_file_untrusted"):
        project_aliases_from_writer_export(export)
