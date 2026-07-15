from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid

import pytest

from gateway.support_ops_alias_projection import load_alias_projection_document
from gateway.support_ops_team_registry import (
    STATIC_ALIAS_MEMBER_KEYS,
    TEAM_MEMBERS_BY_KEY,
    normalize_team_member_alias,
)
from scripts import canonical_brain_alias_projector as projector


def _event(number: int, alias: str) -> dict:
    summary = "Requester explicitly clarified the alias"
    return {
        "event_id": str(uuid.UUID(int=number, version=5)),
        "schema_version": "canonical_event.v1",
        "event_type": "person.alias.learned",
        "occurred_at": f"2026-07-14T10:00:{number:02d}+00:00",
        "case_id": "case:alias-learning",
        "source": {
            "system": "hermes_agent",
            "component": "canonical_writer",
            "source_refs": {},
            "observed_session": {},
        },
        "actor": {},
        "subject": {},
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
            "member_key": "alex",
            "idempotency_key": f"alias:{number}",
            "summary": summary,
            "canonical_content_sha256": hashlib.sha256(
                f"alias:{number}".encode()
            ).hexdigest(),
        },
    }


def _write_export(path, rows) -> None:
    path.write_text(
        json.dumps(
            {"events": rows},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o640)


def _layout(tmp_path):
    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir(mode=0o750)
    public.mkdir(mode=0o2750)
    public.chmod(0o2750)
    export = private / "canonical-events.json"
    output = public / "team-member-aliases.json"
    receipt = public / "team-member-aliases.receipt.json"
    return export, output, receipt


def _publish(export, output, receipt):
    export_stat = export.lstat()
    output_directory_stat = output.parent.lstat()
    return projector.publish_alias_projection(
        export,
        output,
        receipt,
        writer_uid=export_stat.st_uid,
        projector_uid=output_directory_stat.st_uid,
        projector_gid=export_stat.st_gid,
        gateway_gid=output_directory_stat.st_gid,
        public_directory_mode=stat.S_IMODE(output_directory_stat.st_mode),
    )


def _read_projection(output):
    return load_alias_projection_document(
        output,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
    )


def test_recurring_projection_is_monotonic_and_restart_safe(tmp_path):
    export, output, receipt = _layout(tmp_path)
    _write_export(export, [_event(1, "Niki")])
    first = _publish(export, output, receipt)
    first_projection = _read_projection(output)

    second = _publish(export, output, receipt)

    assert _read_projection(output) == first_projection
    assert first["projection_sha256"] == second["projection_sha256"]
    assert second["previous_projection_sha256"] == first["projection_sha256"]
    assert second["replaced_existing"] is True
    assert projector.validate_run_receipt(
        json.loads(receipt.read_text(encoding="utf-8"))
    ) == second


def test_stale_export_cannot_regress_published_alias_state(tmp_path):
    export, output, receipt = _layout(tmp_path)
    _write_export(export, [_event(1, "Niki"), _event(2, "Nick")])
    _publish(export, output, receipt)
    before = output.read_bytes()

    _write_export(export, [_event(1, "Niki")])
    with pytest.raises(
        projector.AliasProjectorError,
        match="progress_regressed",
    ):
        _publish(export, output, receipt)

    assert output.read_bytes() == before


def test_wrong_export_identity_or_mode_fails_closed(tmp_path):
    export, output, receipt = _layout(tmp_path)
    _write_export(export, [_event(1, "Niki")])
    export.chmod(0o660)
    with pytest.raises(
        projector.AliasProjectorError,
        match="writer_export_file_untrusted",
    ):
        _publish(export, output, receipt)
    assert not output.exists()


def test_export_path_toctou_is_detected(tmp_path, monkeypatch):
    export, _output, _receipt = _layout(tmp_path)
    replacement = export.with_name("replacement.json")
    _write_export(export, [_event(1, "Niki")])
    _write_export(replacement, [_event(2, "Nick")])
    real_read = projector.os.read
    swapped = False

    def racing_read(descriptor, size):
        nonlocal swapped
        value = real_read(descriptor, size)
        if value and not swapped:
            swapped = True
            os.replace(replacement, export)
        return value

    monkeypatch.setattr(projector.os, "read", racing_read)
    with pytest.raises(
        projector.AliasProjectorError,
        match="writer_export_file_changed",
    ):
        projector.project_aliases_from_writer_export(export)


def test_partial_projection_write_preserves_previous_document(tmp_path, monkeypatch):
    export, output, receipt = _layout(tmp_path)
    _write_export(export, [_event(1, "Niki")])
    _publish(export, output, receipt)
    before = output.read_bytes()
    _write_export(export, [_event(1, "Niki"), _event(2, "Nick")])
    real_write = projector.os.write
    failed = False

    def partial_write(descriptor, value):
        nonlocal failed
        if not failed and len(value) > 8:
            failed = True
            return 0
        return real_write(descriptor, value)

    monkeypatch.setattr(projector.os, "write", partial_write)
    with pytest.raises(
        projector.AliasProjectorError,
        match="output_write_failed",
    ):
        _publish(export, output, receipt)
    assert output.read_bytes() == before
    assert not list(output.parent.glob(".*.tmp.*"))


def test_receipt_failure_never_invents_alias_and_retry_recovers(
    tmp_path, monkeypatch
):
    export, output, receipt = _layout(tmp_path)
    _write_export(export, [_event(1, "Niki")])
    real_write_receipt = projector._write_json_file
    failed = False

    def fail_once(path, value):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected receipt failure")
        return real_write_receipt(path, value)

    monkeypatch.setattr(projector, "_write_json_file", fail_once)
    with pytest.raises(OSError, match="injected receipt failure"):
        _publish(export, output, receipt)

    assert _read_projection(output)["aliases"] == {"niki": "alex"}
    assert not receipt.exists()
    recovered = _publish(export, output, receipt)
    assert recovered["projection_sha256"] == _read_projection(output)["receipt"][
        "projection_sha256"
    ]
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o640
