from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gateway import production_secret_stager as stager
from gateway.operational_edge_catalog import CREDENTIALS_BY_DOMAIN


BEARER = "stager-api-bearer-tests-0123456789abcdef"
PASSKEY = "stager-owner-passkey-tests-0123456789abcdef"


def _write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o400)
    os.chown(path, os.geteuid(), os.getegid())


def _paths(tmp_path: Path) -> dict[str, Path]:
    host = tmp_path / "staged" / "host"
    keys = tmp_path / "staged" / "keys"
    host.mkdir(parents=True, mode=0o700)
    keys.mkdir(parents=True, mode=0o700)
    host.chmod(0o700)
    keys.chmod(0o700)
    os.chown(host, os.geteuid(), os.getegid())
    os.chown(keys, os.geteuid(), os.getegid())
    bearer = tmp_path / "source" / "api-bearer"
    approval = tmp_path / "source" / "approval-passkey"
    _write_secret(bearer, BEARER)
    _write_secret(approval, PASSKEY)
    return {
        "bearer_source": bearer,
        "approval_source": approval,
        "bearer_verifier_path": host / "api-server-bearer-sha256.json",
        "approval_verifier_path": host / "api-approval-passkey-scrypt.json",
        "writer_private_path": keys / "writer-capability-private.pem",
        "edge_private_path": keys / "discord-edge-receipt-private.pem",
    }


def test_stage_is_create_only_retryable_and_never_records_private_material(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    first = stager.stage_production_secret_foundation(
        **paths,
        require_root=False,
    )
    second = stager.stage_production_secret_foundation(
        **paths,
        require_root=False,
    )

    assert all(first["created"].values())
    assert not any(second["created"].values())
    assert first["writer_public_key_id"] != first["edge_public_key_id"]
    assert second["writer_public_key_id"] == first["writer_public_key_id"]
    assert second["edge_public_key_id"] == first["edge_public_key_id"]
    assert first["schema"] == "muncho-production-secret-staging.v2"
    operational = first["operational_edge_key_foundation"]
    assert operational["writer_public_key_id"] == first["writer_public_key_id"]
    assert first["operational_edge_key_foundation_sha256"] == operational[
        "receipt_sha256"
    ]
    assert first["operational_edge_receipt_public_key_ids"] == {
        row["domain"]: row["public_key_id"] for row in operational["keys"]
    }
    assert len(operational["keys"]) == len(CREDENTIALS_BY_DOMAIN)
    assert all(row["created"] is True for row in operational["keys"])
    assert all(
        row["created"] is False
        for row in second["operational_edge_key_foundation"]["keys"]
    )
    assert second["operational_edge_receipt_public_key_ids"] == first[
        "operational_edge_receipt_public_key_ids"
    ]
    public_receipt = json.dumps(first, sort_keys=True)
    assert BEARER not in public_receipt
    assert PASSKEY not in public_receipt
    for name in (
        "bearer_verifier_path",
        "approval_verifier_path",
        "writer_private_path",
        "edge_private_path",
    ):
        observed = Path(first[name])
        assert observed.is_file()
        assert os.stat(observed).st_mode & 0o777 == 0o400
        assert BEARER.encode() not in observed.read_bytes()
        assert PASSKEY.encode() not in observed.read_bytes()
    for row in operational["keys"]:
        for name, mode in (("private_path", 0o400), ("public_path", 0o444)):
            observed = Path(row[name])
            assert observed.is_file()
            assert os.stat(observed).st_mode & 0o777 == mode
            assert BEARER.encode() not in observed.read_bytes()
            assert PASSKEY.encode() not in observed.read_bytes()
    assert first["private_content_or_digest_recorded"] is False
    assert first["secret_material_recorded"] is False
    assert first["secret_digest_recorded"] is False


def test_stage_rejects_secret_drift_instead_of_replacing_verifier(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    stager.stage_production_secret_foundation(**paths, require_root=False)
    paths["approval_source"].chmod(0o600)
    _write_secret(paths["approval_source"], "different-passkey-0123456789abcdef012345")

    with pytest.raises(
        stager.ProductionSecretStagingError,
        match="approval_verifier_conflict",
    ):
        stager.stage_production_secret_foundation(**paths, require_root=False)


def test_stage_rejects_symlinked_source_and_artifact(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    real = paths["bearer_source"]
    linked = real.with_name("linked-bearer")
    linked.symlink_to(real)
    paths["bearer_source"] = linked
    with pytest.raises(
        stager.ProductionSecretStagingError,
        match="provenance_invalid",
    ):
        stager.stage_production_secret_foundation(**paths, require_root=False)

    paths = _paths(tmp_path / "artifact")
    target = paths["bearer_verifier_path"]
    target.symlink_to(paths["bearer_source"])
    with pytest.raises(
        stager.ProductionSecretStagingError,
        match="artifact_conflict",
    ):
        stager.stage_production_secret_foundation(**paths, require_root=False)
