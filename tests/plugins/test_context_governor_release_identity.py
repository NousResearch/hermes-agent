"""Hostile contract tests for sealed Context Governor release identity."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from ares_runtime import (
    ActivationGrant,
    AresRuntimeLayout,
    materialize_candidate_release,
)
from ares_runtime.image import RELEASE_MANIFEST_SCHEMA


MODULE = (
    Path(__file__).parents[2]
    / "plugins/context_engine/_context_governor/release_identity.py"
)
SPEC = importlib.util.spec_from_file_location("cg_release_identity", MODULE)
assert SPEC and SPEC.loader
identity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = identity
SPEC.loader.exec_module(identity)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _document(schema: str, id_field: str, **fields: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": schema,
        "canonicalization_version": identity.CANONICALIZATION_VERSION,
        **fields,
    }
    value[id_field] = identity.object_id(value, id_field)
    return value


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_bytes(identity.canonical_json(value) + b"\n")
    return path


def _archive(
    path: Path,
    files: dict[str, bytes],
    *,
    duplicate: bool = False,
    traversal: bool = False,
) -> None:
    with tarfile.open(path, "w:") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
            if duplicate:
                bundle.addfile(member, io.BytesIO(data))
        if traversal:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))


def _fixture(tmp_path: Path) -> dict[str, Path]:
    files = {
        "bin/context-governor": b"governor-bytes",
        "adapter/__init__.py": b"adapter-bytes",
        "bootstrap.py": b"bootstrap-bytes",
        "config.template.json": b"template-bytes",
        "rendered.json": b"rendered-bytes",
    }
    runtime_tree_sha256 = _digest(b"[]\n")
    release_manifest = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "runtime_tree_sha256": runtime_tree_sha256,
        "files": [],
    }
    files["payload/release-manifest.json"] = (
        json.dumps(release_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    archive = tmp_path / "candidate.tar"
    entries = [
        {"path": name, "sha256": _digest(data), "size": len(data), "mode": 0o644}
        for name, data in files.items()
    ]
    core = _document(
        identity.CANDIDATE_CORE_SCHEMA,
        "candidate_id",
        payload_files=entries,
        payload_tree_sha256=identity._payload_digest(entries),
        binary_path="bin/context-governor",
        binary_sha256=_digest(files["bin/context-governor"]),
        adapter_bundle_path="adapter/__init__.py",
        adapter_bundle_sha256=_digest(files["adapter/__init__.py"]),
        activation_bootstrap_path="bootstrap.py",
        activation_bootstrap_sha256=_digest(files["bootstrap.py"]),
        config_template_path="config.template.json",
        config_template_sha256=_digest(files["config.template.json"]),
        release_manifest_path="payload/release-manifest.json",
        release_manifest_sha256=_digest(files["payload/release-manifest.json"]),
        runtime_tree_sha256=runtime_tree_sha256,
    )
    cert = _document(
        identity.CERTIFICATION_SET_SCHEMA,
        "certification_set_id",
        candidate_id=core["candidate_id"],
        artifacts=[],
    )
    _archive(
        archive,
        {
            **files,
            "candidate-core-manifest.json": identity.canonical_json(core) + b"\n",
            "certification-set-manifest.json": identity.canonical_json(cert) + b"\n",
        },
    )
    sealed = _document(
        identity.SEALED_CANDIDATE_SCHEMA,
        "sealed_candidate_id",
        candidate_id=core["candidate_id"],
        certification_set_id=cert["certification_set_id"],
        archive_sha256=_digest(archive.read_bytes()),
    )
    post = _document(
        identity.POST_SEAL_EVIDENCE_SCHEMA,
        "post_seal_evidence_set_id",
        sealed_candidate_id=sealed["sealed_candidate_id"],
        archive_sha256=sealed["archive_sha256"],
        artifacts=[],
    )
    releases = tmp_path / "releases"
    auth = _document(
        identity.ACTIVATION_AUTHORIZATION_SCHEMA,
        "activation_authorization_id",
        candidate_id=core["candidate_id"],
        certification_set_id=cert["certification_set_id"],
        sealed_candidate_id=sealed["sealed_candidate_id"],
        post_seal_evidence_set_id=post["post_seal_evidence_set_id"],
        archive_sha256=sealed["archive_sha256"],
        authorization_state="NON_AUTHORIZING",
        non_authorizing=True,
        rendered_config_path="rendered.json",
        rendered_config_sha256=_digest(files["rendered.json"]),
        approved_release_root=str(releases / sealed["sealed_candidate_id"]),
        governed_key_policy={
            "snapshot_schema": "AresContextGovernorKeySnapshotV2",
            "authority": "descriptor-backed-ares-owned",
            "caller_key_material": "forbidden",
        },
    )
    return {
        "archive": archive,
        "core": _write_json(tmp_path / "core.json", core),
        "cert": _write_json(tmp_path / "cert.json", cert),
        "sealed": _write_json(tmp_path / "sealed.json", sealed),
        "post": _write_json(tmp_path / "post.json", post),
        "auth": _write_json(tmp_path / "auth.json", auth),
        "releases": releases,
    }


def _materialize(paths: dict[str, Path]):
    return identity.materialize_verified_release(
        archive=paths["archive"],
        candidate_core=paths["core"],
        certification_set=paths["cert"],
        sealed_candidate=paths["sealed"],
        post_seal_evidence=paths["post"],
        authorization=paths["auth"],
        release_parent=paths["releases"],
    )


def test_verified_release_uses_only_the_sealed_archive(tmp_path: Path):
    paths = _fixture(tmp_path)
    verified = _materialize(paths)

    assert verified.binary.read_bytes() == b"governor-bytes"
    assert verified.adapter_root.read_bytes() == b"adapter-bytes"
    assert verified.release_root.name == verified.sealed_candidate_id


def test_ares_materializer_seals_the_verified_release_tree(tmp_path: Path):
    paths = _fixture(tmp_path)
    auth = json.loads(paths["auth"].read_text())
    auth["approved_release_root"] = f"content-addressed/{auth['sealed_candidate_id']}"
    auth["activation_authorization_id"] = identity.object_id(
        auth, "activation_authorization_id"
    )
    _write_json(paths["auth"], auth)
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    grant = ActivationGrant(
        candidate_id=str(json.loads(paths["core"].read_text())["candidate_id"]),
        certification_set_id=str(
            json.loads(paths["cert"].read_text())["certification_set_id"]
        ),
        sealed_candidate_id=str(auth["sealed_candidate_id"]),
        audit_subject_id="a" * 64,
        audit_subject_sha256="b" * 64,
        audit_result_sha256="c" * 64,
        archive_sha256=str(auth["archive_sha256"]),
        candidate_core_sha256=_digest(paths["core"].read_bytes()),
        sealed_manifest_sha256=_digest(paths["sealed"].read_bytes()),
        release_manifest_sha256=str(
            json.loads(paths["core"].read_text())["release_manifest_sha256"]
        ),
        runtime_tree_sha256=str(
            json.loads(paths["core"].read_text())["runtime_tree_sha256"]
        ),
        custody_event_sequence=2,
        target_platform="linux-x86_64",
        target_release_root=str(layout.releases_dir),
        materializer_contract="AresReleaseMaterializerV1",
        activator_contract="AresReleaseActivatorV1",
        resolver_contract="AresRuntimeResolverV1",
    ).with_grant_id()

    release = materialize_candidate_release(
        archive=paths["archive"],
        candidate_core=paths["core"],
        certification_set=paths["cert"],
        sealed_candidate=paths["sealed"],
        post_seal_evidence=paths["post"],
        authorization=paths["auth"],
        layout=layout,
        grant=grant,
    )

    assert release.release_root == layout.release_dir(grant.sealed_candidate_id)
    assert release.release_root.stat().st_mode & 0o222 == 0
    assert (release.release_root / "bin/context-governor").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("state", [None, "AUTHORIZED", "UNKNOWN_FUTURE_MODE"])
def test_release_materialization_rejects_non_dry_run_authority_states(
    tmp_path: Path, state: str | None
):
    paths = _fixture(tmp_path)
    auth = json.loads(paths["auth"].read_text())
    if state is None:
        auth.pop("authorization_state")
    else:
        auth["authorization_state"] = state
    auth["activation_authorization_id"] = identity.object_id(
        auth, "activation_authorization_id"
    )
    _write_json(paths["auth"], auth)

    with pytest.raises(identity.ReleaseIdentityError, match="NonAuthorizingArtifact"):
        _materialize(paths)
    assert not paths["releases"].exists()


@pytest.mark.parametrize(
    "field,code",
    [
        ("candidate_id", "WrongCandidateId"),
        ("certification_set_id", "WrongCertificationSetId"),
        ("sealed_candidate_id", "WrongSealedCandidateId"),
    ],
)
def test_identity_substitution_fails_before_publication(
    tmp_path: Path, field: str, code: str
):
    paths = _fixture(tmp_path)
    name = {
        "candidate_id": "core",
        "certification_set_id": "cert",
        "sealed_candidate_id": "sealed",
    }[field]
    value = json.loads(paths[name].read_text())
    value[field] = "0" * 64
    _write_json(paths[name], value)

    with pytest.raises(identity.ReleaseIdentityError, match=code):
        _materialize(paths)
    assert not paths["releases"].exists()


def test_archive_and_adapter_mutation_fail_before_publication(tmp_path: Path):
    paths = _fixture(tmp_path)
    _archive(
        paths["archive"],
        {
            "bin/context-governor": b"governor-bytes",
            "adapter/__init__.py": b"counterfeit",
        },
    )

    with pytest.raises(identity.ReleaseIdentityError, match="WrongArchiveDigest"):
        _materialize(paths)
    assert not paths["releases"].exists()


def test_duplicate_and_traversal_archive_members_are_rejected(tmp_path: Path):
    paths = _fixture(tmp_path)
    _archive(paths["archive"], {"bin/context-governor": b"x"}, duplicate=True)
    with pytest.raises(identity.ReleaseIdentityError, match="WrongArchiveDigest"):
        _materialize(paths)
    _archive(paths["archive"], {"bin/context-governor": b"x"}, traversal=True)
    sealed = json.loads(paths["sealed"].read_text())
    sealed["archive_sha256"] = _digest(paths["archive"].read_bytes())
    sealed["sealed_candidate_id"] = identity.object_id(sealed, "sealed_candidate_id")
    _write_json(paths["sealed"], sealed)
    post = json.loads(paths["post"].read_text())
    post["sealed_candidate_id"] = sealed["sealed_candidate_id"]
    post["archive_sha256"] = sealed["archive_sha256"]
    post["post_seal_evidence_set_id"] = identity.object_id(
        post, "post_seal_evidence_set_id"
    )
    _write_json(paths["post"], post)
    auth = json.loads(paths["auth"].read_text())
    auth["sealed_candidate_id"] = sealed["sealed_candidate_id"]
    auth["post_seal_evidence_set_id"] = post["post_seal_evidence_set_id"]
    auth["archive_sha256"] = sealed["archive_sha256"]
    auth["approved_release_root"] = str(
        paths["releases"] / sealed["sealed_candidate_id"]
    )
    auth["activation_authorization_id"] = identity.object_id(
        auth, "activation_authorization_id"
    )
    _write_json(paths["auth"], auth)
    with pytest.raises(identity.ReleaseIdentityError, match="UnsafeReleasePath"):
        _materialize(paths)
