#!/usr/bin/env python3
"""Validate static SpecialistDescriptorV1 artifacts without profile mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ares_runtime.collaboration import (
    ContractError,
    digest,
    validate_specialist_descriptor_set,
)

MANIFEST_REQUIRED = {"schema", "profile_ids", "descriptors", "manifest_digest"}
MANIFEST_SCHEMA = "AresSpecialistDescriptorManifestV1"
CANONICAL_SEMANTIC_REGISTRY_REF = "docs:role-contracts/role-contracts.json"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_descriptor_provenance(
    descriptors: list[dict[str, Any]], *, registry_digest: str
) -> list[str]:
    """Bind every static descriptor to the exact registry bytes this run read."""
    errors: list[str] = []
    for raw in descriptors:
        profile_id = raw.get("profile_id")
        provenance = raw.get("provenance")
        label = profile_id if isinstance(profile_id, str) else "unknown"
        if not isinstance(provenance, dict):
            # Schema validation reports this shape failure with more context.
            continue
        if provenance.get("semantic_registry_ref") != CANONICAL_SEMANTIC_REGISTRY_REF:
            errors.append(f"SEMANTIC_REGISTRY_REF_MISMATCH: {label}")
        if provenance.get("semantic_registry_digest") != registry_digest:
            errors.append(f"SEMANTIC_REGISTRY_DIGEST_MISMATCH: {label}")
    return errors


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _role_artifacts(registry: Any) -> tuple[dict[str, list[str]], list[str]]:
    errors: list[str] = []
    if not isinstance(registry, dict) or not isinstance(registry.get("roles"), list):
        return {}, ["registry roles must be a list"]
    result: dict[str, list[str]] = {}
    for role in registry["roles"]:
        if not isinstance(role, dict):
            errors.append("registry contains a non-object role")
            continue
        role_id = role.get("role_id")
        artifacts = role.get("required_artifacts")
        if not isinstance(role_id, str) or not role_id:
            errors.append("registry role missing role_id")
            continue
        if role_id in result:
            errors.append(f"registry contains duplicate role_id: {role_id}")
            continue
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"registry role has no required artifacts: {role_id}")
            continue
        artifact_ids: list[str] = []
        for artifact in artifacts:
            artifact_id = (
                artifact.get("artifact_id") if isinstance(artifact, dict) else None
            )
            if not isinstance(artifact_id, str) or not artifact_id:
                errors.append(
                    f"registry role has invalid required artifact identity: {role_id}"
                )
                break
            artifact_ids.append(artifact_id)
        if len(artifact_ids) != len(artifacts):
            errors.append(
                f"registry role has invalid required artifact identity: {role_id}"
            )
            continue
        if len(set(artifact_ids)) != len(artifact_ids):
            errors.append(
                f"registry role has duplicate required artifact identity: {role_id}"
            )
            continue
        result[role_id] = artifact_ids
    return result, errors


def _descriptor_files(directory: Path, manifest: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.json") if path != manifest)


def _validate_manifest(manifest: Any, descriptors: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if set(manifest) != MANIFEST_REQUIRED:
        return ["manifest fields do not match the canonical manifest contract"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest schema is invalid")
    supplied = manifest.get("manifest_digest")
    without_digest = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if supplied != digest(without_digest):
        errors.append("manifest digest mismatch")
    profile_ids = manifest.get("profile_ids")
    if (
        not isinstance(profile_ids, list)
        or profile_ids != sorted(profile_ids)
        or len(profile_ids) != len(set(profile_ids))
    ):
        errors.append("manifest profile_ids must be sorted and unique")
    actual = sorted(descriptors, key=lambda item: item["profile_id"])
    expected_entries = [
        {
            "profile_id": item["profile_id"],
            "path": f"{item['profile_id']}.json",
            "descriptor_digest": item["descriptor_digest"],
        }
        for item in actual
    ]
    if profile_ids != [item["profile_id"] for item in actual]:
        errors.append("manifest profile_ids do not match descriptor artifacts")
    if manifest.get("descriptors") != expected_entries:
        errors.append("manifest descriptors do not match descriptor artifacts")
    return errors


def validate(
    *,
    registry_path: Path,
    descriptor_dir: Path,
    profile_ids: list[str],
    manifest_path: Path,
    require_disabled: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        registry = _read_json(registry_path)
        registry_digest = _sha256_file(registry_path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"registry unreadable: {exc}"]
    role_artifacts, registry_errors = _role_artifacts(registry)
    errors.extend(registry_errors)

    raw_descriptors: list[dict[str, Any]] = []
    for path in _descriptor_files(descriptor_dir, manifest_path):
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"descriptor unreadable {path.name}: {exc}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"descriptor must be an object: {path.name}")
            continue
        raw_descriptors.append(raw)
    errors.extend(
        _validate_descriptor_provenance(raw_descriptors, registry_digest=registry_digest)
    )
    errors.extend(
        validate_specialist_descriptor_set(
            raw_descriptors,
            profile_ids=profile_ids,
            semantic_role_artifacts=role_artifacts,
            require_disabled=require_disabled,
        )
    )
    manifest: Any = None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"manifest unreadable: {exc}")
    else:
        errors.extend(_validate_manifest(manifest, raw_descriptors))

    if errors:
        return None, errors
    return {
        "schema": "AresSpecialistDescriptorValidationV1",
        "ok": True,
        "profile_count": len(profile_ids),
        "descriptor_count": len(raw_descriptors),
        "enabled_descriptor_count": sum(item["enabled"] for item in raw_descriptors),
        "semantic_registry_ref": CANONICAL_SEMANTIC_REGISTRY_REF,
        "semantic_registry_digest": registry_digest,
        "validator_sha256": _sha256_file(Path(__file__)),
        "descriptor_digests": [
            item["descriptor_digest"]
            for item in sorted(raw_descriptors, key=lambda item: item["profile_id"])
        ],
        "manifest_digest": manifest["manifest_digest"],
    }, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--descriptor-dir", required=True, type=Path)
    parser.add_argument("--profile", action="append", required=True, dest="profiles")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-enabled", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.manifest or args.descriptor_dir / "manifest.json"
    result, errors = validate(
        registry_path=args.registry,
        descriptor_dir=args.descriptor_dir,
        profile_ids=args.profiles,
        manifest_path=manifest_path,
        require_disabled=not args.allow_enabled,
    )
    if errors:
        print(
            json.dumps(
                {
                    "schema": "AresSpecialistDescriptorValidationV1",
                    "ok": False,
                    "errors": errors,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
