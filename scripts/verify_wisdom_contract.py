#!/usr/bin/env python3
"""Fail CI when pinned Wisdom artifacts or canonical algorithms drift."""

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_wisdom.contract import (
    CONTRACT_PIN,
    ContentFile,
    author_description_hash,
    canonical_content_manifest,
    derive_content_hash,
    sanitize_author_description,
    sha256_address,
)


CONTRACTS = ROOT / "hermes_wisdom" / "contracts"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-dir", type=Path, help="Verify artifacts against the pinned Gateway git commit")
    args = parser.parse_args()
    openapi = CONTRACTS / "gateway-openapi.json"
    schema = CONTRACTS / "skill-manifest.schema.v1.json"
    vectors_path = CONTRACTS / "canonical-hash-vectors.v1.json"
    assert digest(openapi) == CONTRACT_PIN.openapi_sha256
    assert digest(schema) == CONTRACT_PIN.manifest_schema_sha256
    assert digest(vectors_path) == CONTRACT_PIN.canonical_vectors_sha256
    if args.gateway_dir is not None:
        sources = {
            openapi: "openapi.json",
            schema: "docs/design/collective-wisdom/contracts/skill-manifest.schema.v1.json",
            vectors_path: "docs/design/collective-wisdom/contracts/canonical-hash-vectors.v1.json",
        }
        for artifact, source in sources.items():
            pinned = subprocess.run(
                ["git", "-C", str(args.gateway_dir), "show", f"{CONTRACT_PIN.gateway_commit}:{source}"],
                check=True, capture_output=True,
            ).stdout
            assert artifact.read_bytes() == pinned, f"Pinned Gateway source differs: {source}"
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    for vector in [vectors, *vectors["content_hash_cases"]]:
        files: list[ContentFile] = []
        for item in vector["files"]:
            body = base64.b64decode(item["content_base64"], validate=True)
            assert body.decode("utf-8") == item["content_utf8"]
            assert sha256_address(body) == item["hash"]
            files.append(
                ContentFile(path=item["path"], mode=item["mode"], hash=item["hash"])
            )
        assert (
            canonical_content_manifest(files).decode("utf-8")
            == vector["canonical_content_manifest_utf8"]
        )
        assert derive_content_hash(files) == vector["content_hash"]
    assert (
        next(
            item["hash"]
            for item in vectors["files"]
            if item["path"] == "skill.manifest.json"
        )
        == vectors["package_manifest_hash"]
    )
    canonical = sanitize_author_description(vectors["author_description_input"])
    assert canonical == vectors["canonical_author_description"]
    assert author_description_hash(canonical) == vectors["author_description_hash"]
    print(
        json.dumps(
            {
                "ok": True,
                "gateway_commit": CONTRACT_PIN.gateway_commit,
                "openapi_sha256": digest(openapi),
                "manifest_schema_sha256": digest(schema),
                "canonical_vectors_sha256": digest(vectors_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
