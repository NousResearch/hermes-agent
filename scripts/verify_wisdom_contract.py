#!/usr/bin/env python3
"""Fail CI when pinned Wisdom artifacts or canonical algorithms drift."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_wisdom.contract import (
    CONTRACT_PIN,
    ContentFile,
    author_description_hash,
    derive_content_hash,
    sanitize_author_description,
    sha256_address,
)


CONTRACTS = ROOT / "hermes_wisdom" / "contracts"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    openapi = CONTRACTS / "gateway-openapi.json"
    schema = CONTRACTS / "skill-manifest.schema.v1.json"
    vectors_path = CONTRACTS / "canonical-hash-vectors.v1.json"
    assert digest(openapi) == CONTRACT_PIN.openapi_sha256
    assert digest(schema) == CONTRACT_PIN.manifest_schema_sha256
    assert digest(vectors_path) == CONTRACT_PIN.canonical_vectors_sha256
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    files: list[ContentFile] = []
    for item in vectors["files"]:
        body = base64.b64decode(item["content_base64"], validate=True)
        assert sha256_address(body) == item["hash"]
        files.append(
            ContentFile(path=item["path"], mode=item["mode"], hash=item["hash"])
        )
    assert derive_content_hash(files) == vectors["content_hash"]
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
