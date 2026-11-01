#!/usr/bin/env python3
"""Deterministic runtime identities for Kubrick receipts.

All hashes use canonical relative paths and file bytes. Absolute checkout paths,
file metadata, timestamps, and environment details are excluded.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_bundle_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    normalized = sorted({path.resolve() for path in paths if path.is_file()})
    for path in normalized:
        relative = path.relative_to(root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "kubrick.manifest.yaml").read_text(encoding="utf-8"))


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*([^#\s]+)", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def corpus_files(root: Path = ROOT) -> list[Path]:
    references = root / "references"
    files = [
        references / "corpus-index.yaml",
        references / "executable-corpus-registry.yaml",
        references / "symbolic-narrative-patterns.yaml",
        references / "narrative-affordance-registry.md",
        references / "transformation-grammar-registry.md",
    ]
    for directory in ("patterns", "pattern-manifests", "corpus"):
        base = references / directory
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return files


def corpus_identity(root: Path = ROOT) -> tuple[str, str]:
    index = root / "references/corpus-index.yaml"
    version = _yaml_scalar(index, "index_version")
    return version, file_bundle_digest(root, corpus_files(root))


def schema_bundle_identity(root: Path = ROOT) -> str:
    manifest = _manifest(root)
    paths = [root / relative for relative in manifest["schemas"]]
    return f"sha256:{file_bundle_digest(root, paths)}"


def provider_adapter_identity(provider: str, root: Path = ROOT) -> str:
    if provider == "none":
        return "none"
    common = [root / "scripts/build_model_adapter_packet.py"]
    provider_scripts = {
        "generic": [root / "scripts/adapt_provider.py"],
        "grok-imagine": [root / "scripts/adapt_provider.py", root / "scripts/adapt_grok_imagine.py"],
        "flux": [root / "scripts/adapt_provider.py", root / "scripts/adapt_flux.py"],
        "sd3": [root / "scripts/adapt_provider.py", root / "scripts/adapt_sd3.py"],
        "midjourney": [root / "scripts/adapt_provider.py", root / "scripts/adapt_midjourney.py"],
    }
    return f"sha256:{file_bundle_digest(root, common + provider_scripts.get(provider, []))}"


def compile_identity(
    normalized_input: dict[str, Any],
    *,
    mode: str,
    provider: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    manifest = _manifest(root)
    corpus_version, corpus_digest = corpus_identity(root)
    options = {"mode": mode, "provider": provider}
    return {
        "kubrick_version": manifest["version"],
        "corpus_version": corpus_version,
        "corpus_digest": f"sha256:{corpus_digest}",
        "schema_bundle_version": schema_bundle_identity(root),
        "provider_adapter_version": provider_adapter_identity(provider, root),
        "command": "compile",
        "normalized_options": options,
        "normalized_input_digest": f"sha256:{value_digest(normalized_input)}",
        "command_digest": f"sha256:{value_digest({'command': 'compile', 'options': options})}",
    }
