#!/usr/bin/env python3
"""Compare Aegis QA LCM artifact/config manifest against Apollo target.

Inputs are JSON files or Markdown reports containing a fenced JSON manifest. The
script compares the plugin artifact hash plus the LCM threshold config,
redaction ruleset, schema version, and encryption mode. It never mutates config
or controls any live process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "threshold_config",
    "redaction_ruleset",
    "schema_version",
    "encryption_mode",
)
HASH_FIELD = "plugin_artifact_sha256"
PATH_FIELD = "plugin_artifact_path"


@dataclass(frozen=True)
class Comparison:
    name: str
    aegis_value: str
    apollo_value: str
    passed: bool


@dataclass(frozen=True)
class PreflightResult:
    status: str
    comparisons: dict[str, Comparison]
    failures: list[str]
    aegis_report: Path
    apollo_target: Path
    out_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def hash_artifact(path: str | Path) -> str:
    artifact = Path(path).expanduser().resolve()
    if artifact.is_file():
        digest = hashlib.sha256()
        digest.update(artifact.read_bytes())
        return digest.hexdigest()
    if artifact.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in artifact.rglob("*") if p.is_file()):
            if "__pycache__" in child.parts:
                continue
            rel = child.relative_to(artifact).as_posix().encode("utf-8")
            digest.update(rel)
            digest.update(b"\0")
            digest.update(child.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(f"plugin artifact path does not exist: {artifact}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    text = manifest_path.read_text(encoding="utf-8")
    parsed = _parse_manifest_text(text, manifest_path)
    if not isinstance(parsed, dict):
        raise ValueError(f"manifest in {manifest_path} must be a JSON object")
    return parsed


def _parse_manifest_text(text: str, manifest_path: Path) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _looks_like_manifest(parsed):
            return parsed
    raise ValueError(f"could not find JSON preflight manifest in {manifest_path}")


def _looks_like_manifest(value: dict[str, Any]) -> bool:
    return (HASH_FIELD in value or PATH_FIELD in value) and all(field in value for field in REQUIRED_FIELDS)


def _artifact_hash(manifest: dict[str, Any], *, manifest_path: Path) -> str:
    existing = manifest.get(HASH_FIELD)
    if existing is not None:
        return str(existing).strip().lower()
    artifact = manifest.get(PATH_FIELD)
    if not artifact:
        raise ValueError(f"{manifest_path} must include {HASH_FIELD} or {PATH_FIELD}")
    artifact_path = Path(str(artifact)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    return hash_artifact(artifact_path)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _field_value(manifest: dict[str, Any], field: str, *, manifest_path: Path) -> str:
    if field == HASH_FIELD:
        return _artifact_hash(manifest, manifest_path=manifest_path)
    if field not in manifest:
        raise ValueError(f"{manifest_path} missing required field {field}")
    return _canonical(manifest[field])


def compare_manifests(
    aegis_manifest: dict[str, Any],
    apollo_manifest: dict[str, Any],
    *,
    aegis_path: Path,
    apollo_path: Path,
) -> dict[str, Comparison]:
    comparisons: dict[str, Comparison] = {}
    for field in (HASH_FIELD, *REQUIRED_FIELDS):
        aegis_value = _field_value(aegis_manifest, field, manifest_path=aegis_path)
        apollo_value = _field_value(apollo_manifest, field, manifest_path=apollo_path)
        comparisons[field] = Comparison(
            name=field,
            aegis_value=aegis_value,
            apollo_value=apollo_value,
            passed=aegis_value == apollo_value,
        )
    return comparisons


def run_preflight(*, aegis_report: str | Path, apollo_target: str | Path, out_path: str | Path | None) -> PreflightResult:
    aegis_path = Path(aegis_report).expanduser().resolve()
    apollo_path = Path(apollo_target).expanduser().resolve()
    aegis_manifest = load_manifest(aegis_path)
    apollo_manifest = load_manifest(apollo_path)
    comparisons = compare_manifests(
        aegis_manifest,
        apollo_manifest,
        aegis_path=aegis_path,
        apollo_path=apollo_path,
    )
    failures = [
        f"{name} drift: Aegis={comparison.aegis_value} Apollo={comparison.apollo_value}"
        for name, comparison in comparisons.items()
        if not comparison.passed
    ]
    status = "PASS" if not failures else "FAIL-LOUD"
    resolved_out = Path(out_path).expanduser().resolve() if out_path is not None else None
    result = PreflightResult(status, comparisons, failures, aegis_path, apollo_path, resolved_out)
    if resolved_out is not None:
        write_report(resolved_out, result)
    return result


def write_report(out_path: Path, result: PreflightResult) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# PRD-6 LCM Config/Hash Preflight",
        "",
        f"Generated: {ts}",
        f"Status: {result.status}",
        f"Aegis QA report: {result.aegis_report}",
        f"Apollo target: {result.apollo_target}",
        "",
        "## Comparisons",
        "",
        "| field | status | Aegis | Apollo |",
        "|---|---|---|---|",
    ]
    for name, comparison in result.comparisons.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    "PASS" if comparison.passed else "DRIFT",
                    _md(_shorten(comparison.aegis_value)),
                    _md(_shorten(comparison.apollo_value)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failures", ""])
    if result.failures:
        lines.extend(f"- {failure}" for failure in result.failures)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Safety notes",
        "",
        "- This preflight is read-only: it hashes artifacts and compares manifests only.",
        "- FAIL-LOUD is the alert contract for Apollo routing when drift is detected.",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shorten(value: str, max_len: int = 160) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aegis-report", required=True, help="Aegis QA JSON/Markdown manifest path.")
    parser.add_argument("--apollo-target", required=True, help="Apollo target JSON/Markdown manifest path.")
    parser.add_argument("--out", help="Optional Markdown report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_preflight(
            aegis_report=args.aegis_report,
            apollo_target=args.apollo_target,
            out_path=args.out,
        )
    except Exception as exc:
        print(f"FAIL-LOUD: {exc}", file=sys.stderr)
        return 2
    print(f"status={result.status} comparisons={len(result.comparisons)} report={result.out_path or '(not written)'}")
    if result.failures:
        for failure in result.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
