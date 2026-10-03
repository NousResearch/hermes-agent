#!/usr/bin/env python3
"""Validate Kubrick's canonical architecture manifest and derived Hermes surfaces."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from manifest_contract import MANIFEST_PATH, ManifestError, load_manifest

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    errors: list[str] = []
    try:
        manifest = load_manifest()
    except ManifestError as exc:
        manifest = {}
        errors.append(str(exc))

    if manifest:
        expected_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if manifest["version"] != expected_version:
            errors.append(
                f"manifest version {manifest['version']!r} does not match VERSION {expected_version!r}"
            )

        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        declared = set(re.findall(r"`([a-z][a-z-]+)`", skill))
        missing_intents = sorted(set(manifest["intents"]) - declared)
        if missing_intents:
            errors.append(f"SKILL.md does not declare intents: {', '.join(missing_intents)}")

        runtime = (ROOT / "references/hermes-runtime-contract.md").read_text(encoding="utf-8")
        for profile in manifest["runtime_profiles"]:
            if profile not in runtime.lower():
                errors.append(f"runtime contract does not document profile {profile!r}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "version": manifest.get("version"),
        "host": manifest.get("host"),
        "intent_count": len(manifest.get("intents", {})),
        "action_count": sum(
            len(spec.get("actions", {})) for spec in manifest.get("intents", {}).values()
        ),
        "alias_count": len(manifest.get("legacy_aliases", {})),
        "schema_count": len(manifest.get("schemas", [])),
        "provider_count": len(manifest.get("providers", [])),
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
