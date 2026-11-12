#!/usr/bin/env python3
"""Load and structurally validate Kubrick's canonical Hermes manifest.

The manifest uses JSON syntax in a .yaml file. JSON is valid YAML, and this keeps
manifest loading available in Kubrick's stdlib runtime profile.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "kubrick.manifest.yaml"
REQUIRED_TOP_LEVEL = {
    "manifest_version",
    "name",
    "version",
    "host",
    "runtime_profiles",
    "intents",
    "legacy_aliases",
    "artifact_types",
    "schemas",
    "providers",
    "recipes",
    "optional_integrations",
    "authority_classes",
    "exit_codes",
}


class ManifestError(ValueError):
    """The canonical manifest is missing, malformed, or internally inconsistent."""


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest must use stdlib-readable JSON/YAML syntax: {exc}") from exc
    errors = validate_manifest(data, root=path.parent)
    if errors:
        raise ManifestError("; ".join(errors))
    return data


def validate_manifest(data: object, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["manifest root must be an object"]

    missing = sorted(REQUIRED_TOP_LEVEL - set(data))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")
        return errors

    if data.get("name") != "kubrick":
        errors.append("name must be 'kubrick'")
    if data.get("host") != "hermes":
        errors.append("host must be 'hermes' on the main branch")
    if data.get("manifest_version") != 1:
        errors.append("unsupported manifest_version")

    profiles = data.get("runtime_profiles")
    if not isinstance(profiles, dict) or not {"prose", "stdlib", "validation", "dev"} <= set(profiles):
        errors.append("runtime_profiles must define prose, stdlib, validation, and dev")

    intents = data.get("intents")
    if not isinstance(intents, dict) or not intents:
        errors.append("intents must be a non-empty object")
        intents = {}
    for intent, spec in intents.items():
        if not isinstance(spec, dict):
            errors.append(f"intent {intent!r} must be an object")
            continue
        actions = spec.get("actions")
        if not isinstance(actions, dict) or not actions:
            errors.append(f"intent {intent!r} must define actions")
            continue
        default = spec.get("default_action")
        if default is not None and default not in actions:
            errors.append(f"intent {intent!r} default action {default!r} is not declared")
        for action, script in actions.items():
            if not isinstance(script, str) or not (root / "scripts" / script).is_file():
                errors.append(f"intent {intent!r} action {action!r} references missing script {script!r}")

    aliases = data.get("legacy_aliases")
    if not isinstance(aliases, dict):
        errors.append("legacy_aliases must be an object")
        aliases = {}
    for alias, spec in aliases.items():
        if not isinstance(spec, dict):
            errors.append(f"alias {alias!r} must be an object")
            continue
        intent = spec.get("intent")
        action = spec.get("action")
        if intent not in intents:
            errors.append(f"alias {alias!r} references unknown intent {intent!r}")
        elif action not in intents[intent].get("actions", {}):
            errors.append(f"alias {alias!r} references unknown action {intent}:{action}")

    schemas = data.get("schemas")
    if not isinstance(schemas, list) or not schemas:
        errors.append("schemas must be a non-empty list")
    else:
        for relative in schemas:
            if not isinstance(relative, str) or not (root / relative).is_file():
                errors.append(f"schema path is missing: {relative!r}")

    if not isinstance(data.get("providers"), list) or "generic" not in data["providers"]:
        errors.append("providers must include generic")
    if not isinstance(data.get("artifact_types"), list) or not data["artifact_types"]:
        errors.append("artifact_types must be a non-empty list")
    if not isinstance(data.get("recipes"), dict):
        errors.append("recipes must be an object")
    if data.get("authority_classes") != [
        "PROPOSED", "OBSERVATION", "NOT_COMPUTABLE", "AUTHORITATIVE_EXTERNAL"
    ]:
        errors.append("authority_classes must preserve proposal-only local authority ordering")
    if set(data.get("exit_codes", {})) != {"0", "1", "2", "3", "4"}:
        errors.append("exit_codes must define stable codes 0 through 4")

    return errors
