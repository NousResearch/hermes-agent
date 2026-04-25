#!/usr/bin/env python3
"""Validate low-risk Hermes agent interop planning artifacts.

This is intentionally offline: it reads Markdown/JSON artifacts and verifies that
security, privacy, policy, budget, and observability gates are documented before
any A2A/background-computer-use implementation is enabled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "ops" / "agent_interop_security_model.md"
DEFAULT_SCHEMA = REPO_ROOT / "ops" / "agent_interop_policy_schema.json"
DEFAULT_OUT = Path.home() / ".hermes" / "reports" / "agent_interop_artifacts.md"

REQUIRED_SECURITY_TERMS = [
    "Bearer token",
    "HMAC",
    "timestamp",
    "nonce",
    "replay",
    "allowed_peers",
    "privacy boundary",
    "outbound filter",
    "redact",
    "credential",
    "token budget",
    "Auto-Concise",
    "background computer-use",
    "disabled by default",
    "manual kill switch",
]

REQUIRED_DEFAULT_POLICY = {
    "external_send_allowed": False,
    "dangerous_command_mode": "deny",
}


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate(model_path: Path, schema_path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    model_text = model_path.read_text(encoding="utf-8") if model_path.exists() else ""
    if not model_text:
        errors.append(f"missing or empty security model: {model_path}")
    else:
        for term in REQUIRED_SECURITY_TERMS:
            if term.lower() not in model_text.lower():
                errors.append(f"security model missing required term: {term}")

    try:
        schema = _load_json(schema_path)
    except Exception as exc:  # noqa: BLE001 - validator should report all failures cleanly
        return errors + [f"cannot read schema {schema_path}: {exc}"], warnings, {}

    for key in ("a2a_peer_policy_required_fields", "task_observability_required_fields", "default_a2a_policy"):
        if key not in schema:
            errors.append(f"schema missing top-level key: {key}")

    peer_fields = schema.get("a2a_peer_policy_required_fields", [])
    obs_fields = schema.get("task_observability_required_fields", [])
    if not isinstance(peer_fields, list) or len(peer_fields) < 10:
        errors.append("a2a_peer_policy_required_fields must list at least 10 fields")
    if not isinstance(obs_fields, list) or len(obs_fields) < 10:
        errors.append("task_observability_required_fields must list at least 10 fields")

    default_policy = schema.get("default_a2a_policy", {})
    if not isinstance(default_policy, dict):
        errors.append("default_a2a_policy must be an object")
        default_policy = {}
    for field, expected in REQUIRED_DEFAULT_POLICY.items():
        if default_policy.get(field) != expected:
            errors.append(f"default_a2a_policy.{field} must be {expected!r}")
    if default_policy.get("write_roots"):
        errors.append("default_a2a_policy.write_roots must be empty for readonly default")
    if default_policy.get("allowed_toolsets"):
        errors.append("default_a2a_policy.allowed_toolsets must be empty for readonly default")
    blocked = default_policy.get("blocked_roots", [])
    if not isinstance(blocked, list) or "~/.credentials" not in blocked:
        errors.append("default_a2a_policy.blocked_roots must include ~/.credentials")

    event_template = schema.get("observability_event_template", {})
    if not isinstance(event_template, dict):
        errors.append("observability_event_template must be an object")
    else:
        missing_obs = [field for field in obs_fields if field not in event_template]
        if missing_obs:
            errors.append("observability_event_template missing fields: " + ", ".join(missing_obs))
        if event_template.get("redaction_count") != 0:
            warnings.append("observability_event_template.redaction_count should default to 0")

    summary = {
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "security_terms_checked": len(REQUIRED_SECURITY_TERMS),
        "peer_policy_fields": len(peer_fields) if isinstance(peer_fields, list) else 0,
        "observability_fields": len(obs_fields) if isinstance(obs_fields, list) else 0,
    }
    return errors, warnings, summary


def render_report(errors: list[str], warnings: list[str], summary: dict[str, Any]) -> str:
    lines = [
        "# Hermes Agent Interop Artifact Validation",
        "",
        "This report is generated offline and does not enable A2A, MCP, or background computer-use.",
        "",
        "## Summary",
        "",
        f"- Security model: `{summary.get('model_path', '')}`",
        f"- Policy schema: `{summary.get('schema_path', '')}`",
        f"- Security terms checked: {summary.get('security_terms_checked', 0)}",
        f"- Peer policy fields: {summary.get('peer_policy_fields', 0)}",
        f"- Observability fields: {summary.get('observability_fields', 0)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
    ]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    if not errors:
        lines += ["", "## Verdict", "", "PASS — artifacts satisfy the offline security/observability gate."]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    errors, warnings, summary = validate(args.model, args.schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(errors, warnings, summary), encoding="utf-8")
    print(json.dumps({"ok": not errors, "errors": errors, "warnings": warnings, "output": str(args.out)}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
