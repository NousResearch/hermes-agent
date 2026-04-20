from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from analyze_telemetry import load_events, split_events, latest_feedback_by_request_id, join_decisions_with_feedback
from suggest_router_changes_v2 import segmented_suggestions


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def make_override_name(segment: dict[str, Any]) -> str:
    parts = []
    for key in ("task_type", "priority", "quota", "privacy", "mode", "primary_model"):
        value = segment.get(key)
        if value:
            parts.append(f"{key}-{value}")
    return "auto-" + "-".join(parts[:4])


def choose_replacement_model(segment: dict[str, Any]) -> str | None:
    primary = segment.get("primary_model")
    task_type = segment.get("task_type")

    if task_type in {"chat", "writing", "research"}:
        return "claude-sonnet-4.6"
    if task_type == "coding":
        return "gpt-5.4"
    if task_type == "batch" and segment.get("privacy") in {"sensitive", "local_only"}:
        return "ollama"
    if primary == "deepseek":
        if task_type == "coding":
            return "gpt-5.4"
        return "claude-sonnet-4.6"
    return None


def suggestion_to_override(suggestion: dict[str, Any]) -> dict[str, Any] | None:
    segment = suggestion.get("segment", {})
    if not segment:
        return None

    target_model = choose_replacement_model(segment)
    if not target_model:
        return None

    when = {}
    for key in ("task_type", "priority", "quota", "privacy", "mode"):
        value = segment.get(key)
        if value:
            when[key] = value

    if len(when) < 2:
        return None

    if "task_type" not in when and "quota" not in when:
        return None

    return {
        "name": make_override_name(segment),
        "when": when,
        "force": target_model,
        "reason": suggestion["message"],
    }


def _override_signature(override: dict[str, Any]) -> str:
    return json.dumps(override.get("when", {}), sort_keys=True, ensure_ascii=False) + "::" + str(override.get("force"))


def build_patch_proposal(config: dict[str, Any], suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    proposed = copy.deepcopy(config)
    overrides = proposed.setdefault("policy_overrides", [])

    generated = []
    seen = {_override_signature(item) for item in overrides if isinstance(item, dict)}

    for suggestion in suggestions:
        if suggestion.get("severity") not in {"high", "medium"}:
            continue

        override = suggestion_to_override(suggestion)
        if not override:
            continue

        signature = _override_signature(override)
        if signature in seen:
            continue

        seen.add(signature)
        overrides.append(override)
        generated.append(override)

    return {
        "generated_count": len(generated),
        "generated_overrides": generated,
        "proposed_config": proposed,
    }


def format_report(result: dict[str, Any]) -> str:
    lines = ["== Proposed Router Config Patch =="]
    lines.append(f"Generated overrides: {result['generated_count']}")

    if not result["generated_overrides"]:
        lines.append("")
        lines.append("לא נוצרו overrides חדשים.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Proposed overrides:")
    for item in result["generated_overrides"]:
        lines.append(f"- name: {item['name']}")
        lines.append(f"  when: {item['when']}")
        lines.append(f"  force: {item['force']}")
        lines.append(f"  reason: {item['reason']}")
        lines.append("")

    lines.append("YAML snippet:")
    lines.append(
        yaml.safe_dump(
            {"policy_overrides": result["generated_overrides"]},
            allow_unicode=True,
            sort_keys=False,
        ).rstrip()
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propose router_config.yaml patch from telemetry")
    parser.add_argument("log_path", help="Path to telemetry JSONL log")
    parser.add_argument("--config", required=True, help="Path to router_config.yaml")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    parser.add_argument("--full-config", action="store_true", help="Print full proposed config YAML")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_yaml(args.config)
    events = load_events(Path(args.log_path))
    decisions, feedbacks = split_events(events)
    latest_feedback = latest_feedback_by_request_id(feedbacks)
    joined = join_decisions_with_feedback(decisions, latest_feedback)

    suggestions = segmented_suggestions(joined, min_feedback_samples=3)
    result = build_patch_proposal(config, suggestions)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.full_config:
        print(dump_yaml(result["proposed_config"]))
        return 0

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
