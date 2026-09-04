#!/usr/bin/env python3
"""Apply and verify the current Hermes benchmark routing policy.

This is intentionally a state-home migration tool, not a benchmark runner.
The benchmark receipts remain historical diagnostic evidence; this script
materialises the operator's later Terra -> Luna promotion and records the
result in ``~/.hermes/benchmark_policy.json`` so an app rebuild does not
silently return to unqualified defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


RUN_ID = "hrl-benchmark-v3-20260831"
VAULT_NOTE = "/Users/mikedemott/LunaBotVault/Investigations/Hermes provider benchmark and routing tune 2026-08-31.md"
ROOT_MANIFEST = "benchmark_profile.json"
POLICY_FILENAME = "benchmark_policy.json"

ROUTE_CAPS = {
    ("openrouter", "thinkingmachines/inkling-small:free", "none"): 4,
    ("openai-codex", "gpt-5.6-luna", "low"): 4,
    ("openai-codex", "gpt-5.5", "low"): 4,
    ("openai-codex", "gpt-5.6-luna", "high"): 2,
    ("openai-codex", "gpt-5.6-sol", "high"): 2,
    ("nous", "tencent/hy3:free", "none"): 2,
    ("ollama-launch", "qwen3.5:4b", "none"): 1,
}

# These profiles were added after the 89-profile benchmark inventory. They
# receive a conservative, role-derived route but remain explicitly marked as
# inherited rather than pretending to have their own benchmark receipt.
INHERITED_ROUTES = {
    "hermesdevelopment": ("openai-codex", "gpt-5.6-luna", "high"),
    "revenue-artifact-auditor": ("openai-codex", "gpt-5.6-sol", "high"),
    "revenue-compliance-guardian": ("openai-codex", "gpt-5.6-sol", "high"),
    "revenue-lab-steward": ("openai-codex", "gpt-5.6-luna", "low"),
    "revenue-outcome-analyst": ("openai-codex", "gpt-5.5", "low"),
    "revenue-routing-analyst": ("openai-codex", "gpt-5.6-sol", "high"),
    "revenue-scout": ("nous", "tencent/hy3:free", "none"),
}

# This is the active auxiliary policy already used by the current root
# config, made explicit for every profile that has the corresponding task.
# Utility/intake work stays on the small local model; review remains Luna-high
# because the operator promoted the former Terra-medium review route.
AUXILIARY_POLICY = {
    "vision": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "compression": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "skills_hub": ("openai-codex", "gpt-5.5", "low", 4),
    "approval": ("openai-codex", "gpt-5.5", "low", 4),
    "review": ("openai-codex", "gpt-5.6-luna", "high", 2),
    "mcp": ("openai-codex", "gpt-5.5", "low", 4),
    "title_generation": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "memory_query_rewrite": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "tts_audio_tags": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "triage_specifier": ("openai-codex", "gpt-5.6-luna", "low", 4),
    "kanban_decomposer": ("openai-codex", "gpt-5.6-luna", "low", 1),
    "profile_describer": ("openrouter", "thinkingmachines/inkling-small:free", "none", 1),
    "goal_judge": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "curator": ("openrouter", "thinkingmachines/inkling-small:free", "none", 1),
    "monitor": ("ollama-launch", "qwen3.5:4b", "none", 1),
    "background_review": ("openai-codex", "gpt-5.6-luna", "high", 2),
    "moa_reference": ("nous", "tencent/hy3:free", "none", 2),
    "moa_aggregator": ("openai-codex", "gpt-5.6-luna", "low", 4),
    "specialist_router": ("openrouter", "thinkingmachines/inkling-small:free", "none", 1),
}


def _scalar_line(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    prefix = " " * indent
    needle = prefix + key + ":"
    for idx in range(start, end):
        if lines[idx].startswith(needle):
            remainder = lines[idx][len(needle):]
            if remainder and remainder[0] not in " \t\r\n":
                continue
            return idx
    return None


def _block_end(lines: list[str], start: int, child_indent: int) -> int:
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip(" "))
            if indent < child_indent:
                return idx
    return len(lines)


def _set_scalar(lines: list[str], start: int, end: int, indent: int, key: str, value: Any, *, insert_after: int) -> None:
    idx = _scalar_line(lines, start, end, indent, key)
    # PyYAML emits a document terminator for some bare scalar dumps. JSON
    # quoting is valid YAML and keeps this line-oriented edit single-line.
    rendered_value = json.dumps(value) if isinstance(value, str) else str(value)
    rendered = f"{' ' * indent}{key}: {rendered_value}\n"
    if idx is None:
        lines.insert(insert_after + 1, rendered)
    else:
        lines[idx] = rendered


def _rewrite_config(path: Path, route: tuple[str, str, str]) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    model = _scalar_line(lines, 0, len(lines), 0, "model")
    if model is None:
        raise ValueError(f"{path}: missing model mapping")
    model_end = _block_end(lines, model, 2)
    for key, value in zip(("provider", "default", "reasoning_effort"), route):
        _set_scalar(lines, model + 1, model_end, 2, key, value, insert_after=model)
        model_end = _block_end(lines, model, 2)

    agent = _scalar_line(lines, 0, len(lines), 0, "agent")
    if agent is not None:
        agent_end = _block_end(lines, agent, 2)
        _set_scalar(lines, agent + 1, agent_end, 2, "reasoning_effort", route[2], insert_after=agent)

    auxiliary = _scalar_line(lines, 0, len(lines), 0, "auxiliary")
    if auxiliary is not None:
        aux_end = _block_end(lines, auxiliary, 2)
        for task, (provider, model_name, effort, cap) in AUXILIARY_POLICY.items():
            task_idx = _scalar_line(lines, auxiliary + 1, aux_end, 2, task)
            if task_idx is None:
                continue
            task_end = _block_end(lines, task_idx, 4)
            for key, value in (("provider", provider), ("model", model_name), ("reasoning_effort", effort), ("max_concurrency", cap)):
                _set_scalar(lines, task_idx + 1, task_end, 4, key, value, insert_after=task_idx)
                task_end = _block_end(lines, task_idx, 4)
                aux_end = _block_end(lines, auxiliary, 2)
            # Re-resolve the next task after possible insertions.
            aux_end = _block_end(lines, auxiliary, 2)

    updated = "".join(lines)
    yaml.safe_load(updated)  # fail closed before writing
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _config_route(path: Path) -> tuple[str, str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = data.get("model") or {}
    agent = data.get("agent") or {}
    if not isinstance(model, dict):
        raise ValueError(f"{path}: model is not a mapping")
    provider = model.get("provider")
    model_name = model.get("default") or model.get("model")
    effort = model.get("reasoning_effort")
    if effort is None:
        effort = agent.get("reasoning_effort")
    return provider, model_name, effort


def _digest_profile_set(routes: dict[str, tuple[str, str, str]]) -> str:
    payload = [
        {"profile_id": name, "provider": r[0], "model": r[1], "reasoning_effort": r[2]}
        for name, r in sorted(routes.items())
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-home", type=Path, default=Path.home() / ".hermes")
    parser.add_argument("--backup-path", type=Path)
    parser.add_argument("--apply", action="store_true", help="write the reconciled configs")
    args = parser.parse_args()
    home = args.hermes_home
    profiles_root = home / "profiles"
    receipt_routes: dict[str, tuple[str, str, str]] = {}
    for receipt in sorted(profiles_root.glob("*/benchmark_profile.json")):
        data = json.loads(receipt.read_text(encoding="utf-8"))
        primary = data.get("route", {}).get("primary", {})
        route = (primary.get("provider"), primary.get("model"), primary.get("reasoning_effort"))
        if route == ("openai-codex", "gpt-5.6-terra", "medium"):
            route = ("openai-codex", "gpt-5.6-luna", "high")
        if route not in ROUTE_CAPS:
            raise ValueError(f"{receipt}: unqualified benchmark route {route!r}")
        receipt_routes[receipt.parent.name] = route

    profile_dirs = sorted(p for p in profiles_root.iterdir() if p.is_dir() and (p / "config.yaml").is_file())
    all_routes = dict(receipt_routes)
    for name, route in INHERITED_ROUTES.items():
        if name in all_routes and all_routes[name] != route:
            raise ValueError(f"{name}: receipt and inherited route disagree")
        all_routes[name] = route
    inherited_names = set(INHERITED_ROUTES) - set(receipt_routes)
    actual_names = {p.name for p in profile_dirs}
    if actual_names != set(all_routes):
        raise ValueError(f"profile inventory mismatch: configs={len(actual_names)} policy={len(all_routes)}")

    root_config = home / "config.yaml"
    changed: list[str] = []
    if args.apply:
        if _rewrite_config(root_config, ("openai-codex", "gpt-5.5", "low")):
            changed.append(str(root_config))
        for profile_dir in profile_dirs:
            if _rewrite_config(profile_dir / "config.yaml", all_routes[profile_dir.name]):
                changed.append(str(profile_dir / "config.yaml"))

    readback = {p.name: _config_route(p / "config.yaml") for p in profile_dirs}
    mismatches = {name: (all_routes[name], readback[name]) for name in sorted(all_routes) if all_routes[name] != readback[name]}
    if mismatches and not args.apply:
        print(json.dumps({"dry_run_mismatches": mismatches}, indent=2, sort_keys=True))
        return 0
    if mismatches:
        raise ValueError(f"readback mismatch: {json.dumps(mismatches, sort_keys=True)}")

    soul_hashes: dict[str, str] = {}
    for profile_dir in profile_dirs:
        soul_path = profile_dir / "SOUL.md"
        soul = soul_path.read_text(encoding="utf-8").strip() if soul_path.is_file() else ""
        if len(soul) < 160:
            raise ValueError(f"{soul_path}: missing or non-dedicated soul")
        soul_hashes[profile_dir.name] = hashlib.sha256(soul.encode("utf-8")).hexdigest()

    aux_readback: dict[str, dict[str, Any]] = {}
    root_data = yaml.safe_load(root_config.read_text(encoding="utf-8")) or {}
    for task, expected in AUXILIARY_POLICY.items():
        item = (root_data.get("auxiliary") or {}).get(task)
        if isinstance(item, dict):
            aux_readback[task] = {"provider": item.get("provider"), "model": item.get("model"), "reasoning_effort": item.get("reasoning_effort"), "max_concurrency": item.get("max_concurrency")}

    policy = {
        "schema_name": "hermes_benchmark_policy_v1",
        "benchmark": {
            "run_id": RUN_ID,
            "classification": "diagnostic-only",
            "authority_ceiling": "no_runtime_or_acceptance_authority",
            "source_note": VAULT_NOTE,
            "root_manifest": str(home / ROOT_MANIFEST),
            "verified_at": "2026-08-31",
            "operator_amendment": "2026-09-02 Terra-medium active routes promoted to Luna-high",
        },
        "durability": {
            "state_home": str(home),
            "policy_path": str(home / POLICY_FILENAME),
            "rebuild_rule": "installer and profile updates preserve existing config.yaml; this policy is state-home data, not a build artifact",
            "backup_path": str(args.backup_path) if args.backup_path else None,
        },
        "profiles": {
            "total": len(all_routes),
            "benchmark_receipt_backed": len(receipt_routes),
            "inherited_role_routes": len(inherited_names),
            "profile_set_sha256": _digest_profile_set(all_routes),
            "routes": {
                name: {"provider": route[0], "model": route[1], "reasoning_effort": route[2], "concurrency_cap": ROUTE_CAPS[route], "qualified_by_benchmark": name in receipt_routes}
                for name, route in sorted(all_routes.items())
            },
            "distribution": {
                f"{provider}/{model}|{effort}": count
                for (provider, model, effort), count in sorted(Counter(all_routes.values()).items())
            },
        },
        "auxiliary_policy": {
            task: {"provider": p, "model": m, "reasoning_effort": e, "max_concurrency": cap}
            for task, (p, m, e, cap) in sorted(AUXILIARY_POLICY.items())
        },
        "souls": {
            "scope": "named profiles; default keeps the global umbrella SOUL.md",
            "profile_count": len(soul_hashes),
            "unique_hashes": len(set(soul_hashes.values())),
            "all_unique": len(soul_hashes) == len(set(soul_hashes.values())),
            "sha256": dict(sorted(soul_hashes.items())),
            "default_sha256": hashlib.sha256((home / "SOUL.md").read_bytes()).hexdigest()
            if (home / "SOUL.md").is_file()
            else None,
        },
        "readback": {
            "config_sha256": hashlib.sha256(root_config.read_bytes()).hexdigest(),
            "changed_files": changed,
            "auxiliary_root": aux_readback,
        },
    }
    if args.apply:
        (home / POLICY_FILENAME).write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"apply": args.apply, "profiles": len(all_routes), "receipt_backed": len(receipt_routes), "changed": len(changed), "policy_path": str(home / POLICY_FILENAME), "profile_set_sha256": policy["profiles"]["profile_set_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
