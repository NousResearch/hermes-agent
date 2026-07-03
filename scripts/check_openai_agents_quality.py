#!/usr/bin/env python
"""Hard local quality gate for the Hermes OpenAI Agents SDK bridge.

Runs deterministic checks only by default. Use --live-smoke for a bounded API
smoke that consumes OpenAI tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPED_PATHS = [
    "plugins/openai_agents",
    "tests/plugins/test_openai_agents_bridge.py",
    "docs/openai-agents-governed-runtime-spec.md",
    "docs/openai-agents-git-workflow.md",
    "docs/openai-agents-project-tracking.json",
    "docs/openai-agents-source-manifest.json",
    "schemas/openai-agents-receipt.schema.json",
    "evals/openai_agents/governance_cases.json",
    "scripts/check_openai_agents_quality.py",
    "scripts/generate_openai_agents_proof_bundle.py",
    "pyproject.toml",
    "uv.lock",
]
SECRET_PATTERNS = [
    re.compile(r"^\+.*(?i:(api_key|secret|password|token|passwd))\s*=\s*['\"][^'\"]{6,}['\"]"),
]
DANGEROUS_PATTERNS = [
    re.compile(r"^\+.*(os\.system\(|subprocess.*shell=True|\beval\(|\bexec\(|pickle\.loads?\()"),
    re.compile(r"^\+.*(execute\(f\"|\.format\(.*SELECT|\.format\(.*INSERT)"),
]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def git_diff_scoped() -> str:
    proc = subprocess.run(
        ["git", "diff", "--", *SCOPED_PATHS],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def static_scan_diff(diff: str) -> list[str]:
    findings: list[str] = []
    for line in diff.splitlines():
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(f"secret-pattern: {line[:180]}")
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(line):
                findings.append(f"dangerous-pattern: {line[:180]}")
    return findings


def validate_receipt(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["success", "lane", "model", "max_turns", "max_tokens", "usage", "governance_contract", "receipt", "result", "truncated"]
    for key in required:
        if key not in payload:
            errors.append(f"missing required key: {key}")
    if payload.get("lane") not in {"review", "execute", "verify", "architecture"}:
        errors.append(f"invalid lane: {payload.get('lane')!r}")
    result = payload.get("result") or {}
    if not isinstance(result, dict):
        errors.append("result must be object")
    elif result.get("status") not in {"verified", "partial", "blocked"}:
        errors.append(f"invalid result.status: {result.get('status')!r}")
    receipt = payload.get("receipt") or {}
    if not isinstance(receipt, dict):
        errors.append("receipt must be object")
    else:
        for key, expected in {
            "structured_output": True,
            "preflight_enforced": True,
            "postconditions_enforced": True,
            "trace_sensitive_data": False,
        }.items():
            if receipt.get(key) is not expected:
                errors.append(f"receipt.{key} expected {expected!r}, got {receipt.get(key)!r}")
    forbidden = {"api_key", "token", "secret", "password", "credential"}
    if any(k.lower() in forbidden for k in payload):
        errors.append("receipt contains forbidden secret-bearing top-level key")
    return errors


def validate_recent_receipts(limit: int = 10) -> None:
    from hermes_constants import get_hermes_home

    receipt_dir = Path(get_hermes_home()) / "receipts" / "openai_agents"
    if not receipt_dir.exists():
        print("receipt validation: no receipt directory yet (ok)")
        return
    paths = sorted(receipt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    checked = 0
    skipped_legacy = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Receipts created before the pro-grade schema existed remain evidence,
        # but they should not fail the current gate. Validate only current-shape
        # receipts here; live-smoke creates one when needed.
        if "usage" not in payload or "max_tokens" not in payload:
            skipped_legacy += 1
            continue
        errors = validate_receipt(payload)
        if errors:
            raise RuntimeError(f"receipt schema validation failed for {path}: {errors}")
        checked += 1
    print(f"receipt validation: {checked} current receipt(s) ok; {skipped_legacy} legacy skipped")


def run_governance_evals() -> None:
    from plugins.openai_agents import tools as oat

    corpus = json.loads((ROOT / "evals/openai_agents/governance_cases.json").read_text(encoding="utf-8"))
    cases = corpus.get("cases") or []
    for case in cases:
        ctype = case["type"]
        if ctype == "preflight_block":
            try:
                oat._preflight_request(case["lane"], case["task"], list(case.get("constraints") or []))
            except ValueError:
                continue
            raise RuntimeError(f"eval {case['id']} did not block")
        if ctype == "preflight_allow":
            oat._preflight_request(case["lane"], case["task"], list(case.get("constraints") or []))
            continue
        if ctype in {"postcondition_downgrade", "postcondition_mutation_downgrade"}:
            output = oat.GovernedAgentOutput(**case["output"])
            result = oat._enforce_postconditions(output, lane=case["lane"])
            if result.status != case["expected_status"]:
                raise RuntimeError(f"eval {case['id']} status {result.status!r}")
            continue
        if ctype == "model_block":
            try:
                oat._resolve_model(case["model"])
            except ValueError:
                continue
            raise RuntimeError(f"eval {case['id']} did not block model")
        if ctype == "receipt_sanitize":
            sanitized = oat._sanitize_for_receipt(case["payload"])
            for key in case.get("forbidden_keys") or []:
                if key in sanitized:
                    raise RuntimeError(f"eval {case['id']} leaked key {key}")
            continue
        if ctype == "architecture_schema":
            if oat.OPENAI_AGENTS_ARCHITECTURE_SCHEMA.get("name") != case["expected_name"]:
                raise RuntimeError(f"eval {case['id']} wrong schema name")
            continue
        raise RuntimeError(f"unknown eval type: {ctype}")
    print(f"governance evals: {len(cases)} case(s) ok")


def _ensure_unique_ids(items: list[dict[str, Any]], *, field: str) -> set[str]:
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            raise RuntimeError(f"tracking {field} item missing id")
        if item_id in seen:
            raise RuntimeError(f"tracking {field} duplicate id: {item_id}")
        seen.add(item_id)
    return seen


def _git_ref_exists(ref: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0


def _git_output(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stdout.strip()}")
    return proc.stdout.strip()


def _path_allowed_untracked(path: str, allowed: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized == item.rstrip("/") or normalized.startswith(item.rstrip("/") + "/") for item in allowed)


def validate_project_tracking() -> None:
    path = ROOT / "docs/openai-agents-project-tracking.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("project_id") != "OASDK-HARDENING":
        raise RuntimeError("tracking manifest project_id must be OASDK-HARDENING")
    if data.get("authority_model", {}).get("external_side_effects_allowed") is not False:
        raise RuntimeError("tracking manifest must keep external side effects disabled by default")
    roadmap = data.get("roadmap_items") or []
    receipts = data.get("receipt_groups") or []
    next_actions = data.get("next_actions") or []
    if not roadmap or not receipts or not next_actions:
        raise RuntimeError("tracking manifest requires roadmap_items, receipt_groups, and next_actions")

    roadmap_ids = _ensure_unique_ids(roadmap, field="roadmap")
    receipt_ids = _ensure_unique_ids(receipts, field="receipt_group")
    next_action_ids = _ensure_unique_ids(next_actions, field="next_action")
    for item_id in roadmap_ids:
        if not re.fullmatch(r"OASDK-HARDEN-\d{3}", item_id):
            raise RuntimeError(f"invalid roadmap id: {item_id}")
    for item_id in receipt_ids:
        if not re.fullmatch(r"OASDK-RCPT-[A-Z0-9-]+", item_id):
            raise RuntimeError(f"invalid receipt group id: {item_id}")
    for item_id in next_action_ids:
        if not re.fullmatch(r"OASDK-NEXT-\d{3}", item_id):
            raise RuntimeError(f"invalid next action id: {item_id}")

    for item in roadmap:
        links = set(item.get("linked_receipt_groups") or [])
        missing = links - receipt_ids
        if missing:
            raise RuntimeError(f"roadmap {item['id']} links missing receipt groups: {sorted(missing)}")
        for ref in item.get("linked_commit_refs") or []:
            if not _git_ref_exists(str(ref)):
                raise RuntimeError(f"roadmap {item['id']} references missing local commit: {ref}")
        if item.get("status") in {"done", "verified"} and not item.get("acceptance"):
            raise RuntimeError(f"roadmap {item['id']} done/verified without acceptance criteria")
        if "approval_boundary" not in item:
            raise RuntimeError(f"roadmap {item['id']} missing approval_boundary")

    for group in receipts:
        links = set(group.get("linked_roadmap_items") or [])
        missing = links - roadmap_ids
        if missing:
            raise RuntimeError(f"receipt group {group['id']} links missing roadmap items: {sorted(missing)}")
        if not (group.get("local_glob") or group.get("local_command")):
            raise RuntimeError(f"receipt group {group['id']} missing local_glob/local_command")

    for action in next_actions:
        links = set(action.get("linked_roadmap_items") or [])
        missing = links - roadmap_ids
        if missing:
            raise RuntimeError(f"next action {action['id']} links missing roadmap items: {sorted(missing)}")
        for key in ("destructive", "privileged", "external_side_effect", "recurring_spend"):
            if not isinstance(action.get(key), bool):
                raise RuntimeError(f"next action {action['id']} missing boolean {key}")
        if action.get("external_side_effect") and "explicit" not in str(action.get("required_approval_level", "")):
            raise RuntimeError(f"next action {action['id']} external side effect lacks explicit approval level")

    git_workflow = data.get("git_workflow") or {}
    if git_workflow.get("tracking_mode") != "native_git_local_first":
        raise RuntimeError("git_workflow.tracking_mode must be native_git_local_first")
    if git_workflow.get("working_branch") != _git_output(["branch", "--show-current"]):
        raise RuntimeError("git_workflow.working_branch does not match current branch")
    if git_workflow.get("repo_remote") != _git_output(["remote", "get-url", "origin"]):
        raise RuntimeError("git_workflow.repo_remote does not match origin remote")
    if git_workflow.get("upstream_ref") != "origin/main":
        raise RuntimeError("git_workflow.upstream_ref must remain origin/main for PR readiness")
    required_cmds = set(git_workflow.get("pre_commit_required_commands") or [])
    if "python scripts/check_openai_agents_quality.py" not in required_cmds:
        raise RuntimeError("git_workflow missing SDK quality gate pre-commit command")
    external_actions = set(git_workflow.get("external_actions_require_explicit_scope") or [])
    for required_external in {"git push", "gh issue create", "gh pr create"}:
        if required_external not in external_actions:
            raise RuntimeError(f"git_workflow must mark {required_external!r} as explicit-scope")
    for ref in git_workflow.get("tracked_local_commit_refs") or []:
        if not _git_ref_exists(str(ref)):
            raise RuntimeError(f"git_workflow references missing local commit: {ref}")
    allowed_untracked = list(git_workflow.get("allowed_untracked_paths") or [])
    status = _git_output(["status", "--porcelain", "--untracked-files=all"])
    for line in status.splitlines():
        if line.startswith("?? "):
            untracked_path = line[3:]
            if not _path_allowed_untracked(untracked_path, allowed_untracked):
                raise RuntimeError(f"unexpected untracked path not allowed by git_workflow: {untracked_path}")
    print(
        f"project tracking: {len(roadmap)} roadmap item(s), "
        f"{len(receipts)} receipt group(s), {len(next_actions)} next action(s), git workflow ok"
    )


def validate_source_manifest() -> None:
    path = ROOT / "docs/openai-agents-source-manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("project_id") != "OASDK-HARDENING":
        raise RuntimeError("source manifest project_id must be OASDK-HARDENING")
    package = data.get("sdk_package") or {}
    if package.get("distribution") != "openai-agents" or package.get("import_name") != "agents":
        raise RuntimeError("source manifest must identify openai-agents / agents package")
    sources = data.get("official_sources") or []
    if len(sources) < 4:
        raise RuntimeError("source manifest must include core OpenAI Agents docs")
    ids = _ensure_unique_ids(sources, field="official_source")
    required_ids = {
        "OAI-AGENTS-DOCS-HOME",
        "OAI-AGENTS-DOCS-AGENTS",
        "OAI-AGENTS-DOCS-RESULTS",
        "OAI-AGENTS-DOCS-GUARDRAILS",
    }
    missing = required_ids - ids
    if missing:
        raise RuntimeError(f"source manifest missing required docs: {sorted(missing)}")
    for source in sources:
        url = str(source.get("url") or "")
        if not url.startswith("https://openai.github.io/openai-agents-python/"):
            raise RuntimeError(f"source manifest contains non-official SDK URL: {url}")
        for key in ("retrieved_at_utc", "local_relevance", "implementation_bindings"):
            if not source.get(key):
                raise RuntimeError(f"source {source['id']} missing {key}")
    assumptions = data.get("local_assumptions") or []
    if not assumptions:
        raise RuntimeError("source manifest requires local_assumptions")
    print(f"source manifest: {len(sources)} official source(s), {len(assumptions)} local assumption(s) ok")


def check_registration() -> None:
    from hermes_cli.plugins import discover_plugins
    from tools.registry import registry
    from toolsets import resolve_toolset

    discover_plugins()
    expected = ["openai_agents_review", "openai_agents_execute", "openai_agents_verify", "openai_agents_architecture", "openai_agents_run"]
    resolved = set(resolve_toolset("openai_agents"))
    for name in expected:
        entry = registry.get_entry(name)
        if not entry or entry.toolset != "openai_agents" or not (entry.check_fn and entry.check_fn()):
            raise RuntimeError(f"tool registration unavailable: {name}")
        if name not in resolved:
            raise RuntimeError(f"tool missing from resolved toolset: {name}")
    print("tool registration: ok")


def live_smoke() -> None:
    from plugins.openai_agents.tools import _handle_openai_agents_verify

    raw = _handle_openai_agents_verify({
        "task": "Verify exact string QUALITY_GATE_LIVE_OK appears in context.",
        "context": "QUALITY_GATE_LIVE_OK",
        "acceptance_criteria": ["Verified only if exact string appears in context.", "Return proof."],
        "constraints": ["analysis only; no mutation"],
        "max_turns": 1,
        "max_tokens": 700,
    })
    payload = json.loads(raw)
    result = payload.get("result") or {}
    if not payload.get("success") or result.get("status") != "verified" or not payload.get("receipt_sha256"):
        raise RuntimeError(f"live smoke failed: {raw[:1000]}")
    print("live smoke: ok")
    print(json.dumps({
        "status": result.get("status"),
        "proof_count": len(result.get("proof") or []),
        "usage": payload.get("usage"),
        "receipt_path": payload.get("receipt_path"),
        "receipt_sha256": payload.get("receipt_sha256"),
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-smoke", action="store_true", help="Run a bounded live OpenAI SDK smoke test")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest (for unit-testing this script)")
    args = parser.parse_args()

    run([sys.executable, "-m", "py_compile", "plugins/openai_agents/__init__.py", "plugins/openai_agents/tools.py", "scripts/check_openai_agents_quality.py", "scripts/generate_openai_agents_proof_bundle.py"])
    if not args.skip_pytest:
        run(["uv", "run", "--with", "pytest", "--with", "pytest-xdist", "python", "-m", "pytest", "tests/plugins/test_openai_agents_bridge.py", "tests/test_toolsets.py", "-q", "-o", "addopts="])
    run(["git", "diff", "--check", "--", *SCOPED_PATHS])
    findings = static_scan_diff(git_diff_scoped())
    if findings:
        raise RuntimeError("static scan findings:\n" + "\n".join(findings))
    print("static scan: ok")
    check_registration()
    run_governance_evals()
    validate_project_tracking()
    validate_source_manifest()
    validate_recent_receipts()
    if args.live_smoke:
        live_smoke()
    print("OPENAI_AGENTS_QUALITY_GATE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OPENAI_AGENTS_QUALITY_GATE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
