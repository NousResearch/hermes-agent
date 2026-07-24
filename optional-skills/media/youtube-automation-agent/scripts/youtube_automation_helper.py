#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "package.json",
    "index.js",
    "setup.js",
    "test.js",
    "schedules/daily-automation.js",
    "utils/credential-manager.js",
]

OPTIONAL_CONFIG_FILES = [
    ".env",
    "config/credentials.json",
    "config/tokens.json",
]

ENDPOINTS = ["/health", "/schedule", "/analytics"]
STAGES = ["strategy", "script", "thumbnail", "seo", "production", "publishing", "analytics"]


@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return value or "youtube-run"


def _default_runs_dir() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return hermes_home / "skills" / "media" / "youtube-automation-agent" / "data" / "runs"


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(path: Path, label: str) -> CheckResult:
    return CheckResult(label=label, ok=path.exists(), detail=str(path))


def _load_package_json(repo: Path) -> dict[str, Any]:
    pkg_path = repo / "package.json"
    return json.loads(pkg_path.read_text(encoding="utf-8"))


def inspect_repo(repo: Path) -> dict[str, Any]:
    results: list[CheckResult] = []
    for rel in REQUIRED_FILES:
        results.append(_check(repo / rel, f"required:{rel}"))

    for rel in OPTIONAL_CONFIG_FILES:
        results.append(_check(repo / rel, f"config:{rel}"))

    node_path = shutil.which("node")
    results.append(CheckResult("command:node", node_path is not None, node_path or "node not found"))
    results.append(_check(repo / "node_modules", "deps:node_modules"))

    package_json = _load_package_json(repo)
    missing_script_targets: list[dict[str, str]] = []
    for name, command in package_json.get("scripts", {}).items():
        parts = command.split()
        if len(parts) >= 2 and parts[0] == "node":
            target = repo / parts[1]
            if not target.exists():
                missing_script_targets.append({"script": name, "target": parts[1]})

    syntax_checks: list[dict[str, Any]] = []
    if node_path:
        for rel in ["index.js", "setup.js", "test.js"]:
            target = repo / rel
            if not target.exists():
                continue
            proc = subprocess.run(
                [node_path, "--check", str(target)],
                text=True,
                capture_output=True,
                cwd=repo,
            )
            syntax_checks.append(
                {
                    "file": rel,
                    "ok": proc.returncode == 0,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )

    required_missing = [r.label for r in results if r.label.startswith("required:") and not r.ok]
    config_missing = [r.label for r in results if r.label.startswith("config:") and not r.ok]
    verdict = "ready"
    if required_missing or missing_script_targets:
        verdict = "blocked"
    elif config_missing or not (repo / "node_modules").exists():
        verdict = "needs-setup"

    return {
        "repo": str(repo),
        "verdict": verdict,
        "checks": [r.__dict__ for r in results],
        "missing_script_targets": missing_script_targets,
        "syntax_checks": syntax_checks,
    }


def _fetch_json(url: str, timeout: int = 5) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-youtube-automation-helper/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return {"status": resp.status, "body": body}


def probe_server(base_url: str) -> dict[str, Any]:
    endpoints: list[dict[str, Any]] = []
    healthy = True
    for endpoint in ENDPOINTS:
        url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            result = _fetch_json(url)
            status = int(result["status"])
            body = result["body"]
            endpoint_ok = 200 <= status < 300
            if endpoint == "/health":
                try:
                    payload = json.loads(body)
                    endpoint_ok = endpoint_ok and payload.get("status") == "healthy"
                except json.JSONDecodeError:
                    endpoint_ok = False
            healthy = healthy and endpoint_ok
            endpoints.append({
                "endpoint": endpoint,
                "ok": endpoint_ok,
                "status": status,
                "body_preview": body[:400],
            })
        except urllib.error.URLError as exc:
            healthy = False
            endpoints.append({
                "endpoint": endpoint,
                "ok": False,
                "status": None,
                "error": str(exc),
            })
    return {"base_url": base_url, "healthy": healthy, "endpoints": endpoints}


def init_run(channel: str, niche: str, audience: str, style: str, frequency: str, topic: str | None, repo: str | None, output: str | None) -> dict[str, Any]:
    title_basis = topic or f"{channel}-{niche}"
    slug = _slugify(title_basis)
    output_path = Path(output).expanduser().resolve() if output else _default_runs_dir() / f"{slug}.json"
    run = {
        "version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "current_stage": STAGES[0],
        "completed_stages": [],
        "run": {
            "channel": channel,
            "niche": niche,
            "audience": audience,
            "style": style,
            "frequency": frequency,
            "topic": topic,
            "repo": repo,
        },
        "stages": {
            stage: {
                "status": "pending",
                "notes": "",
                "artifacts": {},
                "completed_at": None,
            }
            for stage in STAGES
        },
    }
    run["stages"][STAGES[0]]["status"] = "in_progress"
    _save_json(output_path, run)
    return {"workspace": str(output_path), "run": run}


def _ensure_workspace(path: str) -> Path:
    workspace = Path(path).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    return workspace


def get_status(workspace: Path) -> dict[str, Any]:
    payload = _load_json(workspace)
    summary = {
        "workspace": str(workspace),
        "current_stage": payload["current_stage"],
        "completed_stages": payload["completed_stages"],
        "pending_stages": [s for s in STAGES if s not in payload["completed_stages"]],
        "run": payload["run"],
        "stages": payload["stages"],
    }
    return summary


def _stage_brief(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    run = payload["run"]
    base_context = {
        "channel": run["channel"],
        "niche": run["niche"],
        "audience": run["audience"],
        "style": run["style"],
        "frequency": run["frequency"],
        "topic": run.get("topic"),
    }
    prompts = {
        "strategy": {
            "goal": "Find a strong video angle based on the niche, audience, and channel direction.",
            "prompt": f"Research a YouTube content strategy for channel '{run['channel']}' in niche '{run['niche']}' targeting '{run['audience']}'. Produce topic ideas, one selected topic, angle, content type, target keywords, competitor cues, and best publish timing.",
            "expected_artifacts": ["topic_candidates", "selected_topic", "angle", "content_type", "keywords", "best_publish_time"],
        },
        "script": {
            "goal": "Turn the selected strategy into a watch-time-focused video script.",
            "prompt": "Using the completed strategy stage, write a script package with title, hook, introduction, main sections, CTA, estimated duration, and a narration-ready full script.",
            "expected_artifacts": ["title", "hook", "outline", "full_script", "duration_estimate"],
        },
        "thumbnail": {
            "goal": "Design the thumbnail concept that matches the script and maximizes CTR.",
            "prompt": "Using the script stage, create a thumbnail brief with primary text, secondary text, emotional angle, composition, color palette, and visual elements. Include 2-3 alternate concepts.",
            "expected_artifacts": ["primary_text", "secondary_text", "concept", "color_palette", "alternatives"],
        },
        "seo": {
            "goal": "Package the video for YouTube discovery.",
            "prompt": "Using the script and strategy stages, create an SEO package with optimized title, description, tags, hashtags, chapters, and category recommendation.",
            "expected_artifacts": ["seo_title", "description", "tags", "hashtags", "chapters", "category"],
        },
        "production": {
            "goal": "Prepare the production assembly plan mirroring the upstream production agent.",
            "prompt": "Create a production plan for this video: narration asset, visual assets, thumbnail source, caption plan, editing sequence, and final assembly checklist. Call out what can be automated vs what requires manual work.",
            "expected_artifacts": ["asset_list", "narration_plan", "visual_plan", "caption_plan", "assembly_checklist"],
        },
        "publishing": {
            "goal": "Prepare upload and scheduling details for YouTube publishing.",
            "prompt": "Create a publishing package with privacy status, publish time, upload checklist, metadata mapping, thumbnail/captions attachments, and final pre-publish QA steps.",
            "expected_artifacts": ["publish_time", "privacy_status", "upload_checklist", "metadata_map", "qa_steps"],
        },
        "analytics": {
            "goal": "Create the post-publish analytics review loop.",
            "prompt": "Create a 7-day and 30-day analytics review plan with KPIs, thresholds, thumbnail/title iteration triggers, and next-video feedback loop.",
            "expected_artifacts": ["kpis", "thresholds", "review_schedule", "iteration_triggers", "next_video_feedback"],
        },
    }
    return {
        "stage": stage,
        "context": base_context,
        "stage_state": payload["stages"][stage],
        **prompts[stage],
    }


def get_brief(workspace: Path, stage: str | None) -> dict[str, Any]:
    payload = _load_json(workspace)
    selected_stage = stage or payload["current_stage"]
    if selected_stage not in STAGES:
        raise ValueError(f"Unknown stage: {selected_stage}")
    return {
        "workspace": str(workspace),
        "brief": _stage_brief(selected_stage, payload),
    }


def complete_stage(workspace: Path, stage: str, notes: str, artifacts: dict[str, Any] | None) -> dict[str, Any]:
    payload = _load_json(workspace)
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    entry = payload["stages"][stage]
    entry["status"] = "completed"
    entry["notes"] = notes
    entry["artifacts"] = artifacts or {}
    entry["completed_at"] = _now()
    if stage not in payload["completed_stages"]:
        payload["completed_stages"].append(stage)

    next_stage = None
    for candidate in STAGES:
        if candidate not in payload["completed_stages"]:
            next_stage = candidate
            break

    payload["current_stage"] = next_stage or stage
    if next_stage:
        payload["stages"][next_stage]["status"] = "in_progress"
    payload["updated_at"] = _now()
    _save_json(workspace, payload)
    return {
        "workspace": str(workspace),
        "completed_stage": stage,
        "next_stage": next_stage,
        "status": get_status(workspace),
    }


def export_run(workspace: Path) -> dict[str, Any]:
    payload = _load_json(workspace)
    completed = payload["completed_stages"]
    return {
        "workspace": str(workspace),
        "channel": payload["run"]["channel"],
        "niche": payload["run"]["niche"],
        "current_stage": payload["current_stage"],
        "completed_stages": completed,
        "deliverables": {stage: payload["stages"][stage] for stage in completed},
    }


def _print_inspect(report: dict[str, Any]) -> None:
    print(f"Repo: {report['repo']}")
    print(f"Verdict: {report['verdict']}")
    print()
    print("Checks:")
    for item in report["checks"]:
        status = "PASS" if item["ok"] else "WARN"
        print(f"- [{status}] {item['label']} :: {item['detail']}")
    if report["missing_script_targets"]:
        print()
        print("Missing package script targets:")
        for item in report["missing_script_targets"]:
            print(f"- {item['script']} -> {item['target']}")
    if report["syntax_checks"]:
        print()
        print("Node syntax checks:")
        for item in report["syntax_checks"]:
            status = "PASS" if item["ok"] else "FAIL"
            detail = item["stderr"] or item["stdout"] or "ok"
            print(f"- [{status}] {item['file']} :: {detail}")


def _print_probe(report: dict[str, Any]) -> None:
    print(f"Base URL: {report['base_url']}")
    print(f"Healthy: {report['healthy']}")
    print()
    for item in report["endpoints"]:
        status = "PASS" if item["ok"] else "FAIL"
        line = f"- [{status}] {item['endpoint']}"
        if item.get("status") is not None:
            line += f" status={item['status']}"
        if item.get("error"):
            line += f" error={item['error']}"
        print(line)
        if item.get("body_preview"):
            print(f"  preview: {item['body_preview']}")


def _print_status(report: dict[str, Any]) -> None:
    print(f"Workspace: {report['workspace']}")
    print(f"Current stage: {report['current_stage']}")
    print(f"Completed: {', '.join(report['completed_stages']) or '(none)'}")
    print()
    for stage in STAGES:
        entry = report['stages'][stage]
        print(f"- {stage}: {entry['status']}")


def _print_brief(report: dict[str, Any]) -> None:
    brief = report['brief']
    print(f"Workspace: {report['workspace']}")
    print(f"Stage: {brief['stage']}")
    print(f"Goal: {brief['goal']}")
    print()
    print("Context:")
    for key, value in brief['context'].items():
        print(f"- {key}: {value}")
    print()
    print("Prompt:")
    print(brief['prompt'])
    print()
    print("Expected artifacts:")
    for item in brief['expected_artifacts']:
        print(f"- {item}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper for the youtube-automation-agent Hermes skill")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Inspect a local clone of the repo")
    inspect_parser.add_argument("--repo", required=True, help="Path to the local repository clone")
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    probe_parser = sub.add_parser("probe", help="Probe a running local server")
    probe_parser.add_argument("--base-url", default="http://localhost:3456", help="Base URL to probe")
    probe_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    init_parser = sub.add_parser("init-run", help="Create a Hermes-native interactive YouTube workflow run")
    init_parser.add_argument("--channel", required=True)
    init_parser.add_argument("--niche", required=True)
    init_parser.add_argument("--audience", required=True)
    init_parser.add_argument("--style", required=True)
    init_parser.add_argument("--frequency", default="daily")
    init_parser.add_argument("--topic")
    init_parser.add_argument("--repo")
    init_parser.add_argument("--output")
    init_parser.add_argument("--json", action="store_true")

    status_parser = sub.add_parser("status", help="Show workflow status")
    status_parser.add_argument("--workspace", required=True)
    status_parser.add_argument("--json", action="store_true")

    brief_parser = sub.add_parser("brief", help="Show the current or requested stage brief")
    brief_parser.add_argument("--workspace", required=True)
    brief_parser.add_argument("--stage", choices=STAGES)
    brief_parser.add_argument("--json", action="store_true")

    complete_parser = sub.add_parser("complete-stage", help="Mark a stage complete and save artifacts")
    complete_parser.add_argument("--workspace", required=True)
    complete_parser.add_argument("--stage", required=True, choices=STAGES)
    complete_parser.add_argument("--notes", required=True)
    complete_parser.add_argument("--artifacts-json")
    complete_parser.add_argument("--json", action="store_true")

    export_parser = sub.add_parser("export", help="Export all completed deliverables")
    export_parser.add_argument("--workspace", required=True)
    export_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        repo = Path(args.repo).expanduser().resolve()
        if not repo.exists():
            print(f"Repo path does not exist: {repo}", file=sys.stderr)
            return 2
        report = inspect_repo(repo)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_inspect(report)
        return 0 if report["verdict"] != "blocked" else 1

    if args.command == "probe":
        report = probe_server(args.base_url)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_probe(report)
        return 0 if report["healthy"] else 1

    if args.command == "init-run":
        result = init_run(args.channel, args.niche, args.audience, args.style, args.frequency, args.topic, args.repo, args.output)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Workspace: {result['workspace']}")
            print(f"Current stage: {result['run']['current_stage']}")
        return 0

    if args.command == "status":
        workspace = _ensure_workspace(args.workspace)
        result = get_status(workspace)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_status(result)
        return 0

    if args.command == "brief":
        workspace = _ensure_workspace(args.workspace)
        result = get_brief(workspace, args.stage)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_brief(result)
        return 0

    if args.command == "complete-stage":
        workspace = _ensure_workspace(args.workspace)
        artifacts = json.loads(args.artifacts_json) if args.artifacts_json else None
        result = complete_stage(workspace, args.stage, args.notes, artifacts)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Completed stage: {result['completed_stage']}")
            print(f"Next stage: {result['next_stage']}")
        return 0

    if args.command == "export":
        workspace = _ensure_workspace(args.workspace)
        result = export_run(workspace)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
