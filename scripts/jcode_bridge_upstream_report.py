#!/usr/bin/env python3
"""Generate an upstream-sync report for the Hermes <-> jcode bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.jcode_bridge.hermes_service import service_contract_report  # noqa: E402
from plugins.jcode_bridge.tools import handle_jcode_contract_check  # noqa: E402


_CORPUS_RE = re.compile(r"- (?P<files>[\d,]+) files .+~(?P<words>[\d,]+) words")
_SUMMARY_RE = re.compile(
    r"- (?P<nodes>[\d,]+) nodes .+ (?P<edges>[\d,]+) edges .+ "
    r"(?P<communities>[\d,]+) communities detected"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int(raw: str) -> int:
    return int(raw.replace(",", ""))


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _git(cwd: Path, *args: str) -> str | None:
    completed = _run(["git", *args], cwd=cwd)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _repo_state(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "path": str(path),
            "exists": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_count": None,
            "status_sample": [],
        }

    status = _git(path, "status", "--short") or ""
    status_lines = [line for line in status.splitlines() if line.strip()]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "commit": _git(path, "rev-parse", "HEAD"),
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status_lines),
        "status_count": len(status_lines),
        "status_sample": status_lines[:20],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, include_hashes: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return result
    stat = path.stat()
    result.update({
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    })
    if include_hashes:
        result["sha256"] = _sha256(path)
    return result


def _parse_graph_report(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not path.exists():
        return parsed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        corpus = _CORPUS_RE.match(line)
        if corpus:
            parsed["files"] = _int(corpus.group("files"))
            parsed["approx_words"] = _int(corpus.group("words"))
            continue
        summary = _SUMMARY_RE.match(line)
        if summary:
            parsed["nodes"] = _int(summary.group("nodes"))
            parsed["edges"] = _int(summary.group("edges"))
            parsed["communities"] = _int(summary.group("communities"))
            continue
        if parsed.get("nodes") is not None and parsed.get("files") is not None:
            break
    return parsed


def _graph_state(repo_path: Path, *, include_hashes: bool) -> dict[str, Any]:
    graph_dir = repo_path / "graphify-out"
    report_path = graph_dir / "GRAPH_REPORT.md"
    graph_path = graph_dir / "graph.json"
    return {
        "report": _artifact(report_path, include_hashes=include_hashes),
        "graph_json": _artifact(graph_path, include_hashes=include_hashes),
        "summary": _parse_graph_report(report_path),
    }


def _contract_report(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(handle_jcode_contract_check({
        "live": args.live,
        "live_run": args.live_run,
        "live_run_message": args.live_run_message,
        "jcode_bin": args.jcode_bin,
        "cwd": args.live_cwd,
        "timeout_seconds": args.timeout_seconds,
    }))


def _smoke_report() -> dict[str, Any]:
    script = ROOT / "scripts" / "jcode_bridge_smoke.py"
    completed = _run([sys.executable, str(script)], cwd=ROOT)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "error": "failed to parse smoke output",
            "stdout": completed.stdout,
        }
    payload["returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr
    if completed.returncode != 0:
        payload["success"] = False
    return payload


def _mcp_contract_report() -> dict[str, Any]:
    script = ROOT / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"
    completed = _run([sys.executable, str(script), "--check", "--live"], cwd=ROOT)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "error": "failed to parse Hermes MCP contract output",
            "stdout": completed.stdout,
        }
    payload["returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr
    if completed.returncode != 0:
        payload["success"] = False
    return payload


def _latency_report() -> dict[str, Any]:
    script = ROOT / "scripts" / "jcode_bridge_latency_probe.py"
    completed = _run([sys.executable, str(script), "--iterations", "30"], cwd=ROOT)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "error": "failed to parse latency probe output",
            "stdout": completed.stdout,
        }
    payload["returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr
    if completed.returncode != 0:
        payload["success"] = False
    return payload


def _native_tool_report(jcode_path: Path, *, cargo: bool) -> dict[str, Any]:
    script = ROOT / "scripts" / "jcode_native_tool_check.py"
    cmd = [sys.executable, str(script), "--jcode", str(jcode_path)]
    if not cargo:
        cmd.append("--skip-cargo")
    completed = _run(cmd, cwd=ROOT)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "error": "failed to parse native jcode tool output",
            "stdout": completed.stdout,
        }
    payload["returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr
    if completed.returncode != 0:
        payload["success"] = False
    return payload


def _recommendations(report: dict[str, Any]) -> list[str]:
    items: list[str] = []
    contract = report.get("bridge_contract", {})
    if not contract.get("success"):
        items.append("Do not bump upstreams until jcode-bridge contract checks pass.")
    service_contract = report.get("hermes_service_contract", {})
    if not service_contract.get("success"):
        items.append("Do not bump upstreams until Hermes service contract checks pass.")
    mcp_contract = report.get("hermes_mcp_contract", {})
    if not mcp_contract.get("success"):
        items.append("Do not bump upstreams until Hermes MCP contract checks pass.")
    latency = report.get("bridge_latency", {})
    if isinstance(latency, dict) and not latency.get("success"):
        items.append("Review bridge latency probe failure before pinning upstreams.")
    native_tool = report.get("jcode_native_tool", {})
    if isinstance(native_tool, dict) and not native_tool.get("success"):
        items.append("Do not bump upstreams until the native jcode Hermes tool check passes.")
    smoke = report.get("bridge_smoke")
    if isinstance(smoke, dict) and not smoke.get("success"):
        items.append("Do not bump upstreams until jcode-bridge smoke checks pass.")
    for name in ("hermes", "jcode"):
        repo = report.get("repos", {}).get(name, {})
        if repo.get("dirty"):
            items.append(f"Review dirty worktree entries before pinning {name}.")
        graph = report.get("graphify", {}).get(name, {})
        if not graph.get("report", {}).get("exists"):
            items.append(f"Run graphify for {name}; GRAPH_REPORT.md is missing.")
    if not items:
        items.append("Bridge contract and Graphify artifacts are present for this snapshot.")
    return items


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    hermes_path = Path(args.hermes).expanduser().resolve()
    jcode_path = Path(args.jcode).expanduser().resolve()
    report: dict[str, Any] = {
        "generated_at": _now_iso(),
        "report_version": "hermes-jcode-upstream-sync.v1",
        "repos": {
            "hermes": _repo_state(hermes_path),
            "jcode": _repo_state(jcode_path),
        },
        "graphify": {
            "hermes": _graph_state(hermes_path, include_hashes=args.hash_artifacts),
            "jcode": _graph_state(jcode_path, include_hashes=args.hash_artifacts),
        },
        "bridge_contract": _contract_report(args),
        "hermes_service_contract": service_contract_report(),
        "hermes_mcp_contract": _mcp_contract_report(),
        "bridge_latency": _latency_report(),
        "jcode_native_tool": _native_tool_report(
            jcode_path,
            cargo=not args.skip_native_cargo,
        ),
    }
    if args.smoke:
        report["bridge_smoke"] = _smoke_report()

    repos_present = all(item.get("exists") for item in report["repos"].values())
    graph_reports_present = all(
        item.get("report", {}).get("exists")
        for item in report["graphify"].values()
    )
    smoke_ok = True
    if args.smoke:
        smoke_ok = bool(report.get("bridge_smoke", {}).get("success"))
    report["success"] = (
        bool(report["bridge_contract"].get("success"))
        and bool(report["hermes_service_contract"].get("success"))
        and bool(report["hermes_mcp_contract"].get("success"))
        and bool(report["bridge_latency"].get("success"))
        and bool(report["jcode_native_tool"].get("success"))
        and repos_present
        and graph_reports_present
        and smoke_ok
    )
    report["recommendations"] = _recommendations(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    repos = report["repos"]
    graphify = report["graphify"]
    lines = [
        "# Hermes/jcode upstream sync report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Repositories",
        "",
        "| Repo | Branch | Commit | Dirty |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("hermes", "jcode"):
        repo = repos[name]
        lines.append(
            f"| {name} | {repo.get('branch') or ''} | {repo.get('commit') or ''} | "
            f"{repo.get('dirty')} |"
        )

    lines.extend([
        "",
        "## Graphify",
        "",
        "| Repo | Files | Nodes | Edges | Communities | Report |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for name in ("hermes", "jcode"):
        graph = graphify[name]
        summary = graph.get("summary", {})
        report_path = graph.get("report", {}).get("path", "")
        lines.append(
            f"| {name} | {summary.get('files', '')} | {summary.get('nodes', '')} | "
            f"{summary.get('edges', '')} | {summary.get('communities', '')} | "
            f"{report_path} |"
        )

    contract = report["bridge_contract"]
    lines.extend([
        "",
        "## Bridge Contract",
        "",
        f"Success: {contract.get('success')}",
        f"Version: {contract.get('contract_version')}",
        f"Schema dir: {contract.get('schema_dir')}",
        f"Fixture dir: {contract.get('fixture_dir')}",
        "",
        "| Check | OK |",
        "| --- | --- |",
    ])
    for check in contract.get("checks", []):
        lines.append(f"| {check.get('name')} | {check.get('ok')} |")

    smoke = report.get("bridge_smoke")
    if isinstance(smoke, dict):
        lines.extend([
            "",
            "## Bridge Smoke",
            "",
            f"Success: {smoke.get('success')}",
            "",
            "| Check | OK |",
            "| --- | --- |",
        ])
        for check in smoke.get("checks", []):
            lines.append(f"| {check.get('name')} | {check.get('ok')} |")

    service_contract = report["hermes_service_contract"]
    lines.extend([
        "",
        "## Hermes Service Contract",
        "",
        f"Success: {service_contract.get('success')}",
        f"Version: {service_contract.get('contract_version')}",
        f"Schema dir: {service_contract.get('schema_dir')}",
        f"Fixture dir: {service_contract.get('fixture_dir')}",
        "",
        "| Check | OK |",
        "| --- | --- |",
    ])
    for check in service_contract.get("checks", []):
        lines.append(f"| {check.get('name')} | {check.get('ok')} |")

    mcp_contract = report["hermes_mcp_contract"]
    lines.extend([
        "",
        "## Hermes MCP Contract",
        "",
        f"Success: {mcp_contract.get('success')}",
        f"Version: {mcp_contract.get('contract_version')}",
        f"Schema dir: {mcp_contract.get('schema_dir')}",
        f"Fixture dir: {mcp_contract.get('fixture_dir')}",
        "",
        "| Check | OK |",
        "| --- | --- |",
    ])
    for check in mcp_contract.get("checks", []):
        lines.append(f"| {check.get('name')} | {check.get('ok')} |")

    latency = report.get("bridge_latency", {})
    summary = latency.get("summary", {}) if isinstance(latency, dict) else {}
    lines.extend([
        "",
        "## Bridge Latency",
        "",
        f"Success: {latency.get('success')}",
        f"Probe: {latency.get('probe')}",
        f"Iterations: {latency.get('iterations')}",
        "",
        "| Metric | ms |",
        "| --- | ---: |",
        f"| min | {summary.get('min_ms', '')} |",
        f"| p50 | {summary.get('p50_ms', '')} |",
        f"| p95 | {summary.get('p95_ms', '')} |",
        f"| max | {summary.get('max_ms', '')} |",
    ])

    native_tool = report.get("jcode_native_tool", {})
    lines.extend([
        "",
        "## jcode Native Hermes Tool",
        "",
        f"Success: {native_tool.get('success')}",
        f"Native tool dir: {native_tool.get('native_tool_dir')}",
        f"jcode path: {native_tool.get('jcode_path')}",
        "",
        "| Check | OK |",
        "| --- | --- |",
    ])
    for check in native_tool.get("checks", []):
        lines.append(f"| {check.get('name')} | {check.get('ok')} |")

    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    for item in report.get("recommendations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes", default=str(ROOT), help="Hermes repo path.")
    parser.add_argument(
        "--jcode",
        default=str(ROOT / ".codex-research" / "jcode"),
        help="jcode repo path.",
    )
    parser.add_argument("--jcode-bin", help="Optional path to a jcode executable.")
    parser.add_argument("--live-cwd", help="Optional working directory for live jcode checks.")
    parser.add_argument("--live", action="store_true", help="Run live jcode version check.")
    parser.add_argument(
        "--live-run",
        action="store_true",
        help="With --live, run one jcode prompt and validate JSON output.",
    )
    parser.add_argument(
        "--live-run-message",
        default="Reply with exactly OK.",
        help="Prompt to use for --live-run.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for live jcode checks.",
    )
    parser.add_argument(
        "--hash-artifacts",
        action="store_true",
        help="Include SHA-256 hashes for Graphify artifacts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run stdlib bridge smoke checks and include them in the report.",
    )
    parser.add_argument(
        "--skip-native-cargo",
        action="store_true",
        help="Only source-check the native jcode Hermes tool; skip cargo check.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    parser.add_argument("--output", help="Optional file path for the report.")
    args = parser.parse_args(argv)

    report = build_report(args)
    content = (
        _markdown(report)
        if args.format == "markdown"
        else json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    )

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")

    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
