#!/usr/bin/env python3
"""Low-lift concurrent fallback probe for Hermes provider lanes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


@dataclass(frozen=True)
class ProbeRoute:
    provider: str
    cli_provider: str
    model: str


def normalize_cli_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized == "google":
        return "gemini"
    return normalized


def load_routes_from_config(config_path: Path) -> list[ProbeRoute]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_routes = payload.get("fallback_providers") or []
    routes: list[ProbeRoute] = []
    for entry in raw_routes:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue
        routes.append(
            ProbeRoute(
                provider=provider,
                cli_provider=normalize_cli_provider(provider),
                model=model,
            )
        )
    return routes


def build_probe_command(route: ProbeRoute) -> str:
    inner = (
        "source venv/bin/activate && "
        "hermes chat -q 'Reply with exactly: PONG' "
        f"--provider {shlex.quote(route.cli_provider)} "
        f"-m {shlex.quote(route.model)} "
        "-Q --max-turns 2"
    )
    return f"bash -lc {shlex.quote(inner)}"


def classify_probe_result(returncode: int, stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode == 0 and "pong" in combined:
        return "ok"
    if "429" in combined or "quota exceeded" in combined or "rate limit" in combined:
        return "rate_limit"
    if "401" in combined or "unauthorized" in combined or "authentication" in combined:
        return "auth"
    if "timeout" in combined:
        return "timeout"
    if "invalid choice" in combined:
        return "config"
    return "error"


def run_probe(route: ProbeRoute, *, repo_root: Path, timeout: int) -> dict[str, Any]:
    command = build_probe_command(route)
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    classification = classify_probe_result(completed.returncode, completed.stdout, completed.stderr)
    ok = classification == "ok"
    return {
        "provider": route.provider,
        "cli_provider": route.cli_provider,
        "model": route.model,
        "ok": ok,
        "classification": classification,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "command": command,
    }


def probe_routes(
    routes: list[ProbeRoute],
    *,
    repo_root: Path,
    timeout: int,
    max_workers: int,
) -> dict[str, Any]:
    workers = max(1, min(max_workers, len(routes) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda route: run_probe(route, repo_root=repo_root, timeout=timeout), routes))
    passed = sum(1 for item in results if item["ok"])
    failed = len(results) - passed
    return {
        "ok": failed == 0,
        "passed": passed,
        "failed": failed,
        "results": results,
    }


def _parse_route_arg(raw: str) -> ProbeRoute:
    provider, sep, model = raw.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise argparse.ArgumentTypeError("route must be provider:model")
    provider = provider.strip()
    return ProbeRoute(provider=provider, cli_provider=normalize_cli_provider(provider), model=model.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe configured Hermes fallback providers with low-lift concurrent PONG checks")
    parser.add_argument("--config", type=Path, default=Path("~/.hermes/config.yaml").expanduser())
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--route", action="append", type=_parse_route_arg, default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routes = list(args.route) or load_routes_from_config(args.config)
    if not routes:
        print("No fallback routes configured.", file=sys.stderr)
        return 2

    summary = probe_routes(
        routes,
        repo_root=args.repo_root,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Fallback probe summary: {summary['passed']} passed, {summary['failed']} failed")
        for item in summary["results"]:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"- {status} {item['provider']} / {item['model']} [{item['classification']}]")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
