from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import shlex
import subprocess
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
    return routes_from_fallback_chain(raw_routes)


def routes_from_fallback_chain(fallback_chain: list[dict[str, Any]] | None) -> list[ProbeRoute]:
    routes: list[ProbeRoute] = []
    for entry in fallback_chain or []:
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
    if returncode == 124:
        return "timeout"
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
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124
    classification = classify_probe_result(returncode, stdout, stderr)
    return {
        "provider": route.provider,
        "cli_provider": route.cli_provider,
        "model": route.model,
        "ok": classification == "ok",
        "classification": classification,
        "returncode": returncode,
        "stdout": str(stdout).strip(),
        "stderr": str(stderr).strip(),
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


def probe_fallback_chain(
    fallback_chain: list[dict[str, Any]] | None,
    *,
    repo_root: Path,
    timeout: int = 180,
    max_workers: int = 4,
) -> dict[str, Any]:
    routes = routes_from_fallback_chain(fallback_chain)
    if not routes:
        return {"ok": False, "passed": 0, "failed": 0, "results": []}
    return probe_routes(routes, repo_root=repo_root, timeout=timeout, max_workers=max_workers)


def reorder_fallback_chain_by_probe(
    fallback_chain: list[dict[str, Any]] | None,
    probe_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    original = [dict(item) for item in (fallback_chain or []) if isinstance(item, dict)]
    if not original or not isinstance(probe_summary, dict):
        return original

    healthy_keys = []
    unhealthy_keys = []
    for item in probe_summary.get("results", []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("provider") or "").strip().lower(), str(item.get("model") or "").strip())
        if not all(key):
            continue
        if item.get("ok"):
            healthy_keys.append(key)
        else:
            unhealthy_keys.append(key)

    if not healthy_keys:
        return original

    rank = {key: idx for idx, key in enumerate(healthy_keys)}
    healthy_items = []
    unknown_items = []
    unhealthy_items = []
    unhealthy_key_set = set(unhealthy_keys)
    healthy_key_set = set(healthy_keys)

    for item in original:
        key = (str(item.get("provider") or "").strip().lower(), str(item.get("model") or "").strip())
        if key in healthy_key_set:
            healthy_items.append((rank[key], item))
        elif key in unhealthy_key_set:
            unhealthy_items.append(item)
        else:
            unknown_items.append(item)

    return [item for _, item in sorted(healthy_items, key=lambda pair: pair[0])] + unknown_items + unhealthy_items


def summary_as_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
