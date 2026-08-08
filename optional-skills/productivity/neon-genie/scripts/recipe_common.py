#!/usr/bin/env python3
"""Shared helpers for Neon Genie packaging recipes (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable
CLI = SCRIPT_DIR / "neon_genie.py"


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(CLI), *args],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(SKILL_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def route_request(brief: Path) -> dict[str, Any]:
    r = run_cli(["do", "route", "--request", str(brief), "--json"])
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "route failed")
    return json.loads(r.stdout)


def build_receipt(
    out_path: Path,
    profiles: list[str],
    *,
    status: str = "PROPOSED",
    promotion_state: str = "RAW_SIGNAL",
    not_computable: str = "",
    packets: list[Path] | None = None,
    data_requests: list | None = None,
    brief: Path | None = None,
    privacy_mode: str | None = None,
    external_actions: list | None = None,
    privacy_context: dict | None = None,
) -> None:
    args = [
        "do",
        "receipt",
        "--profiles",
        ",".join(profiles) if profiles else "core",
        "--status",
        status,
        "--promotion-state",
        promotion_state,
        "--out",
        str(out_path),
    ]
    if not_computable:
        args.extend(["--not-computable", not_computable])
    for p in packets or []:
        args.extend(["--packet", str(p)])
    if data_requests:
        dr_path = out_path.parent / "_data_requests.json"
        write_json(dr_path, data_requests)
        args.extend(["--data-requests", str(dr_path)])
    if brief is not None:
        args.extend(["--brief", str(brief)])
    if privacy_mode:
        args.extend(["--privacy-mode", privacy_mode])
    if privacy_context is not None:
        pc_path = out_path.parent / "_privacy_context.json"
        write_json(pc_path, privacy_context)
        args.extend(["--privacy-context", str(pc_path)])
    if external_actions:
        ea_path = out_path.parent / "_external_actions.json"
        write_json(ea_path, external_actions)
        args.extend(["--external-actions", str(ea_path)])
    r = run_cli(args)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "receipt failed")


def validate_packet(packet: Path, packet_type: str) -> None:
    r = run_cli(["do", "validate", "--packet", str(packet), "--type", packet_type])
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "validate failed")


def finish(
    *,
    recipe: str,
    brief: Path,
    out: Path,
    route: dict[str, Any],
    artifacts: list[Path],
    extra: dict[str, Any] | None = None,
) -> int:
    summary: dict[str, Any] = {
        "recipe": recipe,
        "brief": rel(brief),
        "selected_profiles": route.get("selected"),
        "artifacts": [rel(p) for p in artifacts],
        "status": "PASS",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    if extra:
        summary.update(extra)
    write_json(out / "recipe-summary.json", summary)

    # Canonical entry point for downstream consumers (Wayfinder, orchestrators)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import build_envelope as be  # noqa: WPS433

    try:
        env_path = be.write_envelope(
            out,
            recipe=recipe,
            brief=brief,
            route=route,
        )
        # Validate envelope shape (advisory_only enforced in builder + schema)
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if env.get("authority") != "advisory_only" or env.get("grants_execution") is True:
            raise RuntimeError("envelope violated advisory_only")
        if "run_id" not in env or "primary_artifact" not in env:
            raise RuntimeError("envelope missing run_id or primary_artifact")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: run-envelope: {exc}", file=sys.stderr)
        return 1

    print(f"PASS: {recipe} packaging recipe")
    print(f"  profiles: {', '.join(route.get('selected') or [])}")
    print(f"  out: {out}")
    print(f"  envelope: {env_path.name} ({env.get('run_id')})")
    return 0
