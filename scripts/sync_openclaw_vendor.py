#!/usr/bin/env python3
"""Layered OpenClaw vendor sync into vendor/openclaw-mirror.

Sources (priority):
  1. openclaw-sync/extensions/*     — official-aligned base (slim)
  2. clawdbot-main/extensions/*     — fork advantages overlay
  3. clawdbot-main3/vendor/*        — ShinkaEvolve, AI-Scientist

Also ports selected clawdbot-main scripts/tools/*.py → scripts/openclaw_ports/.

Usage:
  py -3 scripts/sync_openclaw_vendor.py --dry-run
  py -3 scripts/sync_openclaw_vendor.py --execute
  py -3 scripts/sync_openclaw_vendor.py --execute --port-cli-tools

For SakanaAI/AI-Scientist (upstream fetch + local template overlay), use:
  py -3 scripts/sync_ai_scientist_vendor.py --dry-run
  py -3 scripts/sync_ai_scientist_vendor.py --execute
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_TOOLS = REPO_ROOT / "scripts" / "merge_tools"
DEFAULT_CLAW_ROOT = REPO_ROOT.parent / "clawdbot-main3"
DEFAULT_OPENCLAW_SYNC = DEFAULT_CLAW_ROOT / "openclaw-sync"
DEFAULT_CLAWDBOT_MAIN = DEFAULT_CLAW_ROOT / "clawdbot-main"
DEFAULT_VENDOR_ROOT = REPO_ROOT / "vendor" / "openclaw-mirror"
LAYERS_CONFIG = MERGE_TOOLS / "openclaw_vendor_layers.json"

sys.path.insert(0, str(MERGE_TOOLS))
from openclaw_layered_sync import (  # noqa: E402
    SourceRoots,
    apply_layered_plan,
    build_layer_plans,
    diff_layered_plan,
    load_layers_config,
)


@dataclass(frozen=True)
class VendorCopyPlan:
    source: Path
    target: Path
    rel: str


CLI_TOOL_PORT_MAP = {
    "voicevox_speak.py": "voicevox_speak.py",
    "verify_voicevox.py": "verify_voicevox.py",
    "osc_chatbox.py": "osc_chatbox.py",
    "vrchat_evolution_pulse.py": "vrchat_evolution_pulse.py",
    "hakua_evolution_core.py": "hakua_evolution_core.py",
    "singularity_bridge.py": "singularity_bridge.py",
    "channel_audit.py": "channel_audit.py",
    "runtime_config_audit.py": "runtime_config_audit.py",
}


def _vendor_package_plans(claw_root: Path, vendor_root: Path, config: dict) -> list[VendorCopyPlan]:
    plans: list[VendorCopyPlan] = []
    for name in config.get("vendor_packages", {}):
        source = claw_root / "vendor" / name
        if source.is_dir():
            plans.append(VendorCopyPlan(source=source, target=vendor_root / name, rel=name))
    return plans


def _diff_vendor_copy(plan: VendorCopyPlan, skip_dirs: set[str], skip_globs: tuple[str, ...]) -> dict[str, object]:
    from openclaw_layered_sync import collect_files

    src = collect_files(plan.source, skip_dirs=skip_dirs, skip_globs=skip_globs)
    dst = collect_files(plan.target, skip_dirs=skip_dirs, skip_globs=skip_globs) if plan.target.exists() else {}
    added = sorted(set(src) - set(dst))
    removed = sorted(set(dst) - set(src))
    changed = sorted(rel for rel in set(src) & set(dst) if src[rel] != dst[rel])
    return {
        "rel": plan.rel,
        "source": str(plan.source),
        "target": str(plan.target),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _rmtree_robust(path: Path) -> None:
    """Remove a directory tree; clear read-only bits on Windows (.git pack files)."""

    def _onerror(func, p, _exc_info) -> None:
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR)
            func(p)
        else:
            raise

    shutil.rmtree(path, onerror=_onerror)


def _copy_vendor_package(plan: VendorCopyPlan, skip_dirs: set[str]) -> None:
    if plan.target.exists():
        _rmtree_robust(plan.target)
    shutil.copytree(plan.source, plan.target, ignore=shutil.ignore_patterns(*skip_dirs))


def _port_cli_tools(clawdbot_main: Path, dry_run: bool) -> list[dict[str, str]]:
    src_dir = clawdbot_main / "scripts" / "tools"
    dst_dir = REPO_ROOT / "scripts" / "openclaw_ports"
    results: list[dict[str, str]] = []
    header = (
        '"""Ported from clawdbot-main/scripts/tools — run via: py -3 scripts/openclaw_ports/<name>"""\n\n'
    )
    for src_name, dst_name in CLI_TOOL_PORT_MAP.items():
        src = src_dir / src_name
        if not src.is_file():
            results.append({"file": dst_name, "status": "missing-source"})
            continue
        dst = dst_dir / dst_name
        if dry_run:
            results.append({"file": dst_name, "status": "would-port"})
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        body = src.read_text(encoding="utf-8")
        if not body.startswith('"""Ported from clawdbot'):
            body = header + body
        dst.write_text(body, encoding="utf-8")
        results.append({"file": dst_name, "status": "ported"})
    return results


def _write_report(payload: dict[str, object]) -> Path:
    out_dir = REPO_ROOT / "_docs" / "merge-reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"openclaw-vendor-sync-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layered OpenClaw vendor sync (openclaw-sync + clawdbot-main).",
    )
    parser.add_argument("--claw-root", type=Path, default=DEFAULT_CLAW_ROOT)
    parser.add_argument("--openclaw-sync", type=Path, default=None)
    parser.add_argument("--clawdbot-main", type=Path, default=None)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--layers-config", type=Path, default=LAYERS_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--port-cli-tools",
        action="store_true",
        help="Port clawdbot-main scripts/tools/*.py into scripts/openclaw_ports/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.execute and not args.port_cli_tools:
        print("Specify --dry-run, --execute, and/or --port-cli-tools", file=sys.stderr)
        return 2

    claw_root = args.claw_root.resolve()
    openclaw_sync = (args.openclaw_sync or claw_root / "openclaw-sync").resolve()
    clawdbot_main = (args.clawdbot_main or claw_root / "clawdbot-main").resolve()
    vendor_root = args.vendor_root.resolve()
    config = load_layers_config(args.layers_config.resolve())

    skip_dirs = set(config.get("skip_dir_names", []))
    skip_globs = tuple(config.get("skip_globs", []))

    if not openclaw_sync.is_dir():
        print(f"error: openclaw-sync missing: {openclaw_sync}", file=sys.stderr)
        return 2
    if not clawdbot_main.is_dir():
        print(f"error: clawdbot-main missing: {clawdbot_main}", file=sys.stderr)
        return 2

    sources = SourceRoots(openclaw_sync=openclaw_sync, clawdbot_main=clawdbot_main, claw_root=claw_root)
    layer_plans = build_layer_plans(sources, vendor_root, config)
    vendor_plans = _vendor_package_plans(claw_root, vendor_root, config)

    diffs: list[dict[str, object]] = []
    for plan in layer_plans:
        ext_cfg = config["extensions"][plan.extension]
        diffs.append(
            diff_layered_plan(
                plan,
                skip_dirs=skip_dirs,
                skip_globs=skip_globs,
                overlay_paths=ext_cfg.get("overlay_paths", []),
                prefer_overlay_for_changed=ext_cfg.get("prefer_overlay_for_changed", []),
            ),
        )
    for vplan in vendor_plans:
        diffs.append(_diff_vendor_copy(vplan, skip_dirs, skip_globs))

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "openclaw_sync": str(openclaw_sync),
        "clawdbot_main": str(clawdbot_main),
        "vendor_root": str(vendor_root),
        "dry_run": args.dry_run,
        "executed": False,
        "plans": diffs,
    }

    print("OpenClaw layered vendor sync:")
    for item in diffs:
        name = item.get("extension") or item.get("rel")
        overlay_n = item.get("overlay_applied_count", "-")
        print(
            f"  {name}: +{len(item['added'])} ~{len(item['changed'])} -{len(item['removed'])}"
            f" (overlay files: {overlay_n})",
        )

    if args.execute:
        for plan in layer_plans:
            ext_cfg = config["extensions"][plan.extension]
            apply_layered_plan(
                plan,
                skip_dirs=skip_dirs,
                skip_globs=skip_globs,
                overlay_paths=ext_cfg.get("overlay_paths", []),
                prefer_overlay_for_changed=ext_cfg.get("prefer_overlay_for_changed", []),
            )
        for vplan in vendor_plans:
            _copy_vendor_package(vplan, skip_dirs)
        report["executed"] = True
        print("Applied layered vendor sync.")

    if args.port_cli_tools:
        report["cli_ports"] = _port_cli_tools(clawdbot_main, dry_run=not args.execute)
        if report["cli_ports"]:
            print("CLI tool ports:", ", ".join(f"{r['file']}:{r['status']}" for r in report["cli_ports"]))

    path = _write_report(report)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
