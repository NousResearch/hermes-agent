#!/usr/bin/env python3
"""No-network smoke for smart model routing classifier (M3).

Does not call providers. Verifies config shape and classify_turn() when
hermes_cli.smart_routing is importable (falls back to known worktree path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

DEFAULT_SMART_ROUTING_WORKTREE = (
    "/home/khall/.hermes/hermes-agent/.worktrees/smart-routing-final-20260502T014157Z"
)


def _load_config(path: Path) -> dict:
    if yaml is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _ensure_smart_routing_import(source_root: Path, worktree: Path | None) -> tuple[bool, str]:
    roots = [source_root]
    if worktree:
        roots.insert(0, worktree)
    for root in roots:
        if (root / "hermes_cli" / "smart_routing.py").is_file():
            sys.path.insert(0, str(root))
            try:
                import hermes_cli.smart_routing  # noqa: F401
                return True, str(root)
            except Exception as exc:
                return False, f"import failed from {root}: {exc}"
    return False, "hermes_cli/smart_routing.py not found on main or worktree"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/khall/.hermes/hermes-agent"),
    )
    parser.add_argument("--smart-routing-worktree", type=Path, default=Path(DEFAULT_SMART_ROUTING_WORKTREE))
    parser.add_argument("--no-network", action="store_true", help="Assert no network calls (always true here)")
    args = parser.parse_args()

    cfg = _load_config(args.hermes_home / "config.yaml")
    smr = cfg.get("smart_model_routing") if isinstance(cfg, dict) else None
    enabled = bool(isinstance(smr, dict) and smr.get("enabled"))

    main_has_module = (args.source_root / "hermes_cli" / "smart_routing.py").is_file()
    ok_import, import_from = _ensure_smart_routing_import(args.source_root, args.smart_routing_worktree)

    report: dict[str, Any] = {
        "no_network": bool(args.no_network),
        "smart_model_routing_enabled_in_config": enabled,
        "canonical_on_main": main_has_module,
        "import_ok": ok_import,
        "import_from": import_from,
        "samples": [],
        "overall": "pass",
    }

    if enabled and not main_has_module:
        report["blocker"] = "config enabled but smart_routing not on installed main checkout"
        report["overall"] = "needs_operator"

    if ok_import:
        from hermes_cli.smart_routing import classify_turn, build_turn_route

        samples = [
            ("ping", "cheap"),
            ("add unit tests for profiles.py", "routine_coding"),
            ("architecture security review migration", "hard"),
        ]
        for text, expect_kind in samples:
            kind, reason = classify_turn(text, smr or {})
            route = build_turn_route(
                user_message=text,
                model=str((cfg.get("model") or {}).get("default") if isinstance(cfg.get("model"), dict) else ""),
                runtime={"provider": "xai-oauth"},
                config=cfg,
            )
            report["samples"].append(
                {
                    "text": text,
                    "classify_kind": kind,
                    "classify_reason": reason,
                    "expect_kind": expect_kind,
                    "route_kind": route.get("route_kind"),
                    "route_model": route.get("model"),
                    "route_reason": route.get("route_reason"),
                    "classify_match": kind == expect_kind,
                }
            )
        if not all(s["classify_match"] for s in report["samples"]):
            report["overall"] = "fail"
    elif enabled:
        report["overall"] = "fail"
        report["blocker"] = report.get("blocker") or "cannot import smart_routing for classifier smoke"

    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())