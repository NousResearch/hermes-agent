#!/usr/bin/env python3
"""Read-only geometry coverage benchmark for Hermes computer_use."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.computer_use.cua_backend import CuaDriverBackend  # noqa: E402


REPORT_DIR = ROOT / "execution-reports" / "computer-use-geometry"


def _is_actionable(element: Any) -> bool:
    role = str(getattr(element, "role", "") or "").lower()
    actions = getattr(element, "attributes", {}).get("actions")
    return bool(
        actions
        or any(token in role for token in (
            "button", "checkbox", "radio", "menu", "textfield", "textarea",
            "textfield", "link", "slider", "tab", "cell",
        ))
    )


def _metrics(app: str, cap: Any) -> Dict[str, Any]:
    elements = list(getattr(cap, "elements", []) or [])
    bounds_available = [
        e for e in elements
        if getattr(e, "attributes", {}).get("bounds_available") is True
    ]
    direct = [
        e for e in elements
        if getattr(e, "attributes", {}).get("geometry_status") == "direct"
    ]
    derived = [
        e for e in elements
        if getattr(e, "attributes", {}).get("geometry_status") == "derived"
    ]
    missing = [
        e for e in elements
        if getattr(e, "attributes", {}).get("bounds_available") is not True
    ]
    skipped = (
        getattr(cap, "width", 0) == 0
        and getattr(cap, "height", 0) == 0
        and not elements
    )
    return {
        "app": app,
        "status": "skipped" if skipped else "captured",
        "window_title": getattr(cap, "window_title", ""),
        "capture_width": getattr(cap, "width", 0),
        "capture_height": getattr(cap, "height", 0),
        "element_count": len(elements),
        "bounds_available_count": len(bounds_available),
        "missing_bounds_count": len(missing),
        "direct_count": len(direct),
        "derived_count": len(derived),
        "missing_actionable_count": len([e for e in missing if _is_actionable(e)]),
    }


def _markdown(results: Iterable[Dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# computer_use Geometry Benchmark",
        "",
        f"Generated: {generated_at}",
        "",
        "| App | Status | Window | Size | Elements | Bounds | Direct | Derived | Missing actionable |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        size = f"{row.get('capture_width', 0)}x{row.get('capture_height', 0)}"
        lines.append(
            f"| {row.get('app', '')} | {row.get('status', '')} | {row.get('window_title', row.get('error', ''))} | {size} | "
            f"{row.get('element_count', 0)} | {row.get('bounds_available_count', 0)} | "
            f"{row.get('direct_count', 0)} | {row.get('derived_count', 0)} | {row.get('missing_actionable_count', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", default="Finder,iTerm2,Terminal,TextEdit",
                        help="Comma-separated app/window filters to capture.")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="AX helper max depth, passed via environment-compatible backend setting.")
    parser.add_argument("--max-nodes", type=int, default=200,
                        help="AX helper max nodes, passed via environment-compatible backend setting.")
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("computer_use geometry benchmark is macOS-only; skipped.")
        return 0

    import os

    os.environ["HERMES_AX_GEOMETRY_MAX_DEPTH"] = str(args.max_depth)
    os.environ["HERMES_AX_GEOMETRY_MAX_NODES"] = str(args.max_nodes)

    apps = [app.strip() for app in args.apps.split(",") if app.strip()]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    json_path = REPORT_DIR / f"benchmark-{generated_at}.json"
    md_path = REPORT_DIR / f"benchmark-{generated_at}.md"

    backend = CuaDriverBackend()
    results: List[Dict[str, Any]] = []
    try:
        backend.start()
        for app in apps:
            try:
                cap = backend.capture(mode="som", app=app)
                results.append(_metrics(app, cap))
            except Exception as exc:
                results.append({"app": app, "status": "error", "error": repr(exc)})
    finally:
        backend.stop()

    payload = {
        "generated_at": generated_at,
        "apps": apps,
        "results": results,
        "report_markdown": str(md_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(results, generated_at))

    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
