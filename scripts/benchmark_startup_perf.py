#!/usr/bin/env python3
"""Benchmark Hermes startup/plugin/tool-schema hot paths.

This intentionally uses fresh Python subprocesses for each sample so import
costs, plugin discovery, and module-level side effects are measured the way a
new CLI/gateway process experiences them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


CASES: dict[str, str] = {
    "import_model_tools": (
        "import json, time\n"
        "start=time.perf_counter()\n"
        "import model_tools\n"
        "elapsed=time.perf_counter()-start\n"
        "print(json.dumps({'elapsed': elapsed, 'tools': len(model_tools.get_all_tool_names())}))\n"
    ),
    "import_and_get_tool_definitions": (
        "import json, time\n"
        "start=time.perf_counter()\n"
        "import model_tools\n"
        "defs=model_tools.get_tool_definitions(quiet_mode=True)\n"
        "elapsed=time.perf_counter()-start\n"
        "print(json.dumps({'elapsed': elapsed, 'tools': len(defs)}))\n"
    ),
    "get_tool_definitions": (
        "import json, time\n"
        "import model_tools\n"
        "start=time.perf_counter(); model_tools.get_tool_definitions(quiet_mode=True); cold=time.perf_counter()-start\n"
        "start=time.perf_counter(); model_tools.get_tool_definitions(quiet_mode=True); warm=time.perf_counter()-start\n"
        "print(json.dumps({'elapsed': cold, 'cold': cold, 'warm': warm}))\n"
    ),
    "discover_plugins_fast": (
        "import json, time\n"
        "from hermes_cli.plugins import PluginManager\n"
        "start=time.perf_counter(); mgr=PluginManager(); mgr.discover_and_load(include_platforms=False); elapsed=time.perf_counter()-start\n"
        "print(json.dumps({'elapsed': elapsed, 'plugins': len(mgr._plugins), 'platforms_loaded': mgr._platform_plugins_loaded}))\n"
    ),
    "discover_plugins_full": (
        "import json, time\n"
        "from hermes_cli.plugins import PluginManager\n"
        "start=time.perf_counter(); mgr=PluginManager(); mgr.discover_and_load(); elapsed=time.perf_counter()-start\n"
        "print(json.dumps({'elapsed': elapsed, 'plugins': len(mgr._plugins), 'platforms_loaded': mgr._platform_plugins_loaded}))\n"
    ),
}


def _run_case(case: str, code: str) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "case": case,
            "ok": False,
            "elapsed": wall,
            "stderr": proc.stderr.strip()[-2000:],
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout.strip()[-2000:]}
    payload.update({"case": case, "ok": True, "wall": wall})
    return payload


def _summarize(samples: list[dict]) -> dict:
    elapsed = [float(s["elapsed"]) for s in samples if s.get("ok")]
    if not elapsed:
        return {"ok": False, "samples": samples}
    return {
        "ok": True,
        "min": min(elapsed),
        "median": statistics.median(elapsed),
        "mean": statistics.mean(elapsed),
        "max": max(elapsed),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--samples", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args()

    results = {
        name: _summarize([_run_case(name, code) for _ in range(args.samples)])
        for name, code in CASES.items()
    }

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    print("| case | median | min | max | notes |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, summary in results.items():
        if not summary.get("ok"):
            print(f"| {name} | error | error | error | see JSON output |")
            continue
        sample = next((s for s in summary["samples"] if s.get("ok")), {})
        notes = []
        if "tools" in sample:
            notes.append(f"tools={sample['tools']}")
        if "plugins" in sample:
            notes.append(f"plugins={sample['plugins']}")
        if "platforms_loaded" in sample:
            notes.append(f"platforms_loaded={sample['platforms_loaded']}")
        if "warm" in sample:
            notes.append(f"warm={sample['warm']:.6f}s")
        print(
            f"| {name} | {summary['median']:.4f}s | {summary['min']:.4f}s | "
            f"{summary['max']:.4f}s | {', '.join(notes)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
