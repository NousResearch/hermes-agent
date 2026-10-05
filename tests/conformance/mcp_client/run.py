"""Run the official MCP client conformance suite against Hermes' MCP client and compare every check
with ``baseline.json``.

For every protocol version Hermes negotiates (``2025-03-26``, ``2025-06-18``, ``2025-11-25``) the
runner lists the suite's client scenarios and runs them ONE AT A TIME (the sign-in flows pin loopback
callback ports; parallel scenarios would collide on them). The suite starts a scenario server and runs
``client.py`` against it; the suite's own checks plus a synthetic ``hermes-client`` check (did
``client.py`` complete the scenario) are compared with the committed baseline.

The run fails when a check that passes in the baseline fails or is missing, when a check fails that
the baseline does not list as failing, or when a baselined scenario did not run. Checks that started
passing are reported; ``--update-baseline`` locks them in (review the diff — never regenerate it to
hide a regression).

    python tests/conformance/mcp_client/run.py                       # full run, compare with the baseline
    python tests/conformance/mcp_client/run.py --mode 2025-11-25 --scenario auth/scope-step-up --verbose
    python tests/conformance/mcp_client/run.py --update-baseline

Needs ``npx`` (Node 20+) and, the first time, network access to fetch the pinned suite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE_PATH = HERE / "baseline.json"
CONFORMANCE_PACKAGE = "@modelcontextprotocol/conformance@0.2.0-alpha.11"
MODES = ("2025-03-26", "2025-06-18", "2025-11-25")
CLIENT_CHECK = "hermes-client"
# Per-scenario wall clock the suite grants the client; sign-in scenarios fetch metadata, register,
# authorize and exchange, and the simulated user's third refusal waits out a 20s callback timeout.
CLIENT_TIMEOUT_MS = 90_000


def _npx_env() -> dict[str, str]:
    return {**os.environ, "npm_config_ignore_scripts": "true", "npm_config_yes": "true"}


def _npx(args: list[str], *, capture: bool, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["npx", "--yes", CONFORMANCE_PACKAGE, *args], env=_npx_env(), cwd=str(REPO), text=True,
        stdin=subprocess.DEVNULL, capture_output=capture, timeout=timeout, check=False,
    )


def list_scenarios(mode: str) -> list[str]:
    proc = _npx(["list", "--client", "--spec-version", mode], capture=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"conformance list failed for {mode}:\n{proc.stdout}\n{proc.stderr}")
    return [line.strip()[2:].split(" ", 1)[0] for line in proc.stdout.splitlines() if line.strip().startswith("- ")]


def _write_launcher(work: Path) -> Path:
    """The suite splits ``--command`` on spaces, so the client is wrapped in a launcher at a path
    without any; it reuses THIS interpreter (the repo venv) for the client."""
    launcher = work / "client"
    report = work / "report.json"
    launcher.write_text(
        "#!/bin/sh\n"
        f"export HERMES_MCP_CONFORMANCE_REPORT={report}\n"
        f'exec {sys.executable} {HERE / "client.py"} "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


def run_scenario(mode: str, scenario: str, work: Path, launcher: Path, *, verbose: bool) -> dict[str, str]:
    """``{check_id: status}`` for one scenario, including the synthetic client check."""
    out_dir = work / "results"
    shutil.rmtree(out_dir, ignore_errors=True)
    report = work / "report.json"
    report.unlink(missing_ok=True)
    proc = _npx(
        ["client", "--command", str(launcher), "--scenario", scenario, "--spec-version", mode,
         "--timeout", str(CLIENT_TIMEOUT_MS), "-o", str(out_dir)],
        capture=not verbose, timeout=CLIENT_TIMEOUT_MS / 1000 + 120,
    )
    checks: dict[str, str] = {}
    for path in out_dir.rglob("checks.json"):
        for check in json.loads(path.read_text()):
            if check.get("status") in ("SUCCESS", "FAILURE"):
                checks[check["id"]] = check["status"]
    outcome = json.loads(report.read_text()) if report.exists() else {}
    checks[CLIENT_CHECK] = "SUCCESS" if outcome.get("ok") else "FAILURE"
    if not verbose and (proc.returncode != 0 or checks[CLIENT_CHECK] == "FAILURE"):
        print(f"    client errors: {outcome.get('errors') or proc.stderr.strip().splitlines()[-3:]}")
    return checks


def compare(baseline: dict, results: dict) -> tuple[list[str], list[str]]:
    """``(regressions, improvements)`` as ``mode/scenario:check`` lines."""
    regressions: list[str] = []
    improvements: list[str] = []
    for mode, scenarios in results.items():
        base_mode = baseline.get("modes", {}).get(mode, {})
        for scenario, checks in scenarios.items():
            base = base_mode.get(scenario)
            if base is None:
                for check, status in checks.items():
                    if status == "FAILURE":
                        regressions.append(f"{mode}/{scenario}:{check} fails (scenario not in baseline)")
                continue
            for check, status in base.items():
                now = checks.get(check)
                if status == "SUCCESS" and now != "SUCCESS":
                    regressions.append(f"{mode}/{scenario}:{check} {now or 'MISSING'} (baseline SUCCESS)")
                elif status == "FAILURE" and now == "SUCCESS":
                    improvements.append(f"{mode}/{scenario}:{check} now passes")
            for check, status in checks.items():
                if check not in base and status == "FAILURE":
                    regressions.append(f"{mode}/{scenario}:{check} fails (check not in baseline)")
        for scenario in base_mode:
            if scenario not in scenarios:
                regressions.append(f"{mode}/{scenario} did not run")
    return regressions, improvements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", action="append", choices=MODES, help="protocol version(s) to run (default: all)")
    parser.add_argument("--scenario", action="append", help="scenario name(s) to run (default: all listed)")
    parser.add_argument("--verbose", action="store_true", help="stream the suite's output")
    parser.add_argument("--update-baseline", action="store_true", help="write a full run's results to baseline.json")
    args = parser.parse_args()
    if shutil.which("npx") is None:
        raise SystemExit("npx not found: the conformance suite needs Node.js")
    subset = bool(args.mode or args.scenario)
    if args.update_baseline and subset:
        raise SystemExit("--update-baseline needs a full run (no --mode/--scenario)")

    results: dict[str, dict[str, dict[str, str]]] = {}
    work = Path(tempfile.mkdtemp(prefix="hermes-mcp-conformance-run-"))
    try:
        launcher = _write_launcher(work)
        for mode in args.mode or MODES:
            scenarios = list_scenarios(mode)
            if args.scenario:
                scenarios = [s for s in scenarios if s in args.scenario]
            results[mode] = {}
            for scenario in scenarios:
                print(f"[{mode}] {scenario}", flush=True)
                checks = run_scenario(mode, scenario, work, launcher, verbose=args.verbose)
                results[mode][scenario] = checks
                failed = sorted(c for c, s in checks.items() if s == "FAILURE")
                print(f"    {sum(s == 'SUCCESS' for s in checks.values())} passed"
                      + (f", failed: {', '.join(failed)}" if failed else ""), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps({"conformance": CONFORMANCE_PACKAGE, "modes": results}, indent=2) + "\n")
        print(f"wrote {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    if baseline.get("conformance") != CONFORMANCE_PACKAGE:
        print(f"baseline.json is for {baseline.get('conformance')}, not {CONFORMANCE_PACKAGE}; regenerate it")
        return 1
    if subset:  # only compare what ran
        baseline = {"modes": {m: {s: c for s, c in sc.items() if s in results.get(m, {})}
                              for m, sc in baseline["modes"].items() if m in results}}
    regressions, improvements = compare(baseline, results)
    for line in improvements:
        print(f"IMPROVED {line}")
    for line in regressions:
        print(f"REGRESSION {line}")
    if improvements and not regressions:
        print("checks started passing: run with --update-baseline to lock them in")
    print(f"{len(regressions)} regression(s), {len(improvements)} improvement(s)")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
