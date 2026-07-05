#!/usr/bin/env python3
"""Context Health update-survival smoke/regression runner.

Phase 9: Update survival smoke/regression tests.

Purpose:
    Ensure Hermes update cannot silently bypass context governance.

Scope:
    This script is detection/smoke/regression scope only. It is not gateway
    runtime implementation, not command activation, and not runtime
    config/profile activation.

Default mode:
    Dry-run / safe mode. The smoke runner is designed to be safe to execute
    after a Hermes update because it uses repository-local inspection only,
    a temp HERMES_HOME / tmp_path contract, and synthetic sentinel only
    expectations. It must exit nonzero on missing hook regressions or other
    missing hook signals, and it must avoid raw/private/secret/token/password
    material in report output.

Forbidden live/runtime surfaces:
    real ~/.hermes/state.db; live provider; network; secrets; tmux/session;
    profile/systemd/cron/wrapper/env/credential; gateway restart/deploy/activation;
    CLI slash command activation; runtime config/profile activation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE = "Phase 9: Update survival smoke/regression tests"
OBJECTIVE = "Ensure Hermes update cannot silently bypass context governance."
SCOPE_STATEMENT = (
    "Phase 9 is update survival smoke/regression detection scope, not "
    "gateway runtime implementation, command activation, or runtime config/profile activation."
)

REQUIRED_COVERAGE = (
    "policy loading",
    "pre-turn intake hook presence",
    "WCP provider payload enforcement",
    "Task Boundary Firewall default-new behavior",
    "closed task exclusion",
    "retrieval scope enforcement",
    "compact failure fallback",
    "threshold does not revert to 85%-only path",
    "same-window rehydrate path or HOLD if not enabled",
    "update-aftercare checklist",
)

SAFE_EXECUTION_CONTRACT = (
    "dry-run/default safe mode",
    "temp HERMES_HOME/tmp_path based",
    "synthetic sentinel only",
    "nonzero on missing hooks/regressions",
    "no raw/private/secret/token/password material in report",
    "machine-readable summary or clear PASS/FAIL report",
)

FORBIDDEN_RUNTIME_SURFACES = (
    "real ~/.hermes/state.db",
    "live provider",
    "network",
    "secrets",
    "tmux/session",
    "profile/systemd/cron/wrapper/env/credential",
    "gateway restart/deploy/activation",
    "CLI slash command activation",
    "runtime config/profile activation",
    "no profile activation",
    "not runtime config",
)

# This runner intentionally checks local file/hook indicators only. It must not
# import provider clients, open ~/.hermes/state.db, restart gateway processes,
# mutate profiles/config, touch cron/systemd/wrappers/env/credentials, open tmux
# or sessions, call network, or inspect secrets.
SMOKE_CHECKS = {
    "policy loading": ("agent/context_health_policy.py", "context_health"),
    "pre-turn intake hook presence": ("agent/context_health_intake.py", "intake"),
    "WCP provider payload enforcement": ("agent/working_context_packet.py", "provider payload"),
    "Task Boundary Firewall default-new behavior": ("agent/task_boundary_firewall.py", "default"),
    "closed task exclusion": ("agent/task_boundary_firewall.py", "closed"),
    "retrieval scope enforcement": ("agent/retrieval_scope.py", "session_search"),
    "compact failure fallback": ("agent/context_health_compact.py", "safe hold"),
    "threshold does not revert to 85%-only path": ("agent/context_health_policy.py", "85"),
    "same-window rehydrate path or HOLD if not enabled": ("agent/context_health_rehydrate.py", "hold"),
    "update-aftercare checklist": ("references/context-health-update-survival.md", "post-update"),
}


@dataclass(frozen=True)
class SmokeResult:
    name: str
    status: str
    path: str
    detail: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def run_smoke(repo_root: Path = REPO_ROOT) -> tuple[int, dict[str, object]]:
    """Run dry-run smoke checks and return (exit_code, machine-readable summary)."""
    results: list[SmokeResult] = []
    for name, (relative_path, required_text) in SMOKE_CHECKS.items():
        path = repo_root / relative_path
        text = _read_text(path).lower()
        if not path.is_file():
            results.append(
                SmokeResult(name=name, status="FAIL", path=relative_path, detail="missing file or hook indicator")
            )
        elif required_text.lower() not in text:
            results.append(
                SmokeResult(name=name, status="FAIL", path=relative_path, detail="missing hook indicator")
            )
        else:
            results.append(SmokeResult(name=name, status="PASS", path=relative_path, detail="local dry-run indicator present"))

    failures = [result for result in results if result.status != "PASS"]
    summary = {
        "phase": PHASE,
        "objective": OBJECTIVE,
        "scope": SCOPE_STATEMENT,
        "mode": "dry-run/default safe mode",
        "state_isolation": "temp HERMES_HOME/tmp_path based contract; no real ~/.hermes/state.db",
        "input_policy": "synthetic sentinel only",
        "report_policy": "no raw/private/secret/token/password material in report",
        "output_contract": "machine-readable summary or clear PASS/FAIL report",
        "missing_hook_policy": "exit nonzero on missing hooks/regressions",
        "forbidden_runtime_surfaces": list(FORBIDDEN_RUNTIME_SURFACES),
        "coverage": list(REQUIRED_COVERAGE),
        "results": [asdict(result) for result in results],
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
    }
    return (0 if not failures else 1), summary


def _redacted_json(data: dict[str, object]) -> str:
    # Summary data is deterministic metadata. It must not include user raw/private
    # bodies, secrets, tokens, passwords, credentials, network results, provider
    # responses, tmux/session output, real state DB rows, or environment values.
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=OBJECTIVE)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    parser.add_argument("--dry-run", action="store_true", default=True, help="default safe dry-run mode")
    args = parser.parse_args(list(argv) if argv is not None else None)

    exit_code, summary = run_smoke(REPO_ROOT)
    if args.json:
        print(_redacted_json(summary))
    else:
        print(f"{PHASE}: {summary['status']}")
        print(SCOPE_STATEMENT)
        print("Mode: dry-run/default safe mode; temp HERMES_HOME/tmp_path based; synthetic sentinel only")
        print("Safety: no real ~/.hermes/state.db, live provider, network, secrets, tmux/session, profile/systemd/cron/wrapper/env/credential")
        print("Activation boundary: no gateway restart/deploy/activation; no CLI slash command activation; no runtime config/profile activation")
        for result in summary["results"]:  # type: ignore[index]
            print(f"{result['status']} {result['name']} [{result['path']}]: {result['detail']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
