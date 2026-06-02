#!/usr/bin/env python3
"""Safe operational smoke runner for default/redops Hermes profiles.

The live Hermes runtime logs every non-zero terminal tool invocation as a tool
error.  That is useful for real work, but noisy for ops smoke checks: a failed
sub-check can become a fresh WARNING/CRITICAL in the same log we are auditing.

This runner captures each sub-check's rc/stdout/stderr and returns a structured
JSON report.  By default it exits 0 even when a sub-check fails, so agents can
inspect ``ok`` without polluting runtime logs with a terminal-tool failure.  Use
``--strict`` in CI or a human shell when the process exit code should reflect
failure.

The default check set is deliberately doctor-free: ``hermes doctor`` currently
prints a global profile roster, which can enumerate unrelated profiles.  Keep
this script profile-scoped unless a human explicitly passes ``--include-doctor``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - startup failure is reported in main()
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/ameobius/projects/security-workstation/.hermes"))
PROFILES = {
    "default": HERMES_HOME,
    "redops": HERMES_HOME / "profiles" / "redops",
}
AUX_TASKS = [
    "compression",
    "vision",
    "web_extract",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "goal_judge",
    "session_search",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "curator",
]
EXPECTED_AUX = {
    "provider": "custom",
    "base_url": "http://127.0.0.1:8317/v1",
    "model": "gemini-3.1-flash-lite-preview",
}


def _run(name: str, argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 180) -> dict[str, Any]:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    try:
        cp = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=proc_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "name": name,
            "ok": cp.returncode == 0,
            "rc": cp.returncode,
            "stdout_tail": cp.stdout[-4000:],
            "stderr_tail": cp.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "rc": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_aux_config() -> dict[str, Any]:
    if yaml is None:
        return {"name": "aux_config", "ok": False, "error": "PyYAML unavailable"}
    bad: dict[str, list[dict[str, Any]]] = {}
    for profile, root in PROFILES.items():
        cfg_path = root / "config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            bad.setdefault(profile, []).append({"task": "<parse>", "error": str(exc)})
            continue
        aux = cfg.get("auxiliary") or {}
        for task in AUX_TASKS:
            item = aux.get(task) or {}
            mismatch = {
                key: item.get(key)
                for key, expected in EXPECTED_AUX.items()
                if item.get(key) != expected
            }
            if mismatch or not item.get("api_key"):
                bad.setdefault(profile, []).append(
                    {
                        "task": task,
                        "mismatch": mismatch,
                        "api_key_set": bool(item.get("api_key")),
                    }
                )
    return {"name": "aux_config", "ok": not bad, "bad": bad}


def _check_checkpoint_store() -> dict[str, Any]:
    code = """
from pathlib import Path
from tools.checkpoint_manager import CHECKPOINT_BASE, _is_valid_store, _run_git
store = CHECKPOINT_BASE / 'store'
workdir = Path.cwd()
result = {'store': str(store), 'exists': store.exists(), 'valid': _is_valid_store(store)}
if store.exists():
    rc, out, err = _run_git(['rev-parse', '--git-dir'], store, working_dir=workdir)
    result.update({'git_ok': bool(rc), 'git_dir': out.strip(), 'git_err': err.strip()[:300]})
print(result)
raise SystemExit(0 if result.get('exists') and result.get('valid') and result.get('git_ok') else 1)
""".strip()
    return _run(
        "checkpoint_store",
        [sys.executable, "-c", code],
        env={"PYTHONPATH": "."},
        timeout=60,
    )


def _doctor(profile: str) -> dict[str, Any]:
    return _run(
        f"doctor_{profile}",
        [sys.executable, "-m", "hermes_cli.main", "-p", profile, "doctor"],
        env={"PYTHONPATH": ".", "HERMES_CLI_EXPLICIT_PROFILE": profile},
        timeout=180,
    )


def _git_diff_check() -> dict[str, Any]:
    paths = [
        "agent/tool_executor.py",
        "tests/agent/test_tool_executor_log_preview.py",
        "agent/auxiliary_client.py",
        "tests/agent/test_auxiliary_client.py",
        "hermes_logging.py",
        "tests/test_hermes_logging.py",
        "tools/checkpoint_manager.py",
        "tests/tools/test_checkpoint_manager.py",
        "scripts/hermes_ops_smoke.py",
        "tests/scripts/test_hermes_ops_smoke.py",
    ]
    existing = [p for p in paths if (REPO_ROOT / p).exists()]
    return _run("git_diff_check", ["git", "diff", "--check", "--", *existing], timeout=60)


def _pytest_targeted() -> dict[str, Any]:
    return _run(
        "pytest_targeted",
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "tests/agent/test_tool_executor_log_preview.py",
            "tests/agent/test_auxiliary_client.py::TestAuxUnhealthyCache",
            "tests/agent/test_auxiliary_main_first.py",
            "tests/test_hermes_logging.py",
            "tests/tools/test_checkpoint_manager.py",
            "tests/scripts/test_hermes_ops_smoke.py",
            "tests/run_agent/test_run_agent.py::TestExecuteToolCalls",
            "-q",
        ],
        env={"PYTEST_ADDOPTS": "", "PYTHONPATH": "."},
        timeout=600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any sub-check fails")
    parser.add_argument("--skip-pytest", action="store_true", help="skip targeted pytest suite")
    parser.add_argument(
        "--include-doctor",
        action="store_true",
        help=(
            "also run hermes doctor for default/redops; disabled by default "
            "because doctor prints the global profile roster"
        ),
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    checks.append(_check_aux_config())
    checks.append(_check_checkpoint_store())
    if args.include_doctor:
        checks.append(_doctor("default"))
        checks.append(_doctor("redops"))
    checks.append(_git_diff_check())
    if not args.skip_pytest:
        checks.append(_pytest_targeted())

    ok = all(item.get("ok") for item in checks)
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
