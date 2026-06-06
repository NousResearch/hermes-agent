#!/usr/bin/env python3
"""Run pytest with Hermes-friendly resource guardrails.

This wrapper is intentionally conservative for small always-on Hermes hosts:
- preflight disk space and clean known pytest/temp caches before running
- use a bounded basetemp under /tmp and remove it on exit
- disable pytest's cacheprovider so full disks do not fail during cache writes
- fail fast by default and cap xdist workers unless explicitly requested
- kill the whole pytest process group on timeout

Usage:
    ./venv/bin/python scripts/hermes_safe_pytest.py tests/agent/test_curator.py -q
    HERMES_SAFE_PYTEST_TIMEOUT=900 ./venv/bin/python scripts/hermes_safe_pytest.py tests/ -q
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_MIN_FREE_GB = 8.0
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAXFAIL = "1"
DEFAULT_TB = "short"


@dataclass(frozen=True)
class PytestPlan:
    command: list[str]
    basetemp: Path
    timeout_seconds: int
    keep_tmp: bool


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _has_option(args: Sequence[str], *names: str) -> bool:
    prefixes = tuple(name + "=" for name in names)
    for arg in args:
        if arg in names or arg.startswith(prefixes):
            return True
    return False


def _option_value(args: Sequence[str], *names: str) -> str | None:
    prefixes = tuple(name + "=" for name in names)
    for i, arg in enumerate(args):
        if arg in names and i + 1 < len(args):
            return args[i + 1]
        for name in prefixes:
            if arg.startswith(name):
                return arg.split("=", 1)[1]
    return None


def _remove_option_with_value(args: Sequence[str], *names: str) -> list[str]:
    result: list[str] = []
    skip_next = False
    prefixes = tuple(name + "=" for name in names)
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in names:
            skip_next = True
            continue
        if arg.startswith(prefixes):
            continue
        result.append(arg)
    return result


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def cleanup_known_pytest_artifacts(tmp_root: Path = Path("/tmp"), *, max_age_seconds: int = 0) -> int:
    """Remove known pytest/node temp artifacts and return estimated bytes removed."""
    now = time.time()
    patterns = (
        "pytest-of-*",
        "pytest-*",
        "hermes-pytest-*",
        "node-compile-cache",
    )
    removed = 0
    for pattern in patterns:
        for path in tmp_root.glob(pattern):
            try:
                if max_age_seconds > 0:
                    age = now - path.stat().st_mtime
                    if age < max_age_seconds:
                        continue
                if path.is_symlink() or path.is_file():
                    size = path.lstat().st_size
                    path.unlink()
                    removed += size
                elif path.is_dir():
                    size = _tree_size(path)
                    shutil.rmtree(path, ignore_errors=True)
                    removed += size
            except OSError:
                continue
    return removed


def _tree_size(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for filename in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, filename)).st_size
            except OSError:
                pass
    return total


def _cap_workers(args: Sequence[str]) -> list[str]:
    """Cap xdist workers for small hosts while preserving explicit disable."""
    configured = os.getenv("HERMES_SAFE_PYTEST_WORKERS")
    if configured is None:
        # Default for the 2-core/~2GB Hermes VPS: keep xdist off unless the
        # caller opts in. This avoids OOM/SIGTERM cascades and orphan workers.
        configured = "0"

    args = list(args)
    if configured.lower() in {"keep", "preserve"}:
        return args

    try:
        desired = max(0, int(configured))
    except ValueError:
        desired = 0

    args = _remove_option_with_value(args, "-n", "--numprocesses")
    if desired > 0:
        args.extend(["-n", str(desired), "--dist", "loadgroup"])
    else:
        args.extend(["-n", "0"])
    return args


def _pytest_python() -> str:
    configured = os.getenv("HERMES_SAFE_PYTEST_PYTHON")
    if configured:
        return configured
    repo_venv = Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"
    if repo_venv.exists():
        return str(repo_venv)
    return sys.executable


def build_pytest_plan(pytest_args: Sequence[str], *, tmp_root: Path = Path("/tmp")) -> PytestPlan:
    args = list(pytest_args)
    timeout_seconds = _env_int("HERMES_SAFE_PYTEST_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
    keep_tmp = os.getenv("HERMES_SAFE_PYTEST_KEEP_TMP", "").lower() in {"1", "true", "yes"}

    basetemp_value = _option_value(args, "--basetemp")
    if basetemp_value:
        basetemp = Path(basetemp_value)
    else:
        basetemp = Path(tempfile.mkdtemp(prefix="hermes-pytest-", dir=str(tmp_root)))
        args.append(f"--basetemp={basetemp}")

    if not _has_option(args, "-o") and not any(
        a.startswith("-oaddopts=") or a.startswith("-o=addopts=") for a in args
    ):
        args.extend(["-o", "addopts="])

    if not _has_option(args, "-p") and "no:cacheprovider" not in " ".join(args):
        args.extend(["-p", "no:cacheprovider"])

    maxfail = os.getenv("HERMES_SAFE_PYTEST_MAXFAIL", DEFAULT_MAXFAIL)
    if maxfail and maxfail != "0" and not _has_option(args, "--maxfail", "-x"):
        args.append(f"--maxfail={maxfail}")

    if not _has_option(args, "--tb"):
        args.append(f"--tb={os.getenv('HERMES_SAFE_PYTEST_TB', DEFAULT_TB)}")

    args = _cap_workers(args)

    return PytestPlan(
        command=[_pytest_python(), "-m", "pytest", *args],
        basetemp=basetemp,
        timeout_seconds=timeout_seconds,
        keep_tmp=keep_tmp,
    )


def preflight_disk(min_free_gb: float, *, path: Path = Path("/"), tmp_root: Path = Path("/tmp")) -> None:
    if _free_gb(path) >= min_free_gb:
        return
    removed = cleanup_known_pytest_artifacts(tmp_root)
    if removed:
        print(
            f"hermes-safe-pytest: cleaned {removed / (1024**2):.1f} MiB of pytest temp before run",
            file=sys.stderr,
        )
    free_after = _free_gb(path)
    if free_after < min_free_gb:
        raise SystemExit(
            f"hermes-safe-pytest: refusing to run with only {free_after:.1f} GiB free "
            f"(< {min_free_gb:.1f} GiB). Free disk first; raw pytest would likely fail with ENOSPC."
        )


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def run_pytest(plan: PytestPlan) -> int:
    proc = subprocess.Popen(
        plan.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=plan.timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        output, _ = proc.communicate()
        sys.stdout.buffer.write(output)
        print(
            f"\nhermes-safe-pytest: timed out after {plan.timeout_seconds}s; killed pytest process group",
            file=sys.stderr,
        )
        return 124
    finally:
        if not plan.keep_tmp and plan.basetemp.exists():
            shutil.rmtree(plan.basetemp, ignore_errors=True)

    sys.stdout.buffer.write(output)
    return int(proc.returncode or 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-safe pytest runner")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)

    pytest_args = list(ns.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    min_free_gb = _env_float("HERMES_SAFE_PYTEST_MIN_FREE_GB", DEFAULT_MIN_FREE_GB)
    preflight_disk(min_free_gb)
    plan = build_pytest_plan(pytest_args)
    print("hermes-safe-pytest:", " ".join(plan.command), file=sys.stderr)
    rc = run_pytest(plan)
    cleanup_known_pytest_artifacts(Path("/tmp"), max_age_seconds=6 * 60 * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
