"""Run selected engineering-workflow mutations in a disposable source copy.

A mutant only counts when pytest collects and a behavioural assertion fails.
The source checkout is never mutated by this runner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUNNER_TEST = "tests/agent/test_engineering_runner.py"
ROUTER_TEST = "tests/agent/test_engineering_workflow_router.py"
MUTATIONS = (
    (
        "drop_reasoning_config", "agent/engineering_runner.py",
        (('{"reasoning_config": parse_reasoning_effort(route.reasoning_effort)}', "{}", 1),),
        RUNNER_TEST, "forwards_each_stage_reasoning",
    ),
    (
        "force_every_stage_high", "agent/engineering_runner.py",
        (('parse_reasoning_effort(route.reasoning_effort)', 'parse_reasoning_effort("high")', 1),),
        RUNNER_TEST, "forwards_each_stage_reasoning",
    ),
    (
        "swap_worker_reviewer_effort", "agent/engineering_runner.py",
        ((
            "parse_reasoning_effort(route.reasoning_effort)",
            'parse_reasoning_effort("high" if stage == "worker" else '
            '"medium" if stage == "reviewer" else route.reasoning_effort)',
            1,
        ),),
        RUNNER_TEST, "forwards_each_stage_reasoning",
    ),
    (
        "ignore_effort_only_drift", "agent/engineering_workflow.py",
        (("if assignments != admitted_assignments:", "if False:", 1),),
        ROUTER_TEST, "effort_only_assignment_change",
    ),
    (
        "allow_strict_route_fallback", "agent/engineering_runner.py",
        (("strict_route=True,", "strict_route=False,", 1),),
        RUNNER_TEST, "forwards_each_stage_reasoning",
    ),
    (
        "emit_raw_exception_text", "agent/engineering_workflow.py",
        ((
            'return finish("BLOCKED", error.code)',
            'return finish("BLOCKED", str(error.__context__))',
            1,
        ),),
        ROUTER_TEST, "stage_failures_return_distinct",
    ),
    (
        "collapse_failure_codes", "agent/engineering_workflow.py",
        (("error.code", '"host_stage_failure"', 2),),
        ROUTER_TEST, "stage_failures_return_distinct",
    ),
    (
        "retry_uncertain_worker", "agent/engineering_workflow.py",
        ((
            'except Exception:\n                raise StageBoundaryError("execution_boundary_failed") from None',
            'except Exception:\n'
            '                execute_worker(worker_text, precheck, '
            'lambda payload: call_stage("worker", payload), plan)\n'
            '                raise StageBoundaryError("execution_boundary_failed") from None',
            1,
        ),),
        ROUTER_TEST, "stage_failures_return_distinct",
    ),
)
COPY_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".sh"}
COLLECTION_ERRORS = (
    "ERROR collecting", "ImportError while importing", "ModuleNotFoundError",
    "SyntaxError", "errors during collection",
)


def copy_source(destination: Path) -> int:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    copied = 0
    for name in listed:
        if not name:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in COPY_SUFFIXES:
            continue
        source = ROOT / relative
        if not source.is_file() or source.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    return copied


def run_tests(source: Path, test_file: str, selection: str | None = None) -> subprocess.CompletedProcess:
    bash = Path(r"C:\Program Files\Git\bin\bash.exe") if os.name == "nt" else Path("/usr/bin/bash")
    if not bash.is_file():
        raise RuntimeError("Git Bash or /usr/bin/bash is required")
    env = os.environ.copy()
    env["HERMES_PYTHON"] = Path(sys.executable).resolve().as_posix()
    env["HERMES_TEST_WORKERS"] = "1"
    command = [str(bash), "scripts/run_tests.sh", test_file, "--file-retries=0", "-q"]
    if selection:
        command.extend(["-k", selection])
    return subprocess.run(
        command, cwd=source, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )


def passed(result: subprocess.CompletedProcess) -> bool:
    output = result.stdout + result.stderr
    return result.returncode == 0 and bool(re.search(r"\b\d+ tests? passed\b", output))


def detected(result: subprocess.CompletedProcess) -> bool:
    output = result.stdout + result.stderr
    return (
        result.returncode == 1
        and bool(re.search(r"\b\d+ failed\b", output))
        and not any(marker in output for marker in COLLECTION_ERRORS)
    )


def main() -> int:
    tests = sorted({spec[3] for spec in MUTATIONS})
    baseline = [run_tests(ROOT, test) for test in tests]
    if not all(passed(result) for result in baseline):
        print(json.dumps({"baseline_passed": False, "exit_codes": [r.returncode for r in baseline]}))
        return 2

    temp_base = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix="hm119964-mutations-", dir=temp_base)).resolve()
    if temporary.parent != temp_base or temporary.is_symlink():
        raise RuntimeError("Unsafe temporary source path")
    try:
        source = temporary / "source"
        source.mkdir()
        copied = copy_source(source)
        copy_baseline = [run_tests(source, test) for test in tests]
        if not all(passed(result) for result in copy_baseline):
            print(json.dumps({
                "baseline_passed": True, "copy_baseline_passed": False,
                "copied_files": copied,
                "exit_codes": [r.returncode for r in copy_baseline],
                "output_tail": [(r.stdout + r.stderr)[-1000:] for r in copy_baseline],
            }))
            return 2

        results = []
        for name, relative, edits, test, selection in MUTATIONS:
            target = source / relative
            original = target.read_text(encoding="utf-8")
            changed = original
            for before, after, expected_count in edits:
                if changed.count(before) != expected_count:
                    raise RuntimeError(f"{name}: reviewed source anchor changed")
                changed = changed.replace(before, after)
            try:
                compile(changed, relative, "exec")
                target.write_text(changed, encoding="utf-8")
                result = run_tests(source, test, selection)
                output = result.stdout + result.stderr
                results.append({
                    "mutation": name, "detected": detected(result),
                    "exit_code": result.returncode,
                    "failed_assertions": re.findall(r"(?m)^FAILED .+$", output)[:3],
                    "collection_error": any(marker in output for marker in COLLECTION_ERRORS),
                })
            except (SyntaxError, subprocess.TimeoutExpired) as error:
                results.append({"mutation": name, "detected": False, "error_type": type(error).__name__})
            finally:
                target.write_text(original, encoding="utf-8")
        total = sum(item["detected"] for item in results)
        print(json.dumps({
            "baseline_passed": True, "copy_baseline_passed": True,
            "copied_files": copied, "detected": total, "selected": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2))
        return 0 if total == len(results) else 1
    finally:
        if temporary.parent != temp_base or temporary.is_symlink():
            raise RuntimeError("Refusing unsafe mutation cleanup")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())