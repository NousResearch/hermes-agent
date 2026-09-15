"""Phase 0 calibration harness: do the gates separate good from bad?

Runs validity + activation over recent HUMAN-authored merged fix commits
(known-good) and over synthetic bad patches derived from them (known-bad):

- known-good: the merge diff itself. Validity must PASS; activation must
  PASS (regression test red on base, green on merge).
- known-bad "empty": an empty diff. Validity must FAIL.
- known-bad "syntax_break": a diff that applies but breaks Python parsing.
  Validity must FAIL.
- known-bad "noop": a diff that applies and parses but changes nothing
  behavioral (appended comment). Validity must PASS (it is well-formed) and
  activation must FAIL (the test stays red) — this is the case that proves
  activation discriminates fixes from non-fixes.

Credit is calibrated separately on synthetic paired data (see
``calibrate_credit``): a real LLM battery is out of scope for Phase 0.

Usage:
    python -m evolver.calibrate --repo ~/workspace/hermes/hermes-agent \\
        --workdir /tmp/evolver-calib --report /tmp/evolver-calib/report.json

Exit code 0 iff every expectation above holds (the kill criterion, measured).
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .gates import (ACTIVATION, CREDIT, Patch, GateResult, activation_gate,
                    credit_gate, paired_bootstrap_ci, validity_gate)

# Recent human-authored merged fixes, each with a regression test that is
# red on the parent commit and green on the merge commit. test_file may be a
# file added by the commit (new_test_file=True) or an existing file the
# commit extended (copied over the base checkout for the red run).
GOOD_CASES = [
    {
        "name": "wal-refusal-cross-vm-fs",
        "merge_sha": "d8dcdfd620",
        "test_file": "tests/hermes_state/test_cross_vm_fs_wal_refusal.py",
        "new_test_file": True,
    },
    {
        "name": "redact-widen-key-class",
        "merge_sha": "92d4c0233e",
        "test_file": "tests/agent/test_redact.py",
        "new_test_file": False,
    },
    {
        "name": "process-registry-handle-release",
        "merge_sha": "333733163e",
        "test_file": "tests/tools/test_process_registry.py",
        "new_test_file": False,
    },
    {
        "name": "redact-repr-already-masked",
        "merge_sha": "5280dec0ee",
        "test_file": "tests/agent/test_redact.py",
        "new_test_file": False,
    },
    {
        "name": "streamed-reasoning-details-replay",
        "merge_sha": "0ddb62bce6",
        "test_file": "tests/agent/test_streamed_reasoning_details.py",
        "new_test_file": True,
    },
]

TEST_TIMEOUT_S = 300


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def make_worktree(repo: Path, sha: str, dest: Path, venv_src: Path | None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _run(["git", "-C", str(repo), "worktree", "add", "--detach",
              str(dest), sha], timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add {sha} failed: {r.stderr[:300]}")
    # scripts/run_tests.sh probes .venv first; share the mirror's venv.
    if venv_src is not None:
        link = dest / ".venv"
        if not link.exists():
            link.symlink_to(venv_src)


def remove_worktree(repo: Path, dest: Path):
    _run(["git", "-C", str(repo), "worktree", "remove", "--force", str(dest)],
         timeout=120)
    shutil.rmtree(dest, ignore_errors=True)


def real_run_test(checkout: Path, test_file: str) -> bool:
    """Run one test file via scripts/run_tests.sh; True iff it passes."""
    runner = checkout / "scripts" / "run_tests.sh"
    if not runner.exists():
        raise RuntimeError(f"no scripts/run_tests.sh in {checkout}")
    r = _run(["bash", str(runner), test_file], cwd=checkout,
             timeout=TEST_TIMEOUT_S)
    return r.returncode == 0


def overlay_test_file(base_wt: Path, merge_wt: Path, test_file: str,
                      new_test_file: bool, merge_sha: str):
    """Place the merge's regression test into the base checkout."""
    src = merge_wt / test_file
    dst = base_wt / test_file
    if new_test_file:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        r = _run(["git", "-C", str(base_wt), "checkout", merge_sha, "--",
                  test_file], timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"test overlay failed: {r.stderr[:200]}")


def merge_diff(repo: Path, base_sha: str, merge_sha: str) -> str:
    r = _run(["git", "-C", str(repo), "diff", f"{base_sha}..{merge_sha}",
              "--", ".", ":(exclude)evolver/"],
             timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"diff failed: {r.stderr[:200]}")
    return r.stdout


def first_touched_py_source(repo: Path, base_sha: str, merge_sha: str) -> str | None:
    r = _run(["git", "-C", str(repo), "diff", "--name-only",
              f"{base_sha}..{merge_sha}"], timeout=60)
    for line in r.stdout.splitlines():
        p = line.strip()
        if p.endswith(".py") and not p.startswith("tests/"):
            return p
    return None


def synthetic_diff(base_content: str, extra: str, rel_path: str) -> str:
    """Build a unified diff appending ``extra`` to a base file's content."""
    old = base_content.splitlines(keepends=True)
    new = old + [extra if extra.endswith("\n") else extra + "\n"]
    return "".join(difflib.unified_diff(
        old, new, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}"))


def run_case(repo: Path, workdir: Path, venv_src: Path | None,
             case: dict) -> dict:
    """Run all gates for one good case + its synthetic bad variants."""
    name = case["name"]
    merge_sha = case["merge_sha"]
    base_sha = merge_sha + "^"
    test_file = case["test_file"]
    case_dir = workdir / name
    base_wt = case_dir / "base"
    merge_wt = case_dir / "merge"
    noop_wt = case_dir / "noop"
    out: dict = {"name": name, "merge_sha": merge_sha, "gates": {}}

    def rec(key: str, res: GateResult):
        out["gates"][key] = {"passed": res.passed,
                             "detail": res.detail,
                             "duration_s": round(res.duration_s, 1)}

    try:
        make_worktree(repo, base_sha, base_wt, venv_src)
        make_worktree(repo, merge_sha, merge_wt, venv_src)

        good_patch = Patch(diff=merge_diff(repo, base_sha, merge_sha),
                           base_sha=base_sha,
                           description=f"human fix {merge_sha}")
        # Validity runs on the PRISTINE base checkout: the test overlay below
        # mutates the working tree (new test files / post-image test files),
        # which would make the real diff fail `git apply --check`.
        rec("validity_good", validity_gate(good_patch, base_wt))

        overlay_test_file(base_wt, merge_wt, test_file,
                          case["new_test_file"], merge_sha)

        rec("activation_good",
            activation_gate(good_patch,
                            lambda co, _tf: real_run_test(co, test_file),
                            test_file, base_wt, merge_wt))

        # Known-bad: empty diff.
        rec("validity_empty",
            validity_gate(Patch(diff=""), base_wt))

        # Known-bad: applies cleanly but breaks Python parsing.
        src_rel = first_touched_py_source(repo, base_sha, merge_sha)
        if src_rel:
            base_src = (base_wt / src_rel).read_text(encoding="utf-8")
            broken = Patch(diff=synthetic_diff(base_src, "def broken(:\n",
                                              src_rel))
            rec("validity_syntax_break", validity_gate(broken, base_wt))
            # Known-bad: well-formed but behaviorally a no-op.
            noop_patch = Patch(diff=synthetic_diff(
                base_src, "\n# evolver-calibration: no-op\n", src_rel))
            rec("validity_noop", validity_gate(noop_patch, base_wt))
            make_worktree(repo, base_sha, noop_wt, venv_src)
            overlay_test_file(noop_wt, merge_wt, test_file,
                              case["new_test_file"], merge_sha)
            r = _run(["git", "apply", "-"], input=noop_patch.diff,
                     cwd=noop_wt, timeout=60)
            if r.returncode != 0:
                rec("activation_noop",
                    GateResult(ACTIVATION, False,
                               {"reason": "noop_patch_did_not_apply"}))
            else:
                rec("activation_noop",
                    activation_gate(
                        noop_patch,
                        lambda co, _tf: real_run_test(co, test_file),
                        test_file, base_wt, noop_wt))
        else:
            for k in ("validity_syntax_break", "validity_noop",
                      "activation_noop"):
                out["gates"][k] = {"passed": None,
                                   "detail": {"reason": "no_py_source"},
                                   "duration_s": 0.0}
    finally:
        for wt in (noop_wt, merge_wt, base_wt):
            if wt.exists():
                remove_worktree(repo, wt)
    return out


def calibrate_credit() -> dict:
    """Calibrate the credit gate on synthetic paired data (honestly labeled).

    A real LLM battery is out of scope for Phase 0; this validates the
    statistic's behavior: clear lift credits, noise does not, and thin data
    fails closed.
    """
    scenarios = {
        # Clearly better: +1 on every one of 8 pairs.
        "clear_lift": ([0.5] * 8, [1.5] * 8, True),
        # Pure noise around zero.
        "no_lift": ([0.0, 1.0] * 4, [1.0, 0.0] * 4, False),
        # Clearly worse.
        "negative_lift": ([1.5] * 8, [0.5] * 8, False),
        # Too few pairs: must fail closed, not guess.
        "thin_data": ([0.0, 0.0], [1.0, 1.0], False),
        # Unpaired input: must fail closed.
        "unpaired": ([0.0] * 5, [1.0] * 6, False),
    }
    results = {}
    for name, (base, patched, expect_pass) in scenarios.items():
        res = credit_gate(base, patched)
        results[name] = {
            "passed": res.passed,
            "expected": expect_pass,
            "correct": res.passed == expect_pass,
            "detail": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in res.detail.items()},
        }
    return results


def separation_summary(case_reports: list) -> dict:
    def rate(key, expect):
        vals = [c["gates"].get(key, {}).get("passed") for c in case_reports]
        vals = [v for v in vals if v is not None]
        return {"pass": sum(1 for v in vals if v),
                "total": len(vals),
                "expected_pass": expect,
                "as_expected": sum(1 for v in vals if v == expect)}

    return {
        "validity_good": rate("validity_good", True),
        "validity_empty": rate("validity_empty", False),
        "validity_syntax_break": rate("validity_syntax_break", False),
        "validity_noop": rate("validity_noop", True),  # well-formed by design
        "activation_good": rate("activation_good", True),
        "activation_noop": rate("activation_noop", False),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--venv", default=None,
                    help="venv for scratch worktrees' scripts/run_tests.sh")
    ap.add_argument("--only", default=None,
                    help="comma-separated case names to run")
    ap.add_argument("--skip-credit", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    venv_src = Path(args.venv).resolve() if args.venv else None

    only = set(args.only.split(",")) if args.only else None
    cases = [c for c in GOOD_CASES if not only or c["name"] in only]
    case_reports = []
    for c in cases:
        try:
            case_reports.append(run_case(repo, workdir, venv_src, c))
        except Exception as e:  # noqa: BLE001 — one bad case must not
            # kill the calibration; record the error honestly instead.
            case_reports.append({
                "name": c["name"], "merge_sha": c["merge_sha"],
                "error": f"{type(e).__name__}: {e}", "gates": {}})
    report = {
        "tool": "evolver.calibrate",
        "cases": case_reports,
        "separation": separation_summary(case_reports),
        "credit": calibrate_credit() if not args.skip_credit else "skipped",
    }
    # Kill criterion, measured: every gate expectation must hold.
    sep = report["separation"]
    ok = all(v["as_expected"] == v["total"] and v["total"] > 0
             for v in sep.values())
    if not args.skip_credit:
        ok = ok and all(v["correct"] for v in report["credit"].values())
    report["calibration_passed"] = ok

    Path(args.report).write_text(json.dumps(report, indent=2),
                                 encoding="utf-8")
    print(json.dumps({"calibration_passed": ok,
                      "separation": {k: f"{v['as_expected']}/{v['total']}"
                                     for k, v in sep.items()}},
                     indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
