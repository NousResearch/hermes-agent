"""The three credit gates: validity -> activation -> credit.

A patch proposal must pass all three, in order, to earn credit:

1. **validity** — the diff is non-empty, applies cleanly to the base tree,
   and every touched Python file still parses. Cheap, mechanical, no tests.
2. **activation** — the claimed failure reproduces on the base tree (red)
   and the patched tree fixes it (green), using the patch's own regression
   test. This is the gate that separates "a patch" from "a fix".
3. **credit** — on a small paired battery, the patched tree is strictly
   better than baseline with statistical confidence. Uses a paired
   bootstrap confidence interval (tighter than a t-test on small batteries;
   no normality assumption). Fails closed on insufficient data.

Gates are pure logic over injected seams (a repo path, a ``run_test``
callable, score lists) so they are unit-testable without git or pytest.
"""
from __future__ import annotations

import ast
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Gate identifiers, in the order they run.
VALIDITY = "validity"
ACTIVATION = "activation"
CREDIT = "credit"

# Credit-gate defaults (pre-registered; changing them changes the gate).
CREDIT_ALPHA = 0.05
CREDIT_N_BOOT = 10000
CREDIT_SEED = 20260914
CREDIT_MIN_PAIRS = 5


@dataclass(frozen=True)
class Patch:
    """A proposed change: unified diff with repo-root-relative paths."""
    diff: str
    base_sha: str = ""
    description: str = ""


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    detail: dict = field(default_factory=dict)
    duration_s: float = 0.0


def _touched_py_files(diff: str) -> list:
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path.endswith(".py") and path != "/dev/null":
                files.append(path)
    return files


def _git(repo: Path, args: list) -> str | None:
    """Run a git plumbing command; return stripped stdout, or None on failure."""
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       cwd=repo, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


def validity_gate(patch: Patch, repo: str | Path) -> GateResult:
    """Gate 1: the patch is well-formed and safe to attempt.

    Checks, in order: non-empty diff; when ``patch.base_sha`` is set, that
    the repo's HEAD is that commit (a diff checked against the wrong tree
    is not a measurement — fail closed); ``git apply --check`` against the
    base tree; every touched ``.py`` file parses after the patch is applied
    (in a scratch worktree, so the caller's tree is never modified).
    """
    t0 = time.monotonic()

    def done(passed: bool, **detail):
        return GateResult(VALIDITY, passed, detail, time.monotonic() - t0)

    if not patch.diff or not patch.diff.strip():
        return done(False, reason="empty_diff")
    repo = Path(repo)
    # 0. The diff must be against the tree we are checking it on.
    if patch.base_sha:
        head = _git(repo, ["rev-parse", "HEAD"])
        want = _git(repo, ["rev-parse", "--verify",
                           f"{patch.base_sha}^{{commit}}"])
        if head is None or want is None or head != want:
            return done(False, reason="base_mismatch",
                        detail_note="repo HEAD does not match patch.base_sha",
                        head=(head or "?")[:12],
                        base_sha=patch.base_sha[:12])
    # 1. Applies cleanly to the base tree?
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch.diff, capture_output=True, text=True,
        cwd=repo, timeout=60)
    if check.returncode != 0:
        return done(False, reason="does_not_apply",
                    stderr=check.stderr.strip()[:500])
    # 2. Touched Python files still parse? Apply in a scratch worktree so the
    #    caller's checkout is untouched.
    py_files = _touched_py_files(patch.diff)
    if not py_files:
        return done(True, reason="no_python_files_touched",
                    files_touched=0)
    with tempfile.TemporaryDirectory(prefix="evolver-validity-") as tmp:
        scratch = Path(tmp) / "wt"
        add = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach",
             str(scratch), "HEAD"],
            capture_output=True, text=True, timeout=120)
        if add.returncode != 0:
            return done(False, reason="worktree_failed",
                        stderr=add.stderr.strip()[:300])
        try:
            apply = subprocess.run(
                ["git", "apply", "-"], input=patch.diff,
                capture_output=True, text=True, cwd=scratch, timeout=60)
            if apply.returncode != 0:
                return done(False, reason="apply_failed_in_scratch",
                            stderr=apply.stderr.strip()[:300])
            broken = []
            for rel in py_files:
                target = scratch / rel
                if not target.exists():
                    continue  # deleted file: nothing to parse
                try:
                    ast.parse(target.read_text(encoding="utf-8"),
                              filename=rel)
                except (SyntaxError, ValueError) as e:
                    broken.append(f"{rel}: {e}")
            if broken:
                return done(False, reason="syntax_error", files=broken)
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force",
                 str(scratch)],
                capture_output=True, timeout=120)
    return done(True, reason="applies_and_parses",
                python_files_checked=len(py_files))


def activation_gate(patch: Patch, run_test, test_ref: str,
                    base_checkout: str | Path,
                    patched_checkout: str | Path) -> GateResult:
    """Gate 2: the failure reproduces on base (red) and the patch fixes it.

    ``run_test(checkout, test_ref)`` returns True when the regression test
    passes. Contract on the caller: ``patched_checkout`` must actually have
    the patch applied, and ``run_test`` must be hermetic in the checkout
    (same test, same environment, both trees) — otherwise the comparison
    is meaningless. Both directions are required: a test that passes on
    base proves nothing, and a patch that leaves it red fixes nothing.
    """
    t0 = time.monotonic()

    def done(passed: bool, **detail):
        return GateResult(ACTIVATION, passed, detail, time.monotonic() - t0)

    try:
        red = not run_test(base_checkout, test_ref)
    except Exception as e:  # noqa: BLE001 — a crashing repro is data
        return done(False, reason="base_run_crashed", error=f"{type(e).__name__}: {e}")
    if not red:
        return done(False, reason="no_repro_on_base",
                    detail_note="regression test already passes without the patch")
    try:
        green = run_test(patched_checkout, test_ref)
    except Exception as e:  # noqa: BLE001
        return done(False, reason="patched_run_crashed",
                    error=f"{type(e).__name__}: {e}", red_on_base=True)
    if not green:
        return done(False, reason="not_fixed", red_on_base=True,
                    detail_note="test still fails with the patch applied")
    return done(True, reason="red_on_base_green_on_patched",
                red_on_base=True, green_on_patched=True)


def paired_bootstrap_ci(baseline: list, patched: list,
                        alpha: float = CREDIT_ALPHA,
                        n_boot: int = CREDIT_N_BOOT,
                        seed: int = CREDIT_SEED) -> dict:
    """Paired bootstrap CI for the mean of (patched - baseline).

    Deterministic given ``seed``. Returns dict with mean_delta, ci_lo,
    ci_hi, n, n_boot. Raises ValueError on insufficient or unpaired data.
    """
    if len(baseline) != len(patched):
        raise ValueError("baseline and patched must be paired (equal length)")
    n = len(baseline)
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")
    deltas = [p - b for b, p in zip(baseline, patched)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {
        "mean_delta": sum(deltas) / n,
        "ci_lo": means[lo_idx],
        "ci_hi": means[hi_idx],
        "n": n,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def credit_gate(baseline: list, patched: list,
                alpha: float = CREDIT_ALPHA,
                n_boot: int = CREDIT_N_BOOT) -> GateResult:
    """Gate 3: strictly positive lift on a paired battery, with confidence.

    Credit is granted iff the lower bound of the (1-alpha) bootstrap CI on
    the mean paired delta is above zero. Fails closed on insufficient data:
    fewer than CREDIT_MIN_PAIRS pairs is not a measurement, it is a guess.
    """
    t0 = time.monotonic()

    def done(passed: bool, **detail):
        return GateResult(CREDIT, passed, detail, time.monotonic() - t0)

    if len(baseline) != len(patched) or len(baseline) < CREDIT_MIN_PAIRS:
        return done(False, reason="insufficient_data",
                    detail_note=f"need >={CREDIT_MIN_PAIRS} paired scores, "
                                f"got {len(baseline)}",
                    n_pairs=len(baseline))
    try:
        stats = paired_bootstrap_ci(baseline, patched, alpha=alpha,
                                    n_boot=n_boot)
    except ValueError as e:
        return done(False, reason="insufficient_data", error=str(e))
    if stats["ci_lo"] > 0:
        return done(True, reason="positive_lift_credited", **stats)
    return done(False, reason="no_significant_lift", **stats)
