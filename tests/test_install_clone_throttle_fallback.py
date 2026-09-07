"""Fresh-install clone throttle handling (#89624).

GitHub throttles packfile generation for this repo with repo-scoped HTTP
429s (not client IP limits): the single big pack behind `--depth 1` dies
mid-transfer with "RPC failed; HTTP 429 / expected 'packfile'", and
clone_repo's HTTPS branch had no retry and no fallback — a fresh install
on an ordinary unauthenticated machine exited 1 at the download stage
(same throttle as the update path in #89287).

The contract pinned here:
- The HTTPS clone is retried with backoff before giving up.
- A failed direct attempt is retried after removing the partial clone.
- When every direct attempt fails, the installer degrades to a blobless
  partial clone (`--filter=blob:none --no-checkout`) and materializes the
  working tree with `git reset --hard HEAD` — the clone itself is
  commits+trees only (small, passes the throttle) and the reset becomes
  the separate blob fetch the retry can wrap (review of #89629: without
  --no-checkout the blob fetch runs inside `git clone`'s own checkout,
  so the throttle kills the whole clone and the fallback degrades to one
  more failed clone).
- Materialization fails closed: both reset attempts failing must remove
  the checkout and report a clone failure, never report success over an
  unusable tree.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _https_branch() -> str:
    text = INSTALL_SH.read_text()
    m = re.search(
        r"log_info \"SSH failed, trying HTTPS\.\.\..*?(?=\n    fi\n)",
        text,
        re.DOTALL,
    )
    assert m is not None, "HTTPS clone branch not found in install.sh"
    return m.group(0)


def test_https_clone_is_retried_with_backoff():
    branch = _https_branch()
    assert re.search(r"for attempt in \$\(seq 1 \"\$max_attempts\"\)", branch), (
        "the HTTPS clone must be retried a bounded number of times, with the "
        "loop bound driven by the same variable the messages report"
    )
    assert re.search(r"sleep \$\(\(attempt \* 5\)\)", branch), (
        "retries must back off between attempts"
    )
    # A failed direct attempt leaves a partial clone; it must be removed
    # before the next attempt or git refuses to clone into a non-empty dir.
    assert re.search(
        r"rm -rf \"\$INSTALL_DIR\" 2>/dev/null  # partial clone is unusable",
        branch,
    ), "each failed direct attempt must clean up the partial clone"


def test_blobless_partial_clone_fallback_exists():
    branch = _https_branch()
    assert "--filter=blob:none" in branch, (
        "after direct attempts fail, degrade to a blobless partial clone "
        "(many small packs — what gets past the repo-scoped 429)"
    )
    assert re.search(
        r"git clone --depth 1 --single-branch --filter=blob:none \\\n"
        r"\s*--no-checkout --branch \"\$BRANCH\"",
        branch,
    ), (
        "the partial clone must defer the checkout (--no-checkout): the blob "
        "fetch otherwise runs inside git clone's own checkout step, the "
        "throttle kills the whole clone, and the fallback never engages"
    )
    assert re.search(r"git reset --hard HEAD", branch), (
        "the partial clone's working tree must be materialized so the rest "
        "of the installer sees the normal files"
    )


def test_materialization_fails_closed():
    """A failed blob materialization must not report a successful clone.

    The reset on a --no-checkout clone is the step that fetches the blobs,
    so it is the step most likely to be throttled. `|| true` plus an
    unconditional `clone_ok=true` would hand the rest of the installer a
    half-materialized tree while printing "Cloned via HTTPS".
    """
    branch = _https_branch()
    fallback = branch.split('log_info "Direct clone throttled')[1]
    assert "|| true" not in fallback, (
        "the materialization retry must not swallow a hard failure"
    )
    m = re.search(
        r"if \(cd \"\$INSTALL_DIR\" \\\n"
        r"\s*&& \(git reset --hard HEAD",
        fallback,
    )
    assert m is not None, (
        "the reset must be guarded: its success is the condition that sets "
        "clone_ok, and a failed reset must clean up the checkout"
    )


def test_partial_clone_failure_still_cleans_up_and_exits():
    branch = _https_branch()
    m = re.search(
        r'if \[ "\$clone_ok" = true \]; then\n\s*log_success "Cloned via HTTPS"'
        r"\n\s*else\n\s*log_error \"Failed to clone repository\"\n\s*exit 1",
        branch,
    )
    assert m is not None, (
        "when the fallback also fails the installer must still report the "
        "failure and exit 1"
    )


def test_fallback_runs_only_after_all_direct_attempts_fail():
    branch = _https_branch()
    direct = branch.split('log_info "Direct clone throttled')[0]
    assert re.search(r"clone_ok != true|clone_ok\" != true", direct), (
        "the blobless fallback must be gated on every direct attempt having "
        "failed — a successful direct clone must never take the fallback path"
    )
