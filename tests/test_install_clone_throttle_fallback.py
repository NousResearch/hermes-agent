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
  partial clone (`--filter=blob:none`) and materializes the working tree
  with `git reset --hard HEAD` — many small packs instead of one big one,
  which is what gets past the throttle.
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
    assert re.search(r"for attempt in 1 2 3 4", branch), (
        "the HTTPS clone must be retried a bounded number of times"
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
    assert re.search(r"git reset --hard HEAD", branch), (
        "the partial clone's working tree must be materialized so the rest "
        "of the installer sees the normal files"
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
