#!/usr/bin/env python3
"""Fail when uv.lock is out of sync with pyproject.toml, telling a registry outage apart.

Run by the ``Verify uv.lock is up-to-date`` step of ``.github/workflows/repo-guards.yml``.
``python -m pm.build_env --source . --check-lock`` re-resolves against PyPI (network), so it is
retried: a registry blip must not read as "lockfile stale", while a genuinely stale lockfile
fails every attempt (deterministic) and costs only seconds. The backoff is 5/15/45s rather than a
flat 10s, because three attempts inside ~20s all land in the same blip.

uv says "Request failed after N retries" when it cannot reach the registry and "lockfile needs
to be updated" when the lock is stale. Only the second is the contributor's to fix, so an
unreachable registry says so instead of sending them to ``hermes pm lock`` with nothing to
regenerate. Writes the ``uv-lockfile`` review status (scripts/ci/review_status.py).

An attempt that outlives ``CHECK_TIMEOUT`` is stopped, its partial log is still printed, and it
reports a timeout without retrying: three more such attempts would outlive the job.

Exit codes: 0 = in sync, 1 = stale lockfile, registry unreachable, or timed out.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.review_status import action_required, publish

CHECK_COMMAND = ("python", "-m", "pm.build_env", "--source", ".", "--check-lock")
CHECK_TEXT = "python -m pm.build_env --source . --check-lock"
RETRY_DELAYS = (5, 15, 45)
ATTEMPTS = 3
# Under the job's 15-minute cap; the check itself takes seconds to a minute.
CHECK_TIMEOUT = 600
ARTIFACT = "uv-lockfile"
REGISTRY_UNREACHABLE = re.compile(
    r"Request failed after|Failed to fetch|error sending request|connection (reset|closed)|timed out",
    re.IGNORECASE,
)

STALE_SUMMARY = f"""\
## ❌ uv.lock is out of sync with pyproject.toml

**If this is a PR:** this check runs against the merged state
(your branch + current `main`), not just your branch.  If
`{CHECK_TEXT}` passes locally, your branch is likely behind
`main` — recent changes to `pyproject.toml` on `main` aren't
reflected in your branch's `uv.lock` yet.

To fix, sync with main and regenerate the lockfile:

```bash
git fetch origin main
git rebase origin/main   # or `git merge origin/main`
hermes pm lock                  # regenerate against new pyproject.toml
git add uv.lock
git commit -m "chore: refresh uv.lock after syncing with main"
git push --force-with-lease  # drop --force-with-lease if you merged
```

**If you only changed pyproject.toml:** run `hermes pm lock` locally
and commit the result.

This check is blocking because the Docker image build uses
`uv sync --frozen --extra all`, which rejects stale lockfiles
— catching it here avoids a ~15 min failed docker run
on `main` post-merge.
"""

UNREACHABLE_STATUS = action_required(
    "uv.lock check", "uv.lock check could not reach PyPI",
    f"`{CHECK_TEXT}` could not reach the package registry after 3 attempts. "
    "This is an infrastructure failure, not a stale lockfile.",
    "Re-run the failed job. No change to `uv.lock` is needed.",
)
TIMEOUT_STATUS = action_required(
    "uv.lock check", "uv.lock check timed out",
    f"`{CHECK_TEXT}` ran longer than {CHECK_TIMEOUT}s and was stopped. "
    "This is an infrastructure failure, not a stale lockfile.",
    "Re-run the failed job. No change to `uv.lock` is needed.",
)
STALE_STATUS = action_required(
    "uv.lock check", "uv.lock out of sync", "uv.lock is out of sync with pyproject.toml.",
    "Run `hermes pm lock` locally and commit the result. If on a PR, sync with main first:\n"
    "```\ngit fetch origin main\ngit rebase origin/main\nhermes pm lock\ngit add uv.lock\n"
    'git commit -m "chore: refresh uv.lock"\n```\n',
)


IN_SYNC, STALE, UNREACHABLE, TIMED_OUT = "in-sync", "stale", "unreachable", "timed-out"


def run_check(log: Path, timeout: float = CHECK_TIMEOUT) -> int | None:
    """One attempt's exit code, or None when it timed out; the output goes to ``log``, then stdout."""
    with log.open("wb") as fh:
        try:
            returncode = subprocess.run(CHECK_COMMAND, stdout=fh, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, timeout=timeout,
                                        check=False).returncode
        except subprocess.TimeoutExpired:
            returncode = None
    sys.stdout.flush()
    sys.stdout.buffer.write(log.read_bytes())
    sys.stdout.flush()
    return returncode


def verify(log: Path, sleep: Callable[[float], None] = time.sleep,
           timeout: float = CHECK_TIMEOUT) -> str:
    """IN_SYNC, STALE, UNREACHABLE or TIMED_OUT after up to three attempts."""
    for attempt in range(1, ATTEMPTS + 1):
        returncode = run_check(log, timeout)
        if returncode is None:
            return TIMED_OUT
        if returncode == 0:
            return IN_SYNC
        if not REGISTRY_UNREACHABLE.search(log.read_text(encoding="utf-8-sig", errors="replace")):
            # Deterministic failure (a real desync): retrying prints the same error twice more.
            return STALE
        if attempt == ATTEMPTS:
            break
        delay = RETRY_DELAYS[attempt - 1]
        print(f"::warning::{CHECK_TEXT} could not reach the registry (attempt {attempt}); "
              f"retrying in {delay}s", flush=True)
        sleep(delay)
    return UNREACHABLE


def main() -> int:
    runner_temp = Path(os.environ["RUNNER_TEMP"])
    verdict = verify(runner_temp / "lock-check.log")
    if verdict == TIMED_OUT:
        print(f"::error title=uv.lock check timed out::Stopped after {CHECK_TIMEOUT}s — "
              "infrastructure failure, not a stale lockfile. Re-run the job.")
        publish(TIMEOUT_STATUS, ARTIFACT)
        return 1
    if verdict == UNREACHABLE:
        print("::error title=uv.lock check could not reach PyPI::Registry unreachable after 3 "
              "attempts — infrastructure failure, not a stale lockfile. Re-run the job.")
        publish(UNREACHABLE_STATUS, ARTIFACT)
        return 1
    if verdict == STALE:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(STALE_SUMMARY)
        print("::error title=uv.lock out of sync::Run `hermes pm lock` locally and commit the "
              "result. If on a PR, sync with main first.")
        publish(STALE_STATUS, ARTIFACT)
        return 1
    publish([], ARTIFACT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
