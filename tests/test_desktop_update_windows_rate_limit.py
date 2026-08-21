"""Regression: a GitHub rate-limit (HTTP 429) must abort the update, not retry it.

`scripts/desktop-update/windows.ps1` retries `hermes update` once when the first
attempt exits non-zero and non-2. That retry exists for the update-boundary
class: the fetch lands fresh code on disk while the already-loaded Python still
holds the stale module, so running the same command a second time picks up the
fix. It is the right response to that failure.

It is the wrong response to HTTP 429. GitHub's secondary rate limit is driven by
request *frequency*, so an immediate second fetch extends the cooldown instead
of clearing it -- the retry is not merely useless there, it actively delays
recovery. There is also nothing on disk for the retry to reload: the failure is
server-side, and the checkout is untouched.

Observed on a Windows install (2026-08-18): the hand-off fetched, took
`error: RPC failed; HTTP 429`, logged "first attempt failed; retrying once", and
immediately re-fetched into the same limit. Both attempts failed and the
cooldown was pushed further out. The checkout was already at origin/main, so the
update had nothing to pull in the first place.

This test is source-level because Linux CI cannot execute the PowerShell
hand-off (same constraint as test_desktop_update_windows_python_handoff.py). The
invariant it guards: a rate-limit detection gate must sit between the first
attempt's exit-code check and the retry invocation, and the pattern it matches
must cover the phrasings git and GitHub actually emit without swallowing
unrelated failures that legitimately deserve the retry.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"

# The exact phrasings git/GitHub emit when the fetch is throttled. Sources:
# git's curl transport ("RPC failed; HTTP 429"), git's bare remote-rejection
# line ("The requested URL returned error: 429"), GitHub's remote: banner
# ("rate-limited due to too many requests"), and the REST-style wording
# ("secondary rate limit").
RATE_LIMIT_SAMPLES = (
    "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429",
    "fatal: unable to access 'https://github.com/x/y.git/': The requested URL returned error: 429",
    "remote: This request was rate-limited due to too many requests. Reduce the frequency",
    "You have exceeded a secondary rate limit. Please wait a few minutes",
)

# Failures that must still reach the retry, plus a decoy that merely contains
# the digits 429. Matching any of these would suppress a legitimate retry.
NON_RATE_LIMIT_SAMPLES = (
    "ModuleNotFoundError: No module named 'hermes_cli'",
    "Desktop build failed",
    "error: RPC failed; HTTP 500 curl 22",
    "copied 429 files successfully",
)


def _read() -> str:
    return WINDOWS_PS1.read_text(encoding="utf-8")


def _rate_limit_patterns(source: str) -> list[str]:
    """Every regex the script matches $res.Output against to detect throttling."""
    return re.findall(r'\$res\.Output\s+-match\s+"([^"]*429[^"]*)"', source)


def test_rate_limit_gate_exists() -> None:
    source = _read()

    patterns = _rate_limit_patterns(source)
    assert patterns, (
        "scripts/desktop-update/windows.ps1 must test $res.Output for a GitHub "
        "rate-limit before retrying the update. Without that gate a 429 is "
        "retried immediately, which extends GitHub's secondary-limit cooldown "
        "instead of clearing it."
    )


def test_rate_limit_gate_precedes_the_retry() -> None:
    """The gate is only useful if it can short-circuit before the second fetch."""
    source = _read()

    retry_marker = "first attempt failed; retrying once"
    retry_at = source.find(retry_marker)
    assert retry_at != -1, (
        "Expected the retry log line in scripts/desktop-update/windows.ps1; the "
        "hand-off retry structure changed -- update this guard."
    )

    gate_positions = [
        m.start()
        for m in re.finditer(r'\$res\.Output\s+-match\s+"[^"]*429[^"]*"', source)
    ]
    assert any(pos < retry_at for pos in gate_positions), (
        "The rate-limit gate must appear BEFORE the retry invocation so a 429 "
        "aborts instead of triggering a second doomed fetch."
    )


def test_rate_limit_gate_also_guards_the_retry_result() -> None:
    """A retry taken for some other reason can itself land in the limit."""
    source = _read()

    retry_at = source.find("first attempt failed; retrying once")
    gate_positions = [
        m.start()
        for m in re.finditer(r'\$res\.Output\s+-match\s+"[^"]*429[^"]*"', source)
    ]
    assert any(pos > retry_at for pos in gate_positions), (
        "A 429 on the retry itself must also be reported as a rate-limit skip "
        "rather than falling through to the generic failure path, so the user "
        "is told to wait instead of re-running immediately."
    )


def test_pattern_matches_real_rate_limit_output() -> None:
    source = _read()
    patterns = _rate_limit_patterns(source)

    for sample in RATE_LIMIT_SAMPLES:
        assert any(re.search(p, sample) for p in patterns), (
            "No rate-limit pattern in scripts/desktop-update/windows.ps1 matches "
            f"this real throttled-fetch output:\n  {sample}\n"
            f"Patterns present: {patterns}"
        )


def test_pattern_does_not_swallow_retryable_failures() -> None:
    source = _read()
    patterns = _rate_limit_patterns(source)

    for sample in NON_RATE_LIMIT_SAMPLES:
        offenders = [p for p in patterns if re.search(p, sample)]
        assert not offenders, (
            "A rate-limit pattern in scripts/desktop-update/windows.ps1 matches "
            f"output that is NOT a rate limit:\n  {sample}\n"
            f"Offending pattern(s): {offenders}. This would suppress a "
            "legitimate retry (or mislabel a real failure as throttling)."
        )
