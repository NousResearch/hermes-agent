"""Opt-in live baseline test for the prompt cache + cost pipeline.

Runs a real one-turn session against the `gemini35` profile and asserts
**shape invariants** on the resulting state.db row:

  - input_tokens > 0
  - output_tokens > 0
  - cost_status == 'estimated'
  - estimated_cost_usd > 0
  - billing_provider == 'gemini'

The exact token counts are NOT asserted — they drift with any system-prompt
or tool-definition change. The invariants catch the regressions that matter:
the cache + cost pipeline got disconnected (cost_status reverts to 'unknown'
or 'none') or token extraction stopped working (input_tokens == 0).

This test is **opt-in**. It costs real API quota, so it only runs when
`HERMES_LIVE_BASELINE=1` is set in the environment. Run it manually before
pushing the PR-1 refactor; do NOT enable in CI.

IMPORTANT: Run via direct `python -m pytest`, NOT `scripts/run_tests.sh`.
The shipped test runner uses `env -i` to enforce CI parity, which strips
`HERMES_LIVE_BASELINE` (and every other non-allowlisted env var). The
intentional design keeps credentials from leaking into the CI suite.

Setup (one-time):
    hermes profile create gemini35 --description "Cache testing profile"
    echo 'GOOGLE_API_KEY=...' > ~/.hermes/profiles/gemini35/.env
    cat > ~/.hermes/profiles/gemini35/config.yaml <<EOF
    model:
      provider: gemini
      default: gemini-3.5-flash
    EOF

Run:
    source .venv/bin/activate
    HERMES_LIVE_BASELINE=1 python -m pytest tests/integration/test_prompt_cache_baseline.py -v

Expected outcomes on `main`:
  - 4 token-related assertions pass (input_tokens, output_tokens,
    billing_provider, model)
  - 2 cost assertions FAIL until #32404 lands (cost_status=='estimated',
    estimated_cost_usd > 0). Those failures *document* the cost-tracking
    bug — they flip to green automatically once #32404 merges.
"""

import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest


PROFILE_NAME = os.environ.get("HERMES_BASELINE_PROFILE", "gemini35")
PROFILE_PATH = Path.home() / ".hermes" / "profiles" / PROFILE_NAME


def _profile_ready() -> bool:
    """Return True if the baseline profile is set up with an API key."""
    if not PROFILE_PATH.is_dir():
        return False
    env_file = PROFILE_PATH / ".env"
    if not env_file.is_file():
        return False
    try:
        env_text = env_file.read_text(encoding="utf-8")
    except OSError:
        return False
    return "GOOGLE_API_KEY=" in env_text and not env_text.strip().endswith("GOOGLE_API_KEY=")


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("HERMES_LIVE_BASELINE"),
        reason="opt-in; set HERMES_LIVE_BASELINE=1 to run (costs API quota)",
    ),
    pytest.mark.skipif(
        not _profile_ready(),
        reason=(
            f"profile not ready at {PROFILE_PATH} — see test module docstring "
            f"for setup. Skipping rather than failing so the test stays portable."
        ),
    ),
]


@pytest.fixture(scope="module")
def latest_session_row():
    """Fire a one-turn live request and return the resulting state.db row."""
    repo_root = Path(__file__).resolve().parents[2]
    hermes_bin = repo_root / ".venv" / "bin" / "hermes"
    if not hermes_bin.is_file():
        pytest.skip(f"venv hermes binary not found at {hermes_bin}")

    prompt = "Reply with exactly: 'baseline-prompt-cache-test'"

    env = {**os.environ, "HERMES_HOME": str(PROFILE_PATH)}

    # Record the current count of sessions so we can pick out the one we
    # added (rather than chasing 'most recent' which races with other shells).
    with sqlite3.connect(PROFILE_PATH / "state.db") as conn:
        before_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE input_tokens > 0"
        ).fetchone()[0]

    proc = subprocess.run(
        [str(hermes_bin), "chat", "-q", prompt],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.fail(f"hermes chat failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")

    # Wait briefly for the post-turn write to land. Hermes commits the
    # session row inside the turn finalizer; in practice this is sub-second.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with sqlite3.connect(PROFILE_PATH / "state.db") as conn:
            row = conn.execute(
                """
                SELECT id, model, input_tokens, output_tokens, cache_read_tokens,
                       estimated_cost_usd, billing_provider, cost_status
                FROM sessions
                WHERE input_tokens > 0
                ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
            after_count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE input_tokens > 0"
            ).fetchone()[0]
        if row and after_count > before_count:
            return row
        time.sleep(0.2)
    pytest.fail("session row did not appear in state.db within 5s after chat completed")


def test_baseline_input_tokens_nonzero(latest_session_row):
    """Token extraction is working — system prompt + tools land in input_tokens."""
    _, _, input_tokens, *_ = latest_session_row
    assert input_tokens > 0, "input_tokens should be > 0 after a live turn"


def test_baseline_output_tokens_nonzero(latest_session_row):
    """The model produced a reply and its token count was captured."""
    _, _, _, output_tokens, *_ = latest_session_row
    assert output_tokens > 0, "output_tokens should be > 0 after a live turn"


def test_baseline_cost_status_estimated(latest_session_row):
    """Pricing pipeline fires end-to-end (covers PR #32404 cost-tracking + this PR)."""
    *_, cost_status = latest_session_row
    assert cost_status == "estimated", (
        f"cost_status was {cost_status!r}, expected 'estimated' — "
        f"resolve_billing_route / pricing entry / cost estimator chain is broken"
    )


def test_baseline_estimated_cost_positive(latest_session_row):
    """The dollar amount is actually computed, not stored as 0.0."""
    *_, estimated_cost, _, _ = latest_session_row
    assert estimated_cost is not None and estimated_cost > 0, (
        f"estimated_cost_usd was {estimated_cost!r}, expected > 0"
    )


def test_baseline_billing_provider_is_gemini(latest_session_row):
    """The Gemini branch of resolve_billing_route is firing."""
    *_, billing_provider, _ = latest_session_row
    assert billing_provider == "gemini", (
        f"billing_provider was {billing_provider!r}, expected 'gemini'"
    )


def test_baseline_model_is_gemini_3_5_flash(latest_session_row):
    """Confirms the profile is actually using gemini-3.5-flash as intended."""
    _, model, *_ = latest_session_row
    assert "gemini-3.5-flash" in model, (
        f"model was {model!r}, expected to contain 'gemini-3.5-flash'"
    )
