"""
Tests for signal-aware tool output pruning in context compression.

These tests validate that the signal scorer correctly classifies tool
outputs and that the pruning pass preserves high-signal content while
aggressively pruning low-signal content.

Run with:
    python -m pytest tests/agent/test_signal_aware_pruning.py -v
"""

import pytest
from agent.signal_scorer import (
    score_tool_output,
    signal_aware_prune_action,
    smart_truncate_for_summarizer,
)


# ---------------------------------------------------------------------------
# Test 1: High-signal errors score above keep threshold
# ---------------------------------------------------------------------------

HIGH_SIGNAL_OUTPUTS = [
    (
        "terminal",
        "ERROR: Connection refused\nTraceback (most recent call last):\n"
        "  File 'app.py', line 42, in connect\n"
        "    raise ConnectionError('database unreachable')\n"
        "ConnectionError: database unreachable",
        "stack trace with connection error",
    ),
    (
        "terminal",
        "FAILED: test_authentication\n"
        "AssertionError: Expected 200, got 401\n"
        "  File 'tests/test_auth.py', line 23\n"
        "    assert response.status_code == 200\n"
        "AssertionError",
        "test failure with assertion",
    ),
    (
        "terminal",
        "CRITICAL: segfault in kernel module at address 0xDEADBEEF\n"
        "Segmentation fault (core dumped)",
        "segfault",
    ),
    (
        "terminal",
        "Security vulnerability CVE-2025-1234 detected in dependency\n"
        "Recommend immediate upgrade to patched version",
        "security vulnerability",
    ),
    (
        "delegate_task",
        "ERROR: Subagent failed with exception\n"
        "Task: deploy to production\n"
        "Result: psycopg2.OperationalError: could not connect to server",
        "delegate_task failure",
    ),
]


@pytest.mark.parametrize("tool_name,content,description", HIGH_SIGNAL_OUTPUTS)
def test_high_signal_kept(tool_name, content, description):
    """High-signal outputs must score ≥ 3 (keep threshold)."""
    score = score_tool_output(tool_name, content)
    action = signal_aware_prune_action(tool_name, content)
    assert score >= 3, (
        f"{description}: expected score ≥ 3, got {score}. "
        f"Error information must survive compression."
    )
    assert action == "keep", (
        f"{description}: expected action='keep', got '{action}'"
    )


# ---------------------------------------------------------------------------
# Test 2: Low-signal outputs score below prune threshold
# ---------------------------------------------------------------------------

LOW_SIGNAL_OUTPUTS = [
    (
        "write_file",
        "File written successfully (245 bytes)",
        "file write success",
    ),
    (
        "terminal",
        "Command completed successfully.\nOK\nDone",
        "command success",
    ),
    (
        "write_file",
        "File saved. Directory created. All done.",
        "multiple success messages",
    ),
    (
        "terminal",
        "3 files processed. Finished. OK",
        "routine terminal output",
    ),
]


@pytest.mark.parametrize("tool_name,content,description", LOW_SIGNAL_OUTPUTS)
def test_low_signal_pruned(tool_name, content, description):
    """Low-signal outputs must score ≤ -2 (prune threshold)."""
    score = score_tool_output(tool_name, content)
    action = signal_aware_prune_action(tool_name, content)
    assert score <= -2, (
        f"{description}: expected score ≤ -2, got {score}. "
        f"Success confirmations should be pruned."
    )
    assert action == "prune", (
        f"{description}: expected action='prune', got '{action}'"
    )


# ---------------------------------------------------------------------------
# Test 3: Secrets always score negative (suppressed)
# ---------------------------------------------------------------------------

SECRET_OUTPUTS = [
    ("terminal", "API_KEY=sk-live-prod-abc123xyz\nConnection successful"),
    ("terminal", "DATABASE_URL=postgresql://admin:hunter2@db.internal:5432/prod"),
    ("terminal", "password=supersecret123\nLogin successful"),
    ("read_file", "export GITHUB_TOKEN=ghp_1A2b3C4d5E6f7G8h9I0j"),
]


@pytest.mark.parametrize("tool_name,content", SECRET_OUTPUTS)
def test_secrets_suppressed(tool_name, content):
    """Secrets must score negative regardless of surrounding content."""
    score = score_tool_output(tool_name, content)
    # The secret pattern contributes -5.  Other patterns (URLs, "successful")
    # may add small positive scores, but the net should still be ≤ 0.
    assert score <= 0, (
        f"Secret in output scored {score}. "
        f"Secrets must never score positive — they should be suppressed."
    )


# ---------------------------------------------------------------------------
# Test 4: Short error messages (>50, ≤200 chars) are still scored
# ---------------------------------------------------------------------------

def test_short_error_scored():
    """Short but critical error messages must be scored correctly."""
    # 130 chars — below the old 200-char minimum but above the new 50-char minimum
    short_error = (
        "ERROR: NullPointerException at line 42\n"
        "Traceback (most recent call last):\n"
        "  File main.py, line 42\n"
        "    result = process()\n"
        "TypeError: NoneType is not callable"
    )
    assert len(short_error) < 200
    score = score_tool_output("terminal", short_error)
    action = signal_aware_prune_action("terminal", short_error)
    assert score >= 3, f"Short error scored {score}, expected ≥ 3 (keep)"
    assert action == "keep", f"Short error action is '{action}', expected 'keep'"


# ---------------------------------------------------------------------------
# Test 5: Hidden error in noisy output detected
# ---------------------------------------------------------------------------

def test_hidden_error_detected():
    """A single critical error in 1000 lines of "OK" must be detected."""
    content = "OK\n" * 500 + "CRITICAL: segfault in kernel module\n" + "OK\n" * 500
    score = score_tool_output("terminal", content)
    # The "OK" × 500 contributes -3 × 2 (capped) = -6
    # The "CRITICAL" + "segfault" contributes 10 + 15 = 25
    # Net ≈ 19.  Must be above keep threshold.
    assert score >= 3, (
        f"Hidden error scored {score}. "
        f"Single critical error in noisy output must be detected."
    )


# ---------------------------------------------------------------------------
# Test 6: Unicode error messages handled
# ---------------------------------------------------------------------------

def test_unicode_errors():
    """Error messages with CJK/emoji must be detected."""
    content = "🚀 ERROR: 测试失败 💥\nファイルが見つかりません\nエラー: 接続拒否\n🔥 CRITICAL: Système en panne"
    score = score_tool_output("terminal", content)
    assert score >= 3, f"Unicode error scored {score}, expected ≥ 3"


# ---------------------------------------------------------------------------
# Test 7: Neutral outputs get "summarize" action
# ---------------------------------------------------------------------------

NEUTRAL_OUTPUTS = [
    ("read_file", "import os\nimport sys\n\ndef main():\n    pass\n" * 10, "file content"),
    ("search_files", '{"matches": [{"file": "a.py", "line": 1}], "total_count": 1}', "search result"),
    ("patch", "Patch applied successfully.\n1 file changed, 3 insertions(+), 1 deletion(-)", "patch diff"),
]


@pytest.mark.parametrize("tool_name,content,description", NEUTRAL_OUTPUTS)
def test_neutral_summarized(tool_name, content, description):
    """Neutral outputs must get action='summarize'."""
    action = signal_aware_prune_action(tool_name, content)
    assert action == "summarize", (
        f"{description}: expected 'summarize', got '{action}'"
    )


# ---------------------------------------------------------------------------
# Test 8: Empty/very short content
# ---------------------------------------------------------------------------

def test_empty_content():
    """Empty and very short content must not crash."""
    assert score_tool_output("terminal", "") < 0
    assert score_tool_output("terminal", "OK") < 0
    assert score_tool_output("terminal", "None") < 0

    # Very short errors (≤50 chars) should still be scored accurately
    tiny_error = "ERROR: fail"
    assert len(tiny_error) <= 50
    # It scores: error(+10) + fail(+8) + short(-2) = 16 → keep
    assert score_tool_output("terminal", tiny_error) > 0


# ---------------------------------------------------------------------------
# Test 9: Smart truncation preserves errors
# ---------------------------------------------------------------------------

def test_smart_truncation_preserves_error_at_end():
    """Error at end of long content must survive truncation."""
    # Stock head/tail: head=4000 captures only 'a's, tail=1500 only 'b's
    # The error in between is lost.  Smart truncation must find it.
    content = "a" * 5000 + "\nERROR: Critical failure\n" + "b" * 1000
    result = smart_truncate_for_summarizer(content, max_chars=6000)
    assert "ERROR: Critical failure" in result


def test_smart_truncation_preserves_error_at_start():
    """Error at start of long content must survive truncation."""
    content = "ERROR: Startup failure\n" + "c" * 15000
    result = smart_truncate_for_summarizer(content, max_chars=6000)
    assert "ERROR: Startup failure" in result


def test_smart_truncation_multiple_errors():
    """Multiple error blocks must all be captured."""
    content = (
        "a" * 800 + "\nERROR: error1\n" +
        "b" * 1500 + "\nFAILED: error2\n" +
        "c" * 1500 + "\nTraceback...\n" +
        "d" * 1500
    )
    result = smart_truncate_for_summarizer(content, max_chars=6000)
    assert "error1" in result and "error2" in result


def test_smart_truncation_short_content_unchanged():
    """Content within budget must return unchanged."""
    content = "Hello world"
    result = smart_truncate_for_summarizer(content, max_chars=100)
    assert result == content


# ---------------------------------------------------------------------------
# Test 10: Rapid oscillation (errors + successes alternating)
# ---------------------------------------------------------------------------

def test_rapid_oscillation():
    """Rapid error/success alternation must score as high-signal."""
    content = ""
    for i in range(20):
        if i % 2 == 0:
            content += f"ERROR: failure in iteration {i}\n"
        else:
            content += f"Success: iteration {i} completed\n"
    score = score_tool_output("terminal", content)
    # 10 errors × 10 points = 100.  10 successes × -3 (capped ×2) = -6.
    # Net ≈ 94.  Must be high-signal.
    assert score >= 10, f"Oscillation scored {score}, expected ≥ 10"
    assert signal_aware_prune_action("terminal", content) == "keep"
