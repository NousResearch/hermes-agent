"""Diff-aware tool-progress truncation + raw-identity dedup (PR #24304, teknium1 review).

These integration tests drive the PRODUCTION progress path in
``gateway/run_turn_runner.py`` end-to-end (agent thread -> progress queue ->
editable bubble) via the same fixtures as ``test_run_progress_topics.py``.

They pin the two review points:

1. Truncation must not make DISTINCT commands render identically and then
   collapse to ``(×N)``: dedup keys on RAW tool identity, and the diff-aware
   renderer reveals the differing tail of a shared-prefix follow-up.
2. A TRUE repeat (identical raw command) still collapses to ``(×N)``.
"""

import time

import pytest

from gateway.config import Platform
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _run_with_agent


class CodeBlockProgressCaptureAdapter(ProgressCaptureAdapter):
    """A markdown platform that renders fenced code blocks (Slack/Discord/Matrix shape),
    so terminal progress takes the ``_progress_terminal_blocks`` code-block path."""

    supports_code_blocks = True


# Long SHARED prefix so both commands exceed the 40-char preview cap and must be
# truncated; only the trailing verb (``alpha`` / ``beta``) differs.
_SHARED_PREFIX = "cd /very/long/shared/prefix/directory/here && "


class DistinctTerminalPrefixAgent:
    """Two DISTINCT terminal commands sharing a long prefix (code-block path)."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", None, {"command": _SHARED_PREFIX + "alpha"})
        time.sleep(0.35)
        cb("tool.started", "terminal", None, {"command": _SHARED_PREFIX + "beta"})
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class RepeatTerminalAgent:
    """The SAME terminal command twice — a genuine repeat that MUST collapse to (×N)."""

    CMD = _SHARED_PREFIX + "alpha"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", self.CMD, {"command": self.CMD})
        time.sleep(0.35)
        cb("tool.started", "terminal", self.CMD, {"command": self.CMD})
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


# Long shared-prefix web_search queries: distinct raw previews, non-terminal path.
_QUERY_PREFIX = "the quick brown fox jumps over the very lazy dog number "


class DistinctPreviewAgent:
    """Two DISTINCT non-terminal previews sharing a long prefix (preview path)."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "web_search", _QUERY_PREFIX + "one", {"query": _QUERY_PREFIX + "one"})
        time.sleep(0.35)
        cb("tool.started", "web_search", _QUERY_PREFIX + "two", {"query": _QUERY_PREFIX + "two"})
        time.sleep(0.35)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _all_progress_text(adapter) -> str:
    """Every byte of progress the adapter ever rendered (sends + edits)."""
    parts = [call["content"] for call in adapter.sent]
    parts += [edit["content"] for edit in adapter.edits]
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_distinct_terminal_commands_do_not_collapse_and_show_tails(monkeypatch, tmp_path):
    """Code-block path: two commands with a shared long prefix must NOT collapse to (×N),
    and the diff-aware renderer must reveal each differing tail (alpha / beta)."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DistinctTerminalPrefixAgent,
        session_id="sess-diff-distinct-terminal",
        config_data={"display": {"tool_progress": "all", "tool_preview_length": 40}},
        platform=Platform.SLACK,
        chat_id="C1",
        chat_type="group",
        thread_id="thread-1",
        adapter_cls=CodeBlockProgressCaptureAdapter,
    )

    assert result["final_response"] == "done"
    text = _all_progress_text(adapter)
    assert "alpha" in text, f"first command tail missing: {text!r}"
    assert "beta" in text, f"diff-aware tail of second command missing: {text!r}"
    # Distinct raw commands must never be deduped into a repeat count.
    assert "(×" not in text, f"distinct commands wrongly collapsed: {text!r}"


@pytest.mark.asyncio
async def test_true_repeat_terminal_command_collapses(monkeypatch, tmp_path):
    """The same raw command twice still collapses to a (×N) repeat count."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RepeatTerminalAgent,
        session_id="sess-diff-repeat-terminal",
        config_data={"display": {"tool_progress": "all", "tool_preview_length": 40}},
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        thread_id=None,
    )

    assert result["final_response"] == "done"
    text = _all_progress_text(adapter)
    assert "(×2)" in text, f"true repeat did not collapse to a repeat count: {text!r}"


@pytest.mark.asyncio
async def test_true_repeat_terminal_command_collapses_code_block(monkeypatch, tmp_path):
    """Same as above but on teknium1's code-block path (supports_code_blocks=True):
    a genuine repeat of the same raw command must still collapse to (×N) there too —
    the raw-key dedup keeps working when the rendered message is a fenced block."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RepeatTerminalAgent,
        session_id="sess-diff-repeat-terminal-cb",
        config_data={"display": {"tool_progress": "all", "tool_preview_length": 40}},
        platform=Platform.SLACK,
        chat_id="C1",
        chat_type="group",
        thread_id="thread-1",
        adapter_cls=CodeBlockProgressCaptureAdapter,
    )

    assert result["final_response"] == "done"
    text = _all_progress_text(adapter)
    assert "(×2)" in text, f"true repeat on code-block path did not collapse: {text!r}"


@pytest.mark.asyncio
async def test_distinct_nonterminal_previews_do_not_collapse(monkeypatch, tmp_path):
    """Non-terminal preview path: distinct shared-prefix previews stay distinct (no (×N))
    and the second reveals its differing tail via the threaded prev."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DistinctPreviewAgent,
        session_id="sess-diff-distinct-preview",
        config_data={"display": {"tool_progress": "all", "tool_preview_length": 40}},
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        thread_id=None,
    )

    assert result["final_response"] == "done"
    text = _all_progress_text(adapter)
    assert "two" in text, f"diff-aware tail of second preview missing: {text!r}"
    assert "(×" not in text, f"distinct previews wrongly collapsed: {text!r}"
