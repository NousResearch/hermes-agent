"""Regression: the concurrent-tool progress heartbeat reported the wrong
tool names when a batch mixed guardrail-blocked and runnable calls.

``execute_tool_calls_concurrent`` builds ``futures`` in lock-step with
``runnable_calls`` (the non-blocked calls), but the heartbeat looked up the
still-running tool names with ``parsed_calls[futures.index(f)]`` — indexing
``parsed_calls`` (which also includes blocked calls) by a *futures* position.
When a blocked call precedes a runnable one, the indices shift and the
heartbeat status line ("concurrent tools running (… remaining: …)") lists the
wrong tool names.
"""

from agent.tool_executor import _running_tool_names


class _F:
    """Stand-in for a concurrent.futures.Future (identity only)."""


def test_running_names_correct_with_blocked_call_offset():
    # parsed_calls = [blocked, runnableA, runnableB]; runnable_calls keeps the
    # original indices, so it is [(1, .., 'read_file', ..), (2, .., 'web_search', ..)].
    runnable_calls = [
        (1, object(), "read_file", {}),
        (2, object(), "web_search", {}),
    ]
    fA, fB = _F(), _F()
    futures = [fA, fB]  # parallel to runnable_calls

    # Only the second runnable tool (web_search) is still running.
    assert _running_tool_names({fB}, futures, runnable_calls) == ["web_search"]
    # Only the first (read_file).
    assert _running_tool_names({fA}, futures, runnable_calls) == ["read_file"]


def test_running_names_skip_futures_not_in_list():
    runnable_calls = [(0, object(), "terminal", {})]
    f0 = _F()
    futures = [f0]
    stray = _F()  # a future that isn't one of ours
    assert _running_tool_names({f0, stray}, futures, runnable_calls) == ["terminal"]


def test_running_names_empty():
    assert _running_tool_names(set(), [], []) == []
