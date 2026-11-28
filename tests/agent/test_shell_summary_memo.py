"""A transcript projection re-parses every terminal command, so a second read pays it all again.

``_history_to_messages`` previews each terminal row through ``agent.display.summarize_shell_command``, whose
helpers walk the command one character at a time. A long session is projected again on every
``session.activate``, ``session.history`` and ``session.transcript`` page, and the same commands were
re-scanned from scratch each time: the report in #154513 measured ~9M scan steps and ~4 s for one 2,420
command session, with every other session in the gateway stalled behind that GIL-holding work.

The helpers are pure functions of their string argument, so they memoize; the summary itself stays
uncached because it localizes its text. These tests pin the summary's output (a memo must not change what a
command renders as) and the scan count of a warm projection (on the base branch a second projection costs
exactly what the first did).
"""
import json
from typing import Any

import pytest


def _summary_module() -> Any:
    """The module owning the summary's character scan: the helpers' module after the fix, the facade before it."""
    try:
        from agent import shell_summary
    except ImportError:  # the base branch keeps every helper in agent.display
        from agent import display as shell_summary
    return shell_summary


def _count_scans(monkeypatch):
    """Append one entry per character the summary's scanner yields, wherever the helpers live."""
    module = _summary_module()
    real = module._scan_quoted
    steps: list[None] = []

    def counting(text):
        for item in real(text):
            steps.append(None)
            yield item

    monkeypatch.setattr(module, "_scan_quoted", counting)
    return steps


def _terminal_history(pairs: int = 200):
    """Assistant/tool row pairs as the store returns them: the command rides the call's arguments."""
    commands = [
        "cd /home/user/project && git status --short && git log --oneline -5",
        'rg -n "def _scan_quoted" agent/display.py | head -20',
        "pytest tests/agent/test_display.py -x -q 2>&1 | tail -30",
        'for f in $(git ls-files "*.py"); do python3 -m py_compile "$f" || echo "bad: $f"; done',
        "cat ~/.hermes/config.yaml | grep -A3 model | sort | uniq -c",
    ]
    history = []
    for index in range(pairs):
        call_id = f"call-{index}"
        command = f"{commands[index % len(commands)]}  # step {index}"
        history.append({
            "role": "assistant", "content": "", "timestamp": index,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "terminal", "arguments": json.dumps({"command": command})}}],
        })
        history.append({"role": "tool", "tool_call_id": call_id, "tool_name": "terminal",
                        "content": "ok\n", "timestamp": index})
    return history


# Captured on the base branch: a memo is only safe if every one of these renders exactly as before.
SUMMARY_CASES = [
    ("", ""),
    ("ls", "ls"),
    ("cd /tmp && ls -la", "ls -la"),
    ('rg -n "def _scan_quoted" agent/display.py | head -20', 'rg -n "def _scan_quoted" agent/display.py'),
    ("cat ~/.hermes/config.yaml > /tmp/cfg.txt 2>&1", "cat ~/.hermes/config.yaml"),
    ("cd /srv && export FOO=bar && python3 -m pytest -x -q; echo done", "python3 -m pytest -x -q + 1 command"),
    ("FOO=1 BAR=2 ssh host 'ls -l | sort'", "FOO=1 BAR=2 ssh host 'ls -l | sort'"),
    ('echo "--- exit=$? ---" && echo PIPESTATUS', 'echo "--- exit=$? ---" && echo PIPESTATUS'),
    ("diff <(sort a) <(sort b) | wc -l", "diff <(sort a) <(sort b)"),
    ("printf '%s' \"a;b|c&&d\"", "printf '%s' \"a;b|c&&d\""),
    ("a && b || c ; d \n e", "a + 3 commands"),
    ("git commit -m 'fix: quotes \"inside\" a message' && git push origin HEAD",
     "git commit -m 'fix: quotes \"inside\" a message' + 1 command"),
    ("sudo -n true 2>&1 | tail -1", "sudo -n true"),
    ("silent; true; false; cd ~", "silent"),
]


@pytest.mark.parametrize(("command", "expected"), SUMMARY_CASES)
def test_summary_output_is_unchanged(command, expected):
    from agent.display import summarize_shell_command

    assert summarize_shell_command(command) == expected


def test_a_warm_projection_does_not_re_scan_the_same_commands(monkeypatch):
    from tui_gateway.server import _history_to_messages

    history = _terminal_history()
    steps = _count_scans(monkeypatch)
    first = _history_to_messages(history)
    cold_scans = len(steps)
    steps.clear()
    second = _history_to_messages(history)
    warm_scans = len(steps)

    assert first == second, "the memoized projection must render the same transcript"
    assert cold_scans > 10_000, f"the fixture should walk thousands of characters, not {cold_scans}"
    assert warm_scans == 0, f"a warm projection re-scanned {warm_scans} characters"


def test_a_single_command_is_scanned_once_per_projection():
    """Two previews of one command share the memo, so the scan does not double."""
    from agent.display import summarize_shell_command

    module = _summary_module()
    real = module._scan_quoted
    calls = []

    def counting(text):
        calls.append(text)
        yield from real(text)

    module._scan_quoted = counting
    try:
        command = "cd /srv && make -j8 && ./bin/run --flag 2>&1 | tail -5"
        first = summarize_shell_command(command)
        cold = len(calls)
        calls.clear()
        assert summarize_shell_command(command) == first
        warm = len(calls)
    finally:
        module._scan_quoted = real

    assert cold > 0
    assert warm == 0


def test_reported_session_size_does_not_thrash_summary_cache(monkeypatch):
    """A second read of the reported 2,420-command session must not redo its character scans."""
    from agent.display import summarize_shell_command

    # Different command and segment strings force the LRU to retain real session-sized history.
    commands = [
        f"cd /tmp/session-{index} && rg -n TODO src/task_{index}.py | head -10"
        for index in range(2_420)
    ]
    steps = _count_scans(monkeypatch)
    first = [summarize_shell_command(command) for command in commands]
    cold_scans = len(steps)
    steps.clear()
    second = [summarize_shell_command(command) for command in commands]

    assert first == second, "warm reads must preserve the same visible command labels"
    assert cold_scans > 10_000, "the session should exercise the real character scanner"
    assert not steps, f"the second read re-scanned {len(steps)} characters after LRU eviction"


def test_the_memo_is_size_bounded():
    """A session's command set must not grow the cache without limit."""
    module = _summary_module()
    cached = [value for value in vars(module).values()
              if callable(value) and hasattr(value, "cache_info")
              and getattr(value, "__module__", None) == module.__name__]
    assert cached, "no memoized shell-summary helper: every command is re-parsed on every projection"
    assert all(helper.cache_info().maxsize for helper in cached), \
        "an unbounded cache grows with a session's command set"
