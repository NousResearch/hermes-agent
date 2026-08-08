"""Cross-surface consistency for temporary chats.

Temporary mode ships on four surfaces (desktop, CLI, chat platforms, one-shot
runs). Each was built at a different time, and the padlock/incognito drift was
found three separate times -- always in a place the previous fix did not look.
These tests replace "check the three sites I remember" with a repo-wide sweep.
"""

import io
import os

import pytest


REPO_ROOT = os.getcwd()
SKIP_DIRS = {".git", "node_modules", "__pycache__", "dist", ".venv", "venv",
             "optional-skills", "build", ".pytest_cache"}
CODE_EXT = (".py", ".ts", ".tsx", ".yaml", ".yml")

PADLOCK = "\U0001f512"
INCOGNITO = "\U0001f575"


def _iter_source_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(CODE_EXT):
                yield os.path.join(root, name)


def _temporary_chat_lines():
    """Lines that render user-facing temporary-chat copy."""
    hits = []
    for path in _iter_source_files():
        try:
            text = io.open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.split("\n"), 1):
            low = line.lower()
            if "temporary chat" not in low and "temporary session" not in low:
                continue
            hits.append((path, lineno, line))
    return hits


def test_no_padlock_anywhere_in_temporary_chat_copy():
    """A padlock claims "encrypted"; temporary mode promises "not recorded".

    Shipped three times in three different files despite two prior fixes, so
    this sweeps the whole repo rather than a remembered list of call sites.
    """
    offenders = [
        (os.path.relpath(p, REPO_ROOT), n, line.strip()[:80])
        for p, n, line in _temporary_chat_lines()
        if PADLOCK in line
    ]
    assert not offenders, (
        "padlock used in temporary-chat copy (use the incognito glyph):\n"
        + "\n".join(f"  {p}:{n}  {t}" for p, n, t in offenders)
    )


def test_the_incognito_glyph_is_actually_used_somewhere():
    """Guards the inverse: a sweep that matches nothing would pass silently."""
    marked = [
        p for p, _, line in _temporary_chat_lines() if INCOGNITO in line
    ]
    assert marked, "no temporary-chat copy carries the incognito glyph"


@pytest.mark.parametrize(
    "tool,action",
    [
        ("memory", "add"),
        ("memory", "replace"),
        ("memory", "remove"),
        ("skill_manage", "create"),
        ("skill_manage", "delete"),
        ("cronjob", "create"),
        ("cronjob", "remove"),
    ],
)
def test_durable_writes_are_blocked_in_a_temporary_chat(tool, action):
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    blocked = check_ephemeral_tool_block(tool, {"action": action})
    assert blocked, f"{tool}.{action} is NOT blocked in a temporary chat"
    # The refusal has to say why and offer the way out, or the agent will
    # simply retry the same call.
    assert "temporary chat" in blocked.lower()
    assert "/new" in blocked


@pytest.mark.parametrize(
    "tool,action",
    [("memory", "search"), ("skill_manage", "view"), ("cronjob", "list")],
)
def test_read_only_actions_still_work_in_a_temporary_chat(tool, action):
    """Blocking reads would make temporary mode useless, not private."""
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    assert check_ephemeral_tool_block(tool, {"action": action}) is None


def test_batched_memory_writes_cannot_smuggle_a_write_past_the_guard():
    """`memory` accepts an operations list instead of a bare action."""
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    smuggled = {
        "operations": [
            {"action": "search"},
            {"action": "add", "content": "x"},
        ]
    }
    assert check_ephemeral_tool_block("memory", smuggled), (
        "a batched memory write slipped past the ephemeral guard"
    )


def test_auto_title_is_skipped_for_temporary_sessions():
    """Auto-title writes to the sessions table -- it must honour the flag."""
    import inspect
    from agent.title_generator import maybe_auto_title

    sig = inspect.signature(maybe_auto_title)
    assert "ephemeral" in sig.parameters, (
        "maybe_auto_title lost its `ephemeral` parameter; temporary chats "
        "would get a title row written to the sessions table"
    )
    src = inspect.getsource(maybe_auto_title)
    assert "if ephemeral" in src, "the ephemeral parameter is accepted but unused"
