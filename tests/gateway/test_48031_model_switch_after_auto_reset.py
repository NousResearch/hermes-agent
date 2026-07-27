"""Regression test for #48031 — /model switch lost after session auto-reset.

When `/model X` is the FIRST message after an idle/daily/suspended auto-reset,
it stores a session model override but the `was_auto_reset` flag is left True
(the slash-command path doesn't pass through the message handler that consumes
it). On the NEXT regular message, the auto-reset cleanup block in
`_handle_message_with_agent` pops the freshly-stored override BEFORE the flag
is consumed, so the switch is silently lost and resolution falls back to the
config default — while the session DB still shows the switched model (a
two-sources-of-truth divergence).

The fix consumes `was_auto_reset` at two sites:
  1. the cleanup block in gateway/run.py captures it into a local and sets the
     attribute False immediately (so it can't re-fire next message);
  2. the slash-command model path in gateway/slash_commands.py consumes it
     before storing the override (so a /model-first-after-reset isn't wiped).

These are AST invariants — load-bearing pins that fail if either consume is
removed (mirrors test_35809_auto_reset_clean_context.py's approach).
"""
from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run
from gateway import slash_commands as gateway_slash


def _assigns_false(node: ast.AST, attr: str) -> bool:
    """True if `node` contains an assignment `<something>.<attr> = False`."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == attr
                    and isinstance(sub.value, ast.Constant)
                    and sub.value.value is False
                ):
                    return True
    return False


def test_run_consumes_was_auto_reset_in_cleanup_block():
    """The auto-reset cleanup block in gateway/run.py must set
    `session_entry.was_auto_reset = False` so the cleanup (which pops the
    session model/reasoning overrides) cannot re-fire on the next message and
    wipe an override stored between turns (#48031)."""
    tree = ast.parse(inspect.getsource(gateway_run))

    # Find the cleanup branch: an `if <flag>:` block that clears the
    # conversation scope (post-funnel: one _clear_conversation_scope call
    # replaced the inline override pops) AND clears the flag. We assert at
    # least one such block sets was_auto_reset False.
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        # The cleanup block routes through the conversation-scope funnel —
        # fingerprint of the transient-state cleanup.
        if "_clear_conversation_scope" in calls and _assigns_false(node, "was_auto_reset"):
            found = True
            break
    assert found, (
        "gateway/run.py auto-reset cleanup block must consume "
        "`was_auto_reset` (set it False) so it can't re-fire and wipe a "
        "model override stored between turns (#48031)."
    )


def test_slash_command_model_path_consumes_was_auto_reset():
    """The slash-command model path in gateway/slash_commands.py must consume
    `was_auto_reset` before storing the new model override, so a
    /model-first-after-auto-reset isn't wiped by the next message's cleanup
    (#48031)."""
    src = inspect.getsource(gateway_slash)
    tree = ast.parse(src)
    assert _assigns_false(tree, "was_auto_reset"), (
        "gateway/slash_commands.py model path must set "
        "`was_auto_reset = False` before storing the model override (#48031)."
    )
