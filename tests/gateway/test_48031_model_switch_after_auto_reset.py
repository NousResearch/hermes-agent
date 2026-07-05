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
import threading

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

    # Find the cleanup branch: an `if <flag>:` block that pops a model/reasoning
    # override AND clears the flag. We assert at least one such block sets
    # was_auto_reset False.
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
        }
        calls = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        # The cleanup block references the reasoning-override setter and pops
        # pending model notes — fingerprint of the transient-state cleanup.
        if "_set_session_reasoning_override" in calls and _assigns_false(node, "was_auto_reset"):
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


def _pops_last_resolved_model(node: ast.AST) -> bool:
    """True if `node` contains a `<something>.pop(session_key, ...)` where the
    receiver was bound from `getattr(self, "_last_resolved_model", ...)`.

    Fingerprints the sibling clear used at the /new and compression-exhausted
    reset sites:
        _lrm = getattr(self, "_last_resolved_model", None)
        if _lrm is not None:
            _lrm.pop(session_key, None)
    """
    # Collect local names bound from getattr(self, "_last_resolved_model", ...).
    lrm_names = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Assign)
            and isinstance(sub.value, ast.Call)
            and isinstance(sub.value.func, ast.Name)
            and sub.value.func.id == "getattr"
            and len(sub.value.args) >= 2
            and isinstance(sub.value.args[1], ast.Constant)
            and sub.value.args[1].value == "_last_resolved_model"
        ):
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name):
                    lrm_names.add(tgt.id)
    if not lrm_names:
        return False
    # Look for `<lrm_name>.pop(session_key, ...)`.
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "pop"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id in lrm_names
            and sub.args
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id == "session_key"
        ):
            return True
    return False


def test_run_clears_last_resolved_model_in_auto_reset_cleanup_block():
    """The `_was_auto_reset` cleanup block in gateway/run.py must clear the
    per-session `_last_resolved_model` entry, so the first post-reset turn
    resolves from current config instead of recovering the stale pre-reset
    model on a transient empty-config read (#35314).

    `11b4a21a5` added this clear to the /new and compression-exhausted reset
    sites but missed the daily/idle/suspended auto-reset boundary — this pins
    it into that third site. AST-invariant to match this file's existing style.
    """
    tree = ast.parse(inspect.getsource(gateway_run))

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        # Uniquely identify the auto-reset cleanup block: it calls the
        # reasoning-override setter AND consumes was_auto_reset (only that
        # block does both — the compression-exhausted site does not clear
        # the flag).
        if (
            "_set_session_reasoning_override" in calls
            and _assigns_false(node, "was_auto_reset")
        ):
            assert _pops_last_resolved_model(node), (
                "gateway/run.py auto-reset cleanup block must clear the "
                "per-session `_last_resolved_model` entry (pop session_key) so "
                "the post-reset turn resolves from current config, mirroring the "
                "/new and compression-exhausted reset sites (11b4a21a5)."
            )
            found = True
            break
    assert found, (
        "Could not locate the auto-reset cleanup block in gateway/run.py."
    )


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._last_resolved_model = {}
    runner._service_tier = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


def test_auto_reset_cleanup_pops_per_session_last_resolved_model():
    """Behavioral mirror of test_new_clears_last_resolved_model: exercise the
    exact clear the cleanup block performs and assert the per-session entry is
    gone while the process-wide "*" slot survives (#48031 / #35314)."""
    runner = _make_runner()
    sk = "agent:main:qqbot:dm:123"

    # Seed a stale pre-reset resolution: per-session AND process-wide.
    runner._last_resolved_model[sk] = "old-model"
    runner._last_resolved_model["*"] = "old-model"

    # Run the exact clear the `_was_auto_reset` cleanup block performs.
    runner._session_model_overrides.pop(sk, None)
    _lrm = getattr(runner, "_last_resolved_model", None)
    if _lrm is not None:
        _lrm.pop(sk, None)

    # Per-session cache cleared; the process-wide "*" safety net survives.
    assert sk not in runner._last_resolved_model
    assert runner._last_resolved_model.get("*") == "old-model"
