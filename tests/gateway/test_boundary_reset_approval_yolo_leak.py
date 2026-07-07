"""Regression tests: auto-reset / session-finalization must clear
approval/YOLO security state, the same as /new, /resume, and /branch.

``_clear_session_boundary_security_state()`` (gateway/run.py) is the shared
conversation-boundary cleanup: it pops pending approvals, pending
skill-reload notes, and pending update-prompt state, then clears
session-scoped dangerous-command approvals ("/approve session") and YOLO
bypass via ``tools.approval.clear_session()``. ``/new``, ``/resume``, and
``/branch`` already call it.

Three other reset paths in gateway/run.py reuse the exact same "full
conversation boundary" comment and already mirror it for
``_session_model_overrides`` / ``_set_session_reasoning_override`` /
``_pending_model_notes`` / ``_last_resolved_model`` (see #58403's sibling
fixes), but never called the security-state helper:

- the daily/idle/suspended auto-reset cleanup (``was_auto_reset`` handling)
- the compression-exhausted immediate auto-reset
- ``_session_expiry_watcher``'s permanent session-finalization block (which
  only manually cleared 2 of the helper's 5 sub-clears, missing
  ``_pending_skills_reload_notes`` and the ``tools.approval``/``slash_confirm``
  state)

Without this, a ``/yolo`` or ``/approve session`` grant made before any of
these resets would silently survive into the "fresh" conversation under the
same ``session_key`` — bypassing dangerous-command approval without the user
ever re-granting it.

These are AST invariants (mirrors test_10710_auto_reset_evicts_cached_agent.py
and test_48031_model_switch_after_auto_reset.py's approach) — load-bearing
pins that fail if the security-state clear is removed from any of the three
cleanup blocks.
"""
from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run


def _calls(node: ast.AST) -> set[str]:
    """Method-call attribute names invoked anywhere under ``node``."""
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }


def _assigns_false(node: ast.AST, attr: str) -> bool:
    """True if ``node`` contains an assignment ``<something>.<attr> = False``."""
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


def _references_name(node: ast.AST, literal: str) -> bool:
    """True if a string constant equal to ``literal`` appears anywhere under ``node``."""
    return any(
        isinstance(n, ast.Constant) and n.value == literal for n in ast.walk(node)
    )


def _find_if_blocks(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, ast.If)]


def test_daily_idle_auto_reset_clears_boundary_security_state():
    """The ``was_auto_reset`` cleanup block must call
    ``_clear_session_boundary_security_state`` so a fresh auto-reset session
    does not inherit the previous conversation's dangerous-command approvals
    or YOLO bypass."""
    tree = ast.parse(inspect.getsource(gateway_run))

    found = False
    for node in _find_if_blocks(tree):
        calls = _calls(node)
        if (
            "_set_session_reasoning_override" in calls
            and _assigns_false(node, "was_auto_reset")
        ):
            assert "_clear_session_boundary_security_state" in calls, (
                "gateway/run.py's was_auto_reset cleanup block must call "
                "`self._clear_session_boundary_security_state(session_key)` "
                "— the same conversation-boundary security clear /new, "
                "/resume, and /branch already apply — so a /yolo or "
                "'/approve session' grant from before the auto-reset does "
                "not silently bypass approval in the fresh conversation."
            )
            found = True
            break
    assert found, (
        "could not locate the auto-reset transient-state cleanup block in "
        "gateway/run.py (fingerprint: _set_session_reasoning_override + "
        "was_auto_reset = False)."
    )


def test_compression_exhausted_auto_reset_clears_boundary_security_state():
    """The compression-exhausted immediate auto-reset must also call
    ``_clear_session_boundary_security_state`` for the same reason."""
    tree = ast.parse(inspect.getsource(gateway_run))

    found = False
    for node in _find_if_blocks(tree):
        if not _references_name(node, "compression_exhausted"):
            continue
        calls = _calls(node)
        if "_set_session_reasoning_override" not in calls:
            continue
        assert "_clear_session_boundary_security_state" in calls, (
            "gateway/run.py's compression-exhausted auto-reset block must "
            "call `self._clear_session_boundary_security_state(session_key)` "
            "— the same conversation-boundary security clear /new, /resume, "
            "and /branch already apply — so a /yolo or '/approve session' "
            "grant from the oversized conversation does not silently bypass "
            "approval in the post-reset conversation."
        )
        found = True
        break
    assert found, (
        "could not locate the compression-exhausted auto-reset block in "
        "gateway/run.py (fingerprint: an `if` testing "
        "agent_result.get('compression_exhausted') whose body calls "
        "_set_session_reasoning_override)."
    )


def test_session_finalization_clears_boundary_security_state():
    """The permanent session-finalization block in
    ``_session_expiry_watcher`` must call
    ``_clear_session_boundary_security_state`` instead of manually popping
    only ``_pending_approvals``/``_update_prompt_pending`` — the manual
    subset missed ``_pending_skills_reload_notes`` and the
    ``tools.approval``/``slash_confirm`` session-scoped approval and YOLO
    state."""
    tree = ast.parse(inspect.getsource(gateway_run))

    # The finalization loop is a `for key, entry in ...:` block (not an
    # `if`), so scan every For node for the fingerprint call.
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        calls = _calls(node)
        if "set_expiry_finalized" not in calls:
            continue
        assert "_clear_session_boundary_security_state" in calls, (
            "gateway/run.py's _session_expiry_watcher finalization block "
            "must call `self._clear_session_boundary_security_state(key)` "
            "— finalization is a conversation boundary just like /new, "
            "/resume, and /branch — so a /yolo or '/approve session' grant "
            "made before the session went idle does not silently survive "
            "into a later resurrection of the same session_key."
        )
        found = True
        break
    assert found, (
        "could not locate the session-finalization loop in "
        "gateway/run.py (fingerprint: a `for` loop calling "
        "session_store.set_expiry_finalized)."
    )
