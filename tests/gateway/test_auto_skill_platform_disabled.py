"""Regression test: gateway auto-skill loading must respect the disabled-skill
gate (channel_skill_bindings / Telegram DM Topics).

`_handle_message_with_agent`'s auto-skill block loads bound skills via
`_load_skill_payload` with a raw identifier — the same bypass class the
stacked (#58888) and bundle (#59156) invocation paths had: it skips
`get_skill_commands()`'s scan-time disabled filter, so an operator who
disables a skill for this platform (or globally) still had its full content
injected into every new session bound to that channel/topic.

This is an AST invariant pin (mirrors
test_10710_auto_reset_evicts_cached_agent.py's approach) rather than a full
functional invocation, since `_handle_message_with_agent` requires a large
mocked harness (session store, adapters, hooks) unrelated to this check.
"""
from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run


def _calls(node: ast.AST) -> set[str]:
    """Bare-name and attribute call targets invoked anywhere under ``node``."""
    names: set[str] = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute):
            names.add(n.func.attr)
        elif isinstance(n.func, ast.Name):
            names.add(n.func.id)
    return names


def _imported_disabled_check_name(node: ast.AST) -> str | None:
    """Local bound name for ``agent.skill_utils.get_disabled_skill_names``
    imported anywhere under ``node`` (accounts for ``as`` aliasing)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom) and sub.module == "agent.skill_utils":
            for alias in sub.names:
                if alias.name == "get_disabled_skill_names":
                    return alias.asname or alias.name
    return None


def test_auto_skill_block_checks_disabled_gate():
    """The auto-skill loading block in gateway/run.py must re-check
    get_disabled_skill_names() before injecting a bound skill's content,
    mirroring the stacked (#58888) and bundle (#59156) gates — otherwise a
    skill disabled for this platform (or globally) still auto-loads into
    every new session bound to it via channel_skill_bindings / DM Topics.
    """
    tree = ast.parse(inspect.getsource(gateway_run))

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = _calls(node)
        if "_load_skill_payload" not in calls or "_build_skill_message" not in calls:
            continue
        found = True

        bound_name = _imported_disabled_check_name(node)
        assert bound_name is not None, (
            "gateway/run.py auto-skill loading block must import "
            "get_disabled_skill_names from agent.skill_utils, mirroring the "
            "stacked (#58888) and bundle (#59156) gates."
        )
        assert bound_name in calls, (
            f"gateway/run.py auto-skill loading block imports "
            f"get_disabled_skill_names (as {bound_name!r}) but never calls it — "
            "a disabled skill would still be auto-loaded into new sessions "
            "bound to it via channel_skill_bindings / DM Topics."
        )
        break

    assert found, (
        "could not locate the auto-skill loading block in gateway/run.py "
        "(fingerprint: _load_skill_payload + _build_skill_message calls)."
    )
