"""A kanban worker that hits a provider quota wall must SAY so in its exit status.

The dispatcher's reap classifier holds a pid, not a transcript: exit status is
the only channel. ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75, EX_TEMPFAIL) means "the
provider walled me off" and releases the task back to ``ready`` without counting
a failure. Exit 0 instead and the dispatcher sees a clean exit while the task is
still ``running`` — a protocol violation — and counts a failure against a worker
that did nothing wrong. Three cards on a live board were blocked exactly that
way, nine runs burned, every log ending in ``API call failed after 3 retries:
HTTP 429``.

The producer existed but only on the ``-Q`` branch of the one-shot path, and
workers are spawned with ``-Q`` only in goal mode — so an ordinary card never
reached it. These tests pin the decision (one helper) and the wiring (both
one-shot paths ask it).

Port note: since the cli.py god-file split, the two one-shot exits live in
``_run_quiet_single_query`` and ``_run_single_query_mode`` rather than inline in
``main()``, and the interactive turn's verdict is parked on the CLI by
``hermes_cli/cli_chat_turn_mixin.py``. The guards follow the code.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import cli as cli_module
from hermes_cli import cli_chat_turn_mixin
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

CLI_SOURCE = Path(cli_module.__file__).read_text(encoding="utf-8")
MIXIN_SOURCE = Path(cli_chat_turn_mixin.__file__).read_text(encoding="utf-8")


def _rate_limited_result() -> dict:
    return {
        "final_response": "API call failed after 3 retries: HTTP 429",
        "failed": True,
        "completed": False,
        "error": "HTTP 429: The service may be temporarily overloaded",
        "failure_reason": "rate_limit",
    }


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_a_quota_wall_reports_the_tempfail_sentinel(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    result = {**_rate_limited_result(), "failure_reason": reason}
    assert cli_module._kanban_quota_wall_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_only_a_kanban_worker_gets_the_sentinel(monkeypatch):
    """Other one-shot runs keep the plain contract automation wrappers expect."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cli_module._kanban_quota_wall_exit_code(_rate_limited_result()) is None


def test_a_real_failure_is_not_laundered_into_a_throttle(monkeypatch):
    """The breaker must still see failures the task itself caused.

    This is the assertion that keeps the fix from becoming a way for every
    failure to dodge the failure counter.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    for reason in ("server_error", "timeout", "auth_permanent", "upstream_rate_limit", None):
        result = {**_rate_limited_result(), "failure_reason": reason}
        assert cli_module._kanban_quota_wall_exit_code(result) is None, reason


def test_a_successful_turn_is_not_a_quota_wall(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    # `failure_reason` can be carried by a turn that did NOT fail (a retry that
    # later succeeded); the `failed` flag is what makes it terminal.
    for result in (
        {"failed": False, "failure_reason": "rate_limit"},
        {"completed": True, "failure_reason": "rate_limit"},
        {},
        None,
        "not a dict",
        42,
    ):
        assert cli_module._kanban_quota_wall_exit_code(result) is None, repr(result)


def test_the_quota_wall_reasons_are_named_in_exactly_one_place():
    """Two copies of this tuple is how the two paths drifted apart in the first place."""
    definitions = re.findall(r"^_QUOTA_WALL_FAILURE_REASONS\s*=", CLI_SOURCE, re.M)
    assert len(definitions) == 1
    inline = re.findall(r'\(\s*"rate_limit"\s*,\s*"billing"\s*\)', CLI_SOURCE)
    assert len(inline) == 1, (
        "the quota-wall reasons appear inline somewhere besides "
        "_QUOTA_WALL_FAILURE_REASONS — the decision has been copied again"
    )


def _module_functions(source: str):
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def test_both_one_shot_paths_ask_the_helper():
    """The defect was an asymmetry, so the guard is about BOTH one-shot paths.

    This reads the source: the two exits live deep inside functions that build a
    whole CLI before reaching them, so there is no honest way to exercise them
    in-process. What IS worth pinning is that neither path decides this for
    itself: exactly one call inside each of the two one-shot runners, and no
    third copy anywhere else in the module.
    """
    functions = _module_functions(CLI_SOURCE)
    for name in ("_run_quiet_single_query", "_run_single_query_mode"):
        assert name in functions, f"one-shot runner {name} moved again — repin this guard"
    calls = {
        name: [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_kanban_quota_wall_exit_code"
        ]
        for name, func in functions.items()
        if name in ("_run_quiet_single_query", "_run_single_query_mode")
    }
    for name, sites in calls.items():
        assert len(sites) == 1, f"expected exactly one helper call in {name}, found {len(sites)}"
    total = sum(
        len(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_kanban_quota_wall_exit_code"
            ]
        )
        for tree in ast.parse(CLI_SOURCE).body
    )
    assert total == 2, f"the helper is called {total}x module-wide — a third path grew its own copy"


def test_the_human_path_can_reach_the_turns_verdict():
    """``chat()`` returns response TEXT, so the verdict has to be parked on the CLI.

    Without the mixin's assignment the human path reads ``None``, the helper
    answers "not a quota wall", and the worker exits 0 again — the original bug,
    with the fix in place and inert. The one-shot caller reads it back through
    ``getattr(cli, "_last_turn_result", None)``.
    """
    stores = [
        node for node in ast.walk(ast.parse(MIXIN_SOURCE))
        if isinstance(node, ast.Attribute)
        and node.attr == "_last_turn_result"
        and isinstance(node.ctx, ast.Store)
    ]
    assert stores, "the mixin no longer assigns cli._last_turn_result"
    reads = [
        node for node in ast.walk(ast.parse(CLI_SOURCE))
        if isinstance(node, ast.Constant) and node.value == "_last_turn_result"
    ]
    assert reads, "nothing reads cli._last_turn_result on the one-shot path"
