"""Warm native schema caches distinguish canonical Kanban ownership."""
from contextlib import ExitStack

import pytest


@pytest.mark.parametrize("reverse", [False, True])
def test_warm_cache_matches_fresh_assembly_across_ownership(reverse, monkeypatch):
    import os

    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from hermes_cli.config import save_config
    from model_tools import get_tool_definitions

    save_config({"tools": {"tool_search": {"defer": [], "kanban_worker_policy": "narrow"}}})
    cases = ["worker", "whitespace", "blank", "interactive", "delegated", "subprocess", "cron"]
    if reverse:
        cases.reverse()
    for case in cases + list(reversed(cases)):
        with monkeypatch.context() as env, ExitStack() as stack:
            env.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
            if case == "interactive":
                env.delenv("HERMES_KANBAN_TASK", raising=False)
            else:
                env.setenv("HERMES_KANBAN_TASK", {"blank": "", "whitespace": " \t "}.get(case, "t_cache"))
            if case == "subprocess":
                env.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
            before = dict(os.environ)
            if case == "delegated":
                stack.enter_context(delegated_child_context())
            if case == "cron":
                stack.enter_context(non_dispatcher_owned_context())
            def names(quiet):
                return {td["function"]["name"] for td in get_tool_definitions(["kanban"], quiet_mode=quiet)}
            cached = names(True)
            fresh = names(False)
            assert cached == fresh, case
            assert dict(os.environ) == before
        # Context exit must return to the owner's warm entry without an invalidation.
        with monkeypatch.context() as owner:
            owner.setenv("HERMES_KANBAN_TASK", "t_cache")
            owner.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
            assert names(True) == names(False)
