"""The editor (ACP gateway path) offers the session's skills and expands a typed ``/skill`` like the CLI."""
import asyncio

import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_session_meta import COMMANDS, MAX_SKILL_COMMANDS, available_commands_update, skill_commands

CATALOG = {
    "pairs": [["/help", "Show commands"], ["/review-pr", "Review a PR against the project checklist"]],
    "skills": {"/review-pr": {"usage": 0, "origin": "project"}, "/status": {"usage": 0, "origin": "user"}},
}
EXPANDED = '[IMPORTANT: The user has invoked the "review-pr" skill, indicating they want you to follow it.]'


class FakeOwner:
    def __init__(self, dispatch=None, catalog=CATALOG):
        self.calls = []
        self.dispatch = dispatch if dispatch is not None else {"type": "skill", "message": EXPANDED}
        self.catalog = catalog

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == "commands.catalog":
            if isinstance(self.catalog, Exception):
                raise self.catalog
            return self.catalog
        if method == "command.dispatch":
            if isinstance(self.dispatch, Exception):
                raise self.dispatch
            return self.dispatch
        raise AssertionError(method)


def agent_with(owner):
    agent = GatewayACPAgent()

    async def client():
        return owner
    agent._client = client
    return agent


def test_skill_commands_reads_catalog_skills_with_their_descriptions():
    assert skill_commands(CATALOG) == {"review-pr": "Review a PR against the project checklist", "status": "Skill"}
    assert skill_commands({}) == {}


def test_available_commands_lists_skills_after_builtins_never_shadowing_one():
    names = [c.name for c in available_commands_update(skill_commands(CATALOG)).available_commands]
    assert names[:len(COMMANDS)] == list(COMMANDS)
    assert names[len(COMMANDS):] == ["review-pr"]  # /status is a built-in
    assert [c.name for c in available_commands_update().available_commands] == list(COMMANDS)


def test_available_commands_caps_skills():
    many = {f"skill-{i}": "x" for i in range(MAX_SKILL_COMMANDS + 10)}
    assert len(available_commands_update(many).available_commands) == len(COMMANDS) + MAX_SKILL_COMMANDS


def test_typed_skill_is_expanded_by_the_owner_for_this_session():
    owner = FakeOwner()
    agent = agent_with(owner)
    assert asyncio.run(agent._run_slash("editor-1", "s1", "/review-pr https://github.com/o/r/pull/1")) == (False, EXPANDED)
    assert owner.calls == [
        ("commands.catalog", {"session_id": "s1"}),
        ("command.dispatch", {"session_id": "s1", "name": "review-pr", "arg": "https://github.com/o/r/pull/1"}),
    ]
    # The catalog is read once per session; an underscore spelling resolves too.
    assert asyncio.run(agent._run_slash("editor-1", "s1", "/review_pr")) == (False, EXPANDED)
    assert [m for m, _ in owner.calls].count("commands.catalog") == 1


def test_words_that_are_not_skills_go_to_the_model_as_typed():
    owner = FakeOwner()
    agent = agent_with(owner)
    for text in ("/not-a-skill do it", "/etc/hosts is wrong", "/"):
        assert asyncio.run(agent._run_slash("editor-1", "s1", text)) == (False, text)
    assert all(method != "command.dispatch" for method, _ in owner.calls)


@pytest.mark.parametrize("owner", [
    FakeOwner(dispatch=RuntimeError("refused")),
    FakeOwner(dispatch={"type": "exec", "output": "ran"}),
    FakeOwner(dispatch={"type": "skill", "message": "  "}),
    FakeOwner(catalog=RuntimeError("no catalog")),
])
def test_a_failed_expansion_keeps_the_typed_text(owner):
    agent = agent_with(owner)
    assert asyncio.run(agent._run_slash("editor-1", "s1", "/review-pr x")) == (False, "/review-pr x")
