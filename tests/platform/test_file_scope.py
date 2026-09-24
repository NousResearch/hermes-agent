"""File tools are held to the task's workspace — found by a real model on a live run.

A worker on the Gemini run, looking for refund data, searched the host and listed every
``state.db`` on it: other agents' conversation history. The runtime's own file guard is,
in its words, "defense-in-depth, NOT a security boundary" — it keeps credential files out,
not other agents' data, and not the agent's own compiled policy, which ``write_file``
could replace. These run the installed policy plugin as a worker would load it.
"""

from __future__ import annotations

import os

import pytest

from nova.apply import apply_bundle
from nova.policy.decide import ALLOW, DENY, decide_paths

from .test_policy_enforcement import load_installed_plugin

AGENT, OTHER = "operations", "customer-support"


def test_the_rule_itself():
    ws, own = "/w/t_1", "/h/profiles/ops"
    kwargs = dict(writable_roots=[ws], readable_roots=[own])
    assert decide_paths("read_file", [ws + "/a.csv"], **kwargs).effect == ALLOW
    assert decide_paths("read_file", [own + "/SOUL.md"], **kwargs).effect == ALLOW
    assert decide_paths("write_file", [own + "/nova-policy.json"], **kwargs).effect == DENY
    assert decide_paths("read_file", ["/w/t_10/a.csv"], **kwargs).effect == DENY, "a prefix is not a parent"
    assert decide_paths("read_file", ["/h/profiles/other/state.db"], **kwargs).effect == DENY
    assert decide_paths("write_file", ["/anything"], writable_roots=[], readable_roots=[own]).effect == DENY


@pytest.fixture
def worker(bundle, runtime, audit, home, tmp_path, monkeypatch):
    """The installed plugin in a worker for a task with its own workspace."""
    apply_bundle(bundle, runtime, audit=audit)
    workspace = tmp_path / "kanban" / "workspaces" / "t_abc"
    workspace.mkdir(parents=True)
    (workspace / "ledger.csv").write_text("id,amount\n")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)  # no acceptance check here
    plugin = load_installed_plugin(home, AGENT, "nova_policy_file_scope")
    # The example's allow-list grants file tools through toolsets; make sure the scope, not
    # the allow-list, is what decides below.
    plugin._load_policy()["allow"].extend(["read_file", "write_file", "patch", "search_files"])
    return plugin, workspace, home


def blocked(plugin, tool, **args):
    result = plugin.pre_tool_call(tool_name=tool, args=args)
    return bool(result and result.get("action") == "block")


def test_the_workspace_is_readable_and_writable(worker):
    plugin, ws, _ = worker
    assert not blocked(plugin, "read_file", path=str(ws / "ledger.csv"))
    assert not blocked(plugin, "read_file", path="ledger.csv"), "relative paths anchor at the workspace"
    assert not blocked(plugin, "write_file", path="findings.md", content="x")
    assert not blocked(plugin, "search_files", pattern="refund"), "the default path is the workspace"


def test_other_agents_history_and_the_shared_board_are_out_of_reach(worker):
    plugin, _, home = worker
    for path in (home / "profiles" / OTHER / "state.db", home / "kanban.db", home / "audit.jsonl", "/etc/passwd"):
        assert blocked(plugin, "read_file", path=str(path)), path
    assert blocked(plugin, "search_files", pattern="*.db", target="files", path="/")
    assert blocked(plugin, "read_file", path="../../../profiles/customer-support/state.db")


def test_an_agent_cannot_rewrite_its_own_policy(worker):
    plugin, _, home = worker
    own = home / "profiles" / AGENT
    assert not blocked(plugin, "read_file", path=str(own / "SOUL.md"))
    for target in (own / "nova-policy.json", own / "plugins" / "nova-policy" / "__init__.py"):
        assert blocked(plugin, "write_file", path=str(target), content="{}")
        assert blocked(plugin, "patch", path=str(target), old_string="a", new_string="b")


def test_a_v4a_patch_is_checked_file_by_file(worker):
    plugin, ws, home = worker
    inside = f"*** Begin Patch\n*** Update File: {ws}/findings.md\n@@\n-a\n+b\n*** End Patch"
    sneaky = inside.replace("*** End Patch", f"*** Add File: {home}/profiles/{AGENT}/nova-policy.json\n+{{}}\n*** End Patch")
    moved = f"*** Begin Patch\n*** Move File: {ws}/a.md -> {home}/kanban.db\n*** End Patch"
    assert not blocked(plugin, "patch", mode="patch", patch=inside)
    assert blocked(plugin, "patch", mode="patch", patch=sneaky)
    assert blocked(plugin, "patch", mode="patch", patch=moved)


def test_a_symlink_out_of_the_workspace_is_followed_and_refused(worker):
    plugin, ws, home = worker
    os.symlink(home / "profiles" / OTHER, ws / "support")
    assert blocked(plugin, "read_file", path=str(ws / "support" / "SOUL.md"))


def test_outside_a_task_nothing_is_writable(worker, monkeypatch):
    plugin, ws, _ = worker
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE")
    assert blocked(plugin, "write_file", path=str(ws / "x.md"), content="x")


def test_the_refusal_says_where_the_agent_may_work(worker):
    plugin, _, home = worker
    message = plugin.pre_tool_call(tool_name="read_file", args={"path": str(home / "kanban.db")})["message"]
    assert "workspace" in message and "knowledge" in message
