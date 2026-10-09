"""Behavior and ownership of the shared execution contract."""
from __future__ import annotations
import ast
import builtins
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import pytest
from commands import COMMAND_REGISTRY, resolve_command
from commands.execution import CommandContext, CommandReply, EXECUTORS, execute_command, resolve_executor, run_execute


def test_every_execution_key_has_one_implementation():
    keys = {cmd.execute for cmd in COMMAND_REGISTRY if cmd.execute}
    assert keys == set(EXECUTORS)
    assert len(set(EXECUTORS.values())) == len(keys)
    assert all(resolve_executor(cmd) for cmd in COMMAND_REGISTRY if cmd.execute)


def test_contract_fields_and_frozen_values_are_preserved():
    assert [f.name for f in fields(CommandContext)] == ["surface", "args", "options", "config_get"]
    assert [f.name for f in fields(CommandReply)] == ["text", "data", "format"]
    assert CommandContext() == CommandContext(surface="cli", args="", options={}, config_get=None)
    assert CommandReply("ok") == CommandReply("ok", {}, "plain")
    with pytest.raises(FrozenInstanceError):
        CommandReply("ok").text = "changed"


@pytest.fixture
def subsystems(monkeypatch):
    import agent.skill_bundles as bundles
    import agent.skill_commands as skills
    import tools.skills_tool as tool
    monkeypatch.setattr(bundles, "list_bundles", lambda: [
        {"slug": "review", "skills": ["audit", "check"], "description": ""}])
    monkeypatch.setattr(bundles, "_bundles_dir", lambda: Path("/fixture/bundles"))
    monkeypatch.setattr(skills, "get_skill_commands", lambda: {
        "/tidy": {"description": "Tidy files"}, "/audit": {"description": ""}})
    monkeypatch.setattr(tool, "_find_all_skills", lambda: [{"name": "handoff"}, {"name": "tidy"}])


def translate(key, **values):
    return key + (":" + ",".join(f"{k}={v}" for k, v in values.items()) if values else "")


def inputs():
    return {
        "version_label": lambda: "Hermes Agent vfixture · canary",
        "egress_status": lambda: "Egress proxy status\nEnabled: no",
        "profile_name": "research", "profile_label": "Research (research)",
        "home_display": "~/.hermes/profiles/research",
        "translate": translate,
        "help_lines": lambda allowed: ["`/help` -- Help", "`/version` -- Version"]
            if allowed is None else ["`/help` -- Help"],
    }


@pytest.mark.parametrize("name", ["version", "egress", "profile", "bundles", "help", "commands"])
def test_all_executors_ignore_surface_and_need_no_cli_import(name, subsystems, monkeypatch):
    requested = []
    original = builtins.__import__
    def guarded(module, *args, **kwargs):
        if module == "hermes_cli" or module.startswith("hermes_cli."):
            requested.append(module)
            raise AssertionError(f"shared executor imported {module}")
        return original(module, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    replies = [execute_command(name, CommandContext(surface=s, options=inputs()))
               for s in ("cli", "gateway", "tui", "acp")]
    assert all(reply == replies[0] for reply in replies)
    assert replies[0].text
    assert not requested


def test_profile_keeps_canonical_identity_and_supplied_label():
    reply = execute_command("profile", CommandContext(options=inputs()))
    assert reply.text == "Profile: Research (research)\nHome: ~/.hermes/profiles/research"
    assert reply.data == {"profile": "research", "home": "~/.hermes/profiles/research"}


def test_bundles_preserve_members_and_default_description(subsystems):
    reply = execute_command("bundles", CommandContext())
    assert "/review — Load 2 skills (2 skills)\n    · audit\n    · check" in reply.text
    assert reply.data["dir"] == str(Path("/fixture/bundles"))
    assert reply.data["bundles"][0]["slug"] == "review"


def test_empty_bundles_preserve_creation_hint(subsystems, monkeypatch):
    import agent.skill_bundles as bundles
    monkeypatch.setattr(bundles, "list_bundles", lambda: [])
    reply = execute_command("bundles", CommandContext())
    assert reply.text.startswith("No skill bundles installed.\nCreate one with: hermes bundles create")
    assert reply.data == {"bundles": [], "dir": str(Path("/fixture/bundles"))}


def test_restricted_help_never_discovers_skills(subsystems, monkeypatch):
    import agent.skill_commands as skills
    def unexpected():
        pytest.fail("restricted help must not discover skill commands")
    monkeypatch.setattr(skills, "get_skill_commands", unexpected)
    reply = execute_command("help", CommandContext(options={**inputs(), "allowed_commands": {"help"}}))
    assert reply.text == "gateway.help.header\n`/help` -- Help"
    assert reply.format == "markdown"


def test_commands_preserve_skills_collisions_and_pagination(subsystems):
    reply = execute_command("commands", CommandContext(options={**inputs(), "page_size": 500}))
    assert "`/audit` — gateway.commands.default_desc" in reply.text
    assert "`/tidy` — Tidy files" in reply.text
    assert "⚠ slash command /handoff unavailable" in reply.text
    last = execute_command("commands", CommandContext(args="999", options={**inputs(), "page_size": 2}))
    assert "gateway.commands.out_of_range:requested=999,page=" in last.text
    assert "gateway.commands.nav_prev:page=" in last.text
    assert last.format == "markdown"


def test_invalid_page_does_not_discover_catalog():
    def unexpected(_):
        pytest.fail("usage error must not discover commands")
    reply = execute_command("commands", CommandContext(args="oops", options={
        "translate": translate, "help_lines": unexpected}))
    assert reply == CommandReply("gateway.commands.usage", format="markdown")


@pytest.mark.parametrize("size", [None, "oops", 0, -10])
def test_invalid_or_small_page_size_remains_usable(size, subsystems):
    reply = execute_command("commands", CommandContext(args="-1", options={**inputs(), "page_size": size}))
    assert "gateway.commands.header:" in reply.text
    assert "gateway.commands.out_of_range:requested=-1,page=1" in reply.text


def test_lookup_failure_and_non_shared_dispatch_are_preserved():
    assert run_execute(resolve_command("model"), CommandContext()) is None
    for name in ("model", "phase7-unknown"):
        with pytest.raises(LookupError, match="no registry-owned executor"):
            execute_command(name, CommandContext())


def test_execution_has_no_cli_or_gateway_implementation_imports():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "commands" / "execution.py").read_text(encoding="utf8"))
    for node in ast.walk(tree):
        names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                 else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
        assert not any(n == "cli" or n.startswith(("hermes_cli", "gateway")) for n in names)
    assert not (root / "hermes_cli" / "slash_exec.py").exists()
