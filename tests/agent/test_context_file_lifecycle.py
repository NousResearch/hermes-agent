"""Startup snapshots survive progressive discovery, cold restore and cache-prefix reconstruction."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.context_file_state import (
    ContextFileSnapshot, bind_context_snapshot, context_manifest_kwargs, persist_system_prompt, restore_context_manifest,
)
from agent.conversation_loop import _restore_or_build_system_prompt
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.system_prompt import build_system_prompt, reconstruct_static_prefix
from hermes_state import SessionDB
from tests.agent.test_system_prompt import _make_agent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home, project = tmp_path / "profile", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (project / ".git").mkdir()
    for name in ("first", "second", "copy"):
        (project / name).mkdir()
    first = project / "first" / "AGENTS.md"
    first.write_text("shared first rules")
    (home / "config.yaml").write_text(json.dumps({"context": {"external_files": [str(first)]}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    monkeypatch.setattr("agent.system_prompt.resolve_context_cwd", lambda **_kw: str(project))
    with SessionDB(home / "state.db") as db:
        db.create_session("s1", source="cli", model="test-model")
        yield home, project, first, db


def make_agent(db, project):
    agent = _make_agent(
        _session_db=db, session_id="s1", model="test-model", provider="openrouter", platform="cli",
        _subdirectory_hints=SubdirectoryHintTracker(str(project)), tools=[], _use_prompt_caching=False,
        _platform_hint_overrides=None, _surface_switch_note="", _gateway_turn_context_notes="",
        _emit_diagnostic_status=lambda *_args: None,
    )
    agent._build_system_prompt = lambda message=None: build_system_prompt(agent, message)
    return agent


def visit(agent, directory):
    return agent._subdirectory_hints.check_tool_call("read_file", {"path": str(directory / "module.py")})


def test_actual_prompt_snapshot_seeds_progressive_identity_and_content_dedup(workspace):
    _, project, first, db = workspace
    agent = make_agent(db, project)
    prompt = build_system_prompt(agent)
    assert "shared first rules" in prompt
    (project / "copy" / "AGENTS.md").write_text(first.read_text())
    assert visit(agent, project / "first") is None
    assert visit(agent, project / "copy") is None
    (project / "second" / "AGENTS.md").write_text("new second rules")
    assert "new second rules" in visit(agent, project / "second")


def test_cold_restore_uses_frozen_manifest_and_never_current_config_files(workspace):
    home, project, first, db = workspace
    initial = make_agent(db, project)
    prompt = initial._cached_system_prompt = build_system_prompt(initial)
    persist_system_prompt(initial, prompt)
    saved_manifest = db.get_session("s1")["context_file_identities"]
    first.write_text("changed same inode")
    second = project / "second" / "AGENTS.md"
    second.write_text("newly configured but never snapshotted")
    (home / "config.yaml").write_text(json.dumps({"context": {"external_files": [str(second)]}}))
    resumed = make_agent(db, project)
    resumed._cached_system_prompt = None
    _restore_or_build_system_prompt(resumed, None, [{"role": "user", "content": "continue"}])
    assert resumed._cached_system_prompt == prompt
    assert context_manifest_kwargs(resumed, prompt)["context_file_identities"] == saved_manifest
    assert visit(resumed, project / "first") is None
    assert "newly configured but never snapshotted" in visit(resumed, project / "second")


def test_static_prefix_reconstruction_does_not_replace_frozen_manifest(workspace):
    home, project, _first, db = workspace
    agent = make_agent(db, project)
    prompt = agent._cached_system_prompt = build_system_prompt(agent)
    saved = context_manifest_kwargs(agent, prompt)
    second = project / "second" / "AGENTS.md"
    second.write_text("new rules at reconstruction time")
    (home / "config.yaml").write_text(json.dumps({"context": {"external_files": [str(second)]}}))
    agent._cached_system_prompt_static = None
    agent._use_prompt_caching = True
    reconstruct_static_prefix(agent)
    assert agent._cached_system_prompt == prompt
    assert context_manifest_kwargs(agent, prompt) == saved
    assert "new rules at reconstruction time" in visit(agent, project / "second")


@pytest.mark.parametrize("manifest", [None, "broken JSON", '{"version":99,"identities":[]}'])
def test_legacy_or_invalid_manifest_does_not_force_prompt_rebuild(workspace, manifest):
    _, project, _first, db = workspace
    agent = make_agent(db, project)
    prompt = build_system_prompt(agent)
    db.update_system_prompt("s1", prompt, manifest)
    resumed = make_agent(db, project)
    resumed._cached_system_prompt = None

    def forbidden_build(_message=None):
        raise AssertionError("legacy rows must keep their frozen prompt")

    resumed._build_system_prompt = forbidden_build
    _restore_or_build_system_prompt(resumed, None, [{"role": "user", "content": "continue"}])
    assert resumed._cached_system_prompt == prompt
    assert context_manifest_kwargs(resumed, prompt) == {}


def test_workspace_rebind_preserves_shared_startup_digests(workspace, tmp_path):
    _, project, first, db = workspace
    agent = make_agent(db, project)
    build_system_prompt(agent)
    other = tmp_path / "other_project"
    (other / "area").mkdir(parents=True)
    (other / "area" / "AGENTS.md").write_text(first.read_text())
    agent._subdirectory_hints.rebind_working_dir(str(other))
    assert visit(agent, other / "area") is None


def test_manifest_is_not_reused_for_different_prompt_bytes():
    agent = SimpleNamespace()
    restore_context_manifest(agent, ContextFileSnapshot({("inode", 1, 2)}).serialize(), "frozen")
    assert context_manifest_kwargs(agent, "frozen")
    assert context_manifest_kwargs(agent, "different") == {}
    restore_context_manifest(agent, ContextFileSnapshot({("inode", 1, 2)}).serialize(), "")
    assert context_manifest_kwargs(agent, "") == {}


def test_july_identity_only_manifest_is_supported_without_filesystem_reads():
    snapshot = ContextFileSnapshot.parse('{"version":1,"identities":[{"kind":"inode","device":1,"inode":2}]}')
    assert snapshot.identities == {("inode", 1, 2)} and not snapshot.digests


@pytest.mark.parametrize("in_place", [False, True])
def test_real_compression_runtime_persists_rebuilt_prompt_with_its_manifest(tmp_path, in_place):
    from tests.agent.test_compression_rotation_state import _build_agent_with_db, _msgs

    with SessionDB(tmp_path / "compression.db") as db:
        db.create_session("parent", source="cli")
        agent = _build_agent_with_db(db, "parent")
        agent.compression_in_place = in_place
        old = ContextFileSnapshot({("inode", 1, 10)})
        new = ContextFileSnapshot({("inode", 1, 20)})
        agent._cached_system_prompt = "old prompt"
        bind_context_snapshot(agent, old, "old prompt")
        persist_system_prompt(agent, "old prompt")

        def rebuilt(_message=None):
            bind_context_snapshot(agent, new, "new prompt")
            return "new prompt"

        agent._build_system_prompt = rebuilt
        try:
            _messages, prompt = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
            row = db.get_session(agent.session_id)
            assert prompt == row["system_prompt"] == "new prompt"
            assert row["context_file_identities"] == new.serialize()
            assert (agent.session_id == "parent") is in_place
        finally:
            agent.close()
