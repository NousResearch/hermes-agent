"""Opt-in skill inheritance through real, offline native-agent construction."""
from __future__ import annotations

import pytest
import hermes_yaml as yaml


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.chdir(tmp_path)
    token = set_hermes_home_override(str(home))
    yield home
    reset_hermes_home_override(token)


def write_config(home, skills):
    (home / "config.yaml").write_text(yaml.safe_dump({
        "skills": skills,
        "model": {"context_length": 128000},
        "delegation": {"orchestrator_enabled": True, "max_spawn_depth": 3},
        "curator": {"enabled": False},
    }), encoding="utf-8")


def write_skill(home, name, body):
    target = home / "skills" / name / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\nname: {name}\ndescription: Fixture.\n---\n\n{body}\n", encoding="utf-8")
    return target


def main_agent(**overrides):
    from run_agent import AIAgent

    args = dict(model="offline-fixture", provider="openai-compat", api_key="offline-fixture",
                base_url="http://127.0.0.1:9/v1", enabled_toolsets=["skills", "delegation"],
                quiet_mode=True, skip_memory=True, skip_background_review=True,
                save_trajectories=False)
    return AIAgent(**(args | overrides))


def child_agent(parent, goal="Observe inheritance", context=None):
    from tools.delegate_tool_results import _build_child_preserving_parent_tools

    return _build_child_preserving_parent_tools(
        task_index=0, goal=goal, context=context, toolsets=None, model=None,
        max_iterations=1, task_count=1, parent_agent=parent)


def stable(agent):
    from agent.system_prompt import build_system_prompt_parts

    return build_system_prompt_parts(agent)["stable"]


def test_native_children_inherit_without_parent_text(profile):
    write_skill(profile, "shared", "SHARED_POLICY_MARKER")
    write_skill(profile, "main-only", "MAIN_ONLY_POLICY_MARKER")
    write_config(profile, {"auto_load": ["shared", "main-only"],
                           "inherit_to_subagents": ["shared"]})
    parent = main_agent()
    child = nested = None
    try:
        assert stable(parent).count("SHARED_POLICY_MARKER") == 1
        assert "MAIN_ONLY_POLICY_MARKER" in stable(parent)
        child = child_agent(parent)
        assert child.skip_context_files is True
        assert "SHARED_POLICY_MARKER" not in child.ephemeral_system_prompt
        assert stable(child).count("SHARED_POLICY_MARKER") == 1
        assert "MAIN_ONLY_POLICY_MARKER" not in stable(child)
        assert "delegate_task" in child.valid_tool_names
        nested = child_agent(child)
        assert stable(nested).count("SHARED_POLICY_MARKER") == 1
        assert "MAIN_ONLY_POLICY_MARKER" not in stable(nested)
        assert nested._delegate_depth == 2
    finally:
        for agent in (nested, child, parent):
            if agent is not None:
                agent.close()


def test_config_set_rejects_nonlist_before_writing(profile):
    from hermes_cli.config import _refuse_container_type_mismatch

    with pytest.raises(SystemExit):
        _refuse_container_type_mismatch("skills.inherit_to_subagents", "shared", {}, False)


@pytest.mark.parametrize("inherited", [False, True])
def test_configured_aliases_render_once(profile, inherited):
    from agent.skill_commands import build_auto_load_prompt

    path = write_skill(profile, "shared", "ALIAS_POLICY_MARKER")
    key = "inherit_to_subagents" if inherited else "auto_load"
    write_config(profile, {key: ["shared", str(path.parent), " shared ", "missing", 7]})
    prompt, loaded, missing = build_auto_load_prompt(home_override=profile, inherited=inherited)
    assert prompt.count("ALIAS_POLICY_MARKER") == 1
    assert loaded == ["shared"]
    assert missing == ["missing"]


def test_subagent_platform_disabled_is_respected(profile):
    from agent.skill_commands import build_auto_load_prompt

    write_skill(profile, "shared", "DISABLED_POLICY_MARKER")
    write_config(profile, {"inherit_to_subagents": ["shared"],
                           "platform_disabled": {"subagent": ["shared"]}})
    prompt, loaded, missing = build_auto_load_prompt(home_override=profile, inherited=True)
    assert not prompt and not loaded
    assert missing == ["shared"]


@pytest.mark.parametrize("value", [None, [], "shared", {"shared": True}, 7, False])
def test_absent_empty_or_malformed_allowlist_is_opt_out(profile, value):
    write_skill(profile, "shared", "OPT_OUT_POLICY_MARKER")
    skills = {"auto_load": ["shared"]}
    if value is not None:
        skills["inherit_to_subagents"] = value
    write_config(profile, skills)
    parent = main_agent()
    child = None
    try:
        child = child_agent(parent)
        assert "OPT_OUT_POLICY_MARKER" not in stable(child)
        assert "OPT_OUT_POLICY_MARKER" in stable(parent)
    finally:
        if child:
            child.close()
        parent.close()


@pytest.mark.parametrize("gate", ["disabled", "ambient-disabled", "ignore-rules", "no-tools"])
def test_inheritance_obeys_disabling_gates(profile, monkeypatch, gate):
    write_skill(profile, "shared", "GATED_POLICY_MARKER")
    skills = {"inherit_to_subagents": ["shared"]}
    if gate == "disabled":
        skills["disabled"] = ["shared"]
    if gate == "ambient-disabled":
        monkeypatch.setenv("HERMES_PLATFORM", "desktop")
        skills["platform_disabled"] = {"desktop": ["shared"]}
    if gate == "ignore-rules":
        monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    write_config(profile, skills)
    parent = main_agent(enabled_toolsets=[] if gate == "no-tools" else ["skills"])
    child = None
    try:
        child = child_agent(parent)
        assert "GATED_POLICY_MARKER" not in stable(child)
        if gate == "no-tools":
            assert not {"skills_list", "skill_view", "skill_manage"} & child.valid_tool_names
    finally:
        if child:
            child.close()
        parent.close()


def test_independent_lists_context_isolation_and_frozen_prefix(profile):
    from agent.system_prompt import build_system_prompt_parts, invalidate_system_prompt

    path = write_skill(profile, "child-only", "CHILD_ONLY_POLICY_MARKER")
    write_skill(profile, "main-only", "MAIN_ONLY_POLICY_MARKER")
    write_config(profile, {"auto_load": ["main-only"], "inherit_to_subagents": ["child-only"]})
    (profile / "SOUL.md").write_text("PERSONALITY_POLICY_MARKER", encoding="utf-8")
    (profile.parent / "AGENTS.md").write_text("PROJECT_POLICY_MARKER", encoding="utf-8")
    parent = main_agent()
    child = internal = None
    try:
        parent_parts = build_system_prompt_parts(parent)
        assert "PERSONALITY_POLICY_MARKER" in parent_parts["stable"]
        assert "PROJECT_POLICY_MARKER" in parent_parts["context"]
        assert "CHILD_ONLY_POLICY_MARKER" not in parent_parts["stable"]
        child = child_agent(parent, context="TASK_INSTRUCTION_MARKER")
        first = stable(child)
        assert first.count("CHILD_ONLY_POLICY_MARKER") == 1
        parts = build_system_prompt_parts(child)
        assert "TASK_INSTRUCTION_MARKER" in child.ephemeral_system_prompt
        for marker in ("PERSONALITY_POLICY_MARKER", "PROJECT_POLICY_MARKER", "MAIN_ONLY_POLICY_MARKER"):
            assert marker not in "\n".join(parts.values())
        path.write_text("---\nname: child-only\ndescription: Changed.\n---\nCHANGED_POLICY_MARKER", encoding="utf-8")
        write_config(profile, {"inherit_to_subagents": []})
        invalidate_system_prompt(child)
        assert stable(child) == first
        internal = main_agent(skip_context_files=True, platform="curator")
        assert "CHILD_ONLY_POLICY_MARKER" not in stable(internal)
    finally:
        for agent in (internal, child, parent):
            if agent:
                agent.close()


def test_bound_profile_wins_over_launch_home(profile, tmp_path, monkeypatch):
    write_skill(profile, "shared", "OWNED_POLICY_MARKER")
    write_config(profile, {"inherit_to_subagents": ["shared"]})
    other = tmp_path / "other"
    other.mkdir()
    write_skill(other, "shared", "OTHER_POLICY_MARKER")
    write_config(other, {"inherit_to_subagents": ["shared"]})
    parent = main_agent()
    child = None
    try:
        child = child_agent(parent)
        monkeypatch.setenv("HERMES_HOME", str(other))
        prompt = stable(child)
        assert prompt.count("OWNED_POLICY_MARKER") == 1
        assert "OTHER_POLICY_MARKER" not in prompt
    finally:
        if child:
            child.close()
        parent.close()


@pytest.mark.parametrize("inherited", [False, True])
def test_distinct_sources_with_same_declared_name_both_load(profile, inherited):
    from agent.skill_commands import build_auto_load_prompt

    for category, body in (("a", "FIRST_COPY_MARKER"), ("b", "SECOND_COPY_MARKER")):
        target = profile / "skills" / category / "same" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(f"---\nname: same\ndescription: Fixture.\n---\n{body}\n", encoding="utf-8")
    key = "inherit_to_subagents" if inherited else "auto_load"
    write_config(profile, {key: ["a/same", "b/same"]})
    prompt, loaded, missing = build_auto_load_prompt(home_override=profile, inherited=inherited)
    assert prompt.count("FIRST_COPY_MARKER") == 1
    assert prompt.count("SECOND_COPY_MARKER") == 1
    assert loaded == ["same", "same"]
    assert not missing


@pytest.mark.parametrize("disabled_platform", ["subagent", "desktop"])
@pytest.mark.parametrize("identifier_kind", ["canonical", "junction", "absolute-junction"])
def test_native_child_exclusion_covers_physical_directory_alias(profile, monkeypatch, identifier_kind, disabled_platform):
    import os
    import subprocess

    for category, marker in (("a", "PHYSICAL_DISABLED_MARKER"), ("b", "OTHER_SOURCE_MARKER")):
        target = profile / "skills" / category / "same" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(f"---\nname: same\ndescription: Fixture.\n---\n{marker}\n", encoding="utf-8")
    alias = profile / "skills" / "alias"
    original = profile / "skills" / "a" / "same"
    if os.name == "nt":
        subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(alias), str(original)],
                       check=True, capture_output=True)
    else:
        alias.symlink_to(original, target_is_directory=True)
    assert (alias / "SKILL.md").resolve() == (original / "SKILL.md").resolve()
    identifier = {"canonical": "a/same", "junction": "alias", "absolute-junction": str(alias)}[identifier_kind]
    monkeypatch.setenv("HERMES_PLATFORM", "desktop")
    write_config(profile, {"inherit_to_subagents": [identifier],
                           "platform_disabled": {disabled_platform: ["a/same"]}})
    parent = main_agent()
    child = None
    try:
        child = child_agent(parent)
        assert child.skip_context_files is True
        assert "PHYSICAL_DISABLED_MARKER" not in stable(child)
    finally:
        if child:
            child.close()
        parent.close()


def test_subagent_path_disabling_covers_absolute_alias(profile):
    from agent.skill_commands import build_auto_load_prompt

    for category in ("a", "b"):
        target = profile / "skills" / category / "same" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(f"---\nname: same\ndescription: Fixture.\n---\n{category}_DISABLED_COPY\n", encoding="utf-8")
    write_config(profile, {"inherit_to_subagents": [str(profile / "skills" / "a" / "same"), "b/same"],
                           "platform_disabled": {"subagent": ["a/same"]}})
    prompt, loaded, missing = build_auto_load_prompt(home_override=profile, inherited=True)
    assert "a_DISABLED_COPY" not in prompt
    assert "b_DISABLED_COPY" in prompt
    assert loaded == ["same"]
    assert missing


def test_subagent_path_disabling_does_not_block_shadowed_external_copy(profile, tmp_path):
    from tools.skills_tool import _is_resolved_skill_disabled, _skill_search_dirs

    external = tmp_path / "external"
    for home, name, body in ((profile, "a/same", "LOCAL_COPY"),
                             (profile, "b/same", "OTHER_LOCAL_COPY"),
                             (external, "a/same", "EXTERNAL_COPY")):
        target = write_skill(home, name, body)
        target.write_text(f"---\nname: same\ndescription: Fixture.\n---\n{body}\n", encoding="utf-8")
    write_config(profile, {"external_dirs": [str(external / "skills")],
                           "platform_disabled": {"subagent": ["a/same"]}})
    roots, _ = _skill_search_dirs()
    dirs = [d for _t, d in roots]
    local = profile / "skills" / "a" / "same" / "SKILL.md"
    assert _is_resolved_skill_disabled("same", local.parent, local, dirs, platform="subagent")
    assert not _is_resolved_skill_disabled("same", target.parent, target, dirs, platform="subagent")


@pytest.mark.parametrize("inherited", [False, True])
def test_plugin_and_filesystem_aliases_share_source_identity(profile, monkeypatch, inherited):
    from agent.skill_commands import build_auto_load_prompt
    from hermes_cli import plugins

    target = write_skill(profile, "shared", "PLUGIN_ALIAS_MARKER")
    manager = plugins.PluginManager()
    manager._plugin_skills["fixture:shared"] = {
        "path": target, "plugin": "fixture", "bare_name": "shared", "description": "Fixture."}
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    key = "inherit_to_subagents" if inherited else "auto_load"
    write_config(profile, {key: ["fixture:shared", "shared", str(target.parent)]})
    prompt, loaded, missing = build_auto_load_prompt(home_override=profile, inherited=inherited)
    assert prompt.count("PLUGIN_ALIAS_MARKER") == 1
    assert loaded == ["fixture:shared"]
    assert not missing


@pytest.mark.parametrize("background", [False, True])
def test_production_dispatch_recurses_without_forwarding(profile, monkeypatch, background):
    """Real construction/dispatch/prompt machinery; only model execution is replaced."""
    import json
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task
    from gateway.session_context import set_session_vars
    from tools.process_registry import process_registry

    write_skill(profile, "shared", "DISPATCH_POLICY_MARKER")
    write_config(profile, {"inherit_to_subagents": ["shared"]})
    observations = []

    def observe(self, user_message, **kwargs):
        assert stable(self).count("DISPATCH_POLICY_MARKER") == 1
        assert "DISPATCH_POLICY_MARKER" not in user_message
        assert "DISPATCH_POLICY_MARKER" not in self.ephemeral_system_prompt
        observations.append((self._delegate_depth, user_message))
        if self._delegate_depth == 1:
            result = json.loads(delegate_task(goal="nested goal", parent_agent=self, background=False))
            assert result["results"][0]["status"] == "completed", result
        return {"final_response": "offline observer completed", "messages": [],
                "completed": True, "api_calls": 0}

    monkeypatch.setattr(AIAgent, "run_conversation", observe)
    parent = main_agent()
    set_session_vars(platform="cli", session_id=parent.session_id, session_key=parent.session_id,
                     session_history_delivery="1", async_delivery=True)
    try:
        result = json.loads(delegate_task(goal="direct goal", parent_agent=parent, background=background))
        if background:
            assert result["status"] == "dispatched", result
            event = process_registry.completion_queue.get(timeout=30)
            assert event["type"] == "async_delegation", event
        else:
            assert result["results"][0]["status"] == "completed", result
        assert observations == [(1, "direct goal"), (2, "nested goal")]
    finally:
        parent.close()
