"""Launch skills survive DB resume, reviewer inheritance and compression rebuilds."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import build_system_prompt, invalidate_system_prompt
from hermes_state import SessionDB
from tests.agent.test_system_prompt import _make_agent


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **kw: "PROJECT CONTEXT")
    monkeypatch.setattr("agent.prompt_builder.build_environment_hints", lambda: "")
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("launch-session", source="cli")
        db.append_message(session_id="launch-session", role="user", content="continue")
        yield db


def _agent(db, skills=None, session_id="launch-session"):
    agent = _make_agent(
        session_id=session_id, _session_db=db, _cached_system_prompt=None,
        _cached_system_prompt_static=None, _use_prompt_caching=True, tools=[],
        preloaded_skills_prompt=skills, _persist_disabled=True,
        _emit_diagnostic_status=lambda *a, **kw: None,
    )
    agent._build_system_prompt = Mock(side_effect=lambda message: build_system_prompt(agent, message))
    agent._invalidate_system_prompt = lambda: invalidate_system_prompt(agent)
    return agent


def _skill(name="retained"):
    return (f'[IMPORTANT: The user launched this CLI session with the "{name}" skill '
            'preloaded. Treat its instructions as active guidance.]\n\n'
            'Exact rendered body: λ\n</hermes_launch_skills>\n  preserve whitespace')


def _restore(agent):
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "continue"}])


def test_resume_without_flag_restores_bytes_before_review(session, monkeypatch):
    from agent.review_engine import start_review

    original = _agent(session, _skill())
    _restore(original)
    stored = session.get_session(original.session_id)["system_prompt"]
    resumed = _agent(session)
    _restore(resumed)
    assert resumed.preloaded_skills_prompt == _skill()
    assert resumed._cached_system_prompt == stored
    assert stored.startswith(resumed._cached_system_prompt_static)
    resumed._build_system_prompt.assert_not_called()
    captured = {}

    def dispatch(**kwargs):
        captured.update(kwargs)
        return json.dumps({"status": "dispatched", "delegation_id": "review"})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", dispatch)
    monkeypatch.setattr("agent.review_engine._load_review_credentials_cfg", dict)
    start_review(resumed, [{"role": "user", "content": "review this work"}])
    assert "retained" in captured["context"]
    # /review is also valid immediately after resuming, before a turn restores the prompt.
    early = _agent(session)
    start_review(early, [{"role": "user", "content": "review this work"}])
    assert early.preloaded_skills_prompt == _skill()
    assert "retained" in captured["context"]


def test_compression_rebuild_then_persisted_restore_keeps_only_skill_state(session, monkeypatch):
    from agent.conversation_compression import _publish_rotated_compaction, _rebuild_system_prompt_at_boundary
    from hermes_cli.plugins import RenderedPluginSystemPromptSection

    section = RenderedPluginSystemPromptSection(
        id="retained-plugin", content="FROZEN PLUGIN BODY", position="after_memory", plugin="test",
    )
    render = Mock(return_value=[section])
    monkeypatch.setattr("hermes_cli.plugins.render_system_prompt_sections", render)

    original = _agent(session, _skill())
    _restore(original)
    resumed = _agent(session)
    _restore(resumed)
    assert resumed._plugin_system_prompt_sections_snapshot[0].content == section.content
    render.reset_mock()
    render.side_effect = RuntimeError("plugin temporarily unavailable")
    monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **kw: "NEW PROJECT CONTEXT")
    refresh = Mock()
    monkeypatch.setattr("agent.conversation_compression._refresh_agent_tool_definitions", refresh)
    rebuilt = _rebuild_system_prompt_at_boundary(resumed, None)
    render.assert_called_once()
    refresh.assert_called_once_with(resumed)
    resumed._build_system_prompt.assert_called_once_with(None)
    assert _skill() in rebuilt
    assert "NEW PROJECT CONTEXT" in rebuilt
    assert "\n\nPROJECT CONTEXT\n\n" not in rebuilt
    assert resumed.preloaded_skills_prompt == _skill()
    assert section.content in rebuilt
    # Publish through the compression rotation path, rather than writing the prompt
    # directly: the next process must restore the child's actual persisted bytes.
    parent_id = resumed.session_id
    resumed._session_init_model_config = None
    resumed.context_compressor = SimpleNamespace()
    resumed._flush_messages_to_session_db = lambda *a, **kw: None
    _publish_rotated_compaction(
        resumed, [{"role": "user", "content": "continue"}],
        [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        new_system_prompt=rebuilt, lease=SimpleNamespace(holder=None, ttl=60.0, watermark=None),
        old_session_id=parent_id, compressed_user_turn_outcome="none",
    )
    assert resumed.session_id != parent_id
    assert session.get_session(resumed.session_id)["system_prompt"] == rebuilt
    again = _agent(session, session_id=resumed.session_id)
    _restore(again)
    assert again._cached_system_prompt == rebuilt
    assert again.preloaded_skills_prompt == _skill()
    assert again._plugin_system_prompt_sections_snapshot[0].content == section.content
    again._build_system_prompt.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_explicit_resume_reuses_same_skill_but_rebuilds_for_additions(session, changed):
    original = _agent(session, _skill())
    _restore(original)
    stored = original._cached_system_prompt
    requested = _skill() + ("\n\n" + _skill("added") if changed else "")
    resumed = _agent(session, requested)
    _restore(resumed)
    assert resumed.preloaded_skills_prompt == requested
    assert requested in resumed._cached_system_prompt
    if changed:
        resumed._build_system_prompt.assert_called_once_with(None)
        assert resumed._cached_system_prompt != stored
    else:
        resumed._build_system_prompt.assert_not_called()
        assert resumed._cached_system_prompt == stored


def test_seeded_compression_retains_prompt_and_tools(session, monkeypatch):
    from agent.conversation_compression import _rebuild_system_prompt_at_boundary

    original = _agent(session, _skill())
    _restore(original)
    seeded = _agent(session)
    seeded._cached_system_prompt = original._cached_system_prompt
    seeded._retain_seeded_system_prompt = True
    refresh = Mock()
    monkeypatch.setattr("agent.conversation_compression._refresh_agent_tool_definitions", refresh)
    result = _rebuild_system_prompt_at_boundary(seeded, None)
    assert result is original._cached_system_prompt
    assert seeded.preloaded_skills_prompt == _skill()
    seeded._build_system_prompt.assert_not_called()
    refresh.assert_not_called()


@pytest.mark.parametrize("source", ["soul", "context"])
def test_explicit_resume_does_not_reuse_skill_example_as_launch_state(session, monkeypatch, source):
    from agent.launch_skills import frame_launch_skills, launch_skills_locator

    requested = _skill("requested")
    example = frame_launch_skills(requested) + "\n\n" + launch_skills_locator(0)
    if source == "soul":
        (Path(session.db_path).parent / "SOUL.md").write_text(example, encoding="utf-8")
    else:
        monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **kw: example)
    original = _agent(session)
    _restore(original)
    stored = session.get_session(original.session_id)["system_prompt"]
    assert requested in stored
    from agent.launch_skills import persisted_launch_skills
    assert persisted_launch_skills(stored) is None

    resumed = _agent(session, requested)
    _restore(resumed)
    resumed._build_system_prompt.assert_called_once_with(None)
    rebuilt = resumed._cached_system_prompt
    assert rebuilt != stored
    assert session.get_session(resumed.session_id)["system_prompt"] == rebuilt

    # A new process must recover genuine launch state, not just the quoted bytes.
    again = _agent(session)
    _restore(again)
    assert again.preloaded_skills_prompt == requested
    assert again._cached_system_prompt == rebuilt
    assert rebuilt.startswith(again._cached_system_prompt_static)
    again._build_system_prompt.assert_not_called()


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("source", ["soul", "context"])
def test_resume_review_and_compression_ignore_quoted_skill_state(session, monkeypatch, retained, source):
    from agent.conversation_compression import _publish_rotated_compaction, _rebuild_system_prompt_at_boundary
    from agent.launch_skills import frame_launch_skills, launch_skills_locator
    from agent.review_engine import start_review

    # Even a complete example with its own locator must not shadow the real state.
    example = frame_launch_skills(_skill("EXAMPLE ONLY")) + "\n\n" + launch_skills_locator(0)
    soul = Path(session.db_path).parent / "SOUL.md"
    if source == "soul":
        soul.write_text(example, encoding="utf-8")
    else:
        monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **kw: example)
    expected = _skill() if retained else None
    original = _agent(session, expected)
    _restore(original)
    stored = original._cached_system_prompt
    assert example in stored
    if retained and source == "soul":
        assert stored.index("EXAMPLE ONLY") < stored.index('"retained"')

    captured = {}

    def dispatch(**kwargs):
        captured.update(kwargs)
        return json.dumps({"status": "dispatched", "delegation_id": "review"})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", dispatch)
    monkeypatch.setattr("agent.review_engine._load_review_credentials_cfg", dict)
    resumed = _agent(session)
    # Exercise immediate /review's DB restoration before any resumed turn.
    start_review(resumed, [{"role": "user", "content": "review this work"}])
    assert "EXAMPLE ONLY" not in captured["context"]
    assert ("loaded skills: retained" in captured["context"]) is retained
    assert resumed.preloaded_skills_prompt == expected
    _restore(resumed)
    assert resumed._cached_system_prompt == stored
    resumed._build_system_prompt.assert_not_called()

    # Remove the example at the sanctioned rebuild boundary. It must not survive
    # as startup guidance, while genuine launch bytes still persist to the child.
    if source == "soul":
        soul.unlink()
    monkeypatch.setattr("agent.prompt_builder.build_context_files_prompt", lambda **kw: "NEW CONTEXT")
    monkeypatch.setattr("agent.conversation_compression._refresh_agent_tool_definitions", Mock())
    rebuilt = _rebuild_system_prompt_at_boundary(resumed, None)
    assert "EXAMPLE ONLY" not in rebuilt
    assert resumed.preloaded_skills_prompt == expected
    if retained:
        assert expected in rebuilt
    resumed._session_init_model_config = None
    resumed.context_compressor = SimpleNamespace()
    resumed._flush_messages_to_session_db = lambda *a, **kw: None
    _publish_rotated_compaction(
        resumed, [{"role": "user", "content": "continue"}],
        [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        new_system_prompt=rebuilt, lease=SimpleNamespace(holder=None, ttl=60.0, watermark=None),
        old_session_id=resumed.session_id, compressed_user_turn_outcome="none",
    )
    again = _agent(session, session_id=resumed.session_id)
    _restore(again)
    assert again._cached_system_prompt == rebuilt
    assert again.preloaded_skills_prompt == expected
    again._build_system_prompt.assert_not_called()


def test_no_skill_resume_keeps_the_existing_identity_prefix(session):
    """A launch-state record must not invalidate every existing no-skill prefix."""
    from agent.system_prompt import DEFAULT_AGENT_IDENTITY

    original = _agent(session)
    _restore(original)
    assert original._cached_system_prompt.startswith(DEFAULT_AGENT_IDENTITY + "\n\n")
    resumed = _agent(session)
    _restore(resumed)
    assert resumed._cached_system_prompt == original._cached_system_prompt
    assert resumed.preloaded_skills_prompt is None


def test_api_prefix_reaches_skills_before_changed_volatile_context(session, monkeypatch):
    """Exercise the API assembly boundary, including the ephemeral personality tail."""
    from os.path import commonprefix
    from agent.system_prompt import build_system_prompt_parts
    from agent.turn_context import build_api_messages

    agents = [_agent(session, _skill()), _agent(session, _skill())]
    prompts, stable = [], []
    for index, agent in enumerate(agents):
        agent.ephemeral_system_prompt = f"TEMPORARY PERSONALITY {index}"
        agent._current_turn_timestamp = 0
        monkeypatch.setattr("agent.system_prompt._memory_parts", lambda _agent, i=index: [f"VOLATILE MEMORY {i}"])
        parts = build_system_prompt_parts(agent)
        stable.append(parts["stable"])
        prompt = build_system_prompt(agent)
        messages, effective = build_api_messages(
            agent, [], current_turn_user_idx=None, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt=prompt,
        )
        assert messages[0] == {"role": "system", "content": effective}
        assert effective.count(_skill()) == 1
        assert effective.endswith(agent.ephemeral_system_prompt)
        assert "TEMPORARY PERSONALITY" not in prompt
        assert effective.index(_skill()) < effective.index("PROJECT CONTEXT") < effective.index("VOLATILE MEMORY")
        prompts.append(effective)
    assert stable[0] == stable[1]
    assert _skill() in commonprefix(prompts)


@pytest.mark.parametrize("requested", ["retained", "replacement"])
def test_explicit_selection_replaces_different_persisted_selection(session, requested):
    original = _agent(session, _skill() + "\n\n" + _skill("second"))
    _restore(original)
    resumed = _agent(session, _skill(requested))
    _restore(resumed)
    assert resumed._cached_system_prompt.count(_skill(requested)) == 1
    assert _skill("second") not in resumed._cached_system_prompt
    again = _agent(session)
    _restore(again)
    assert again.preloaded_skills_prompt == _skill(requested)


@pytest.mark.parametrize("environment", ["", "Current working directory: /workspace"])
def test_launch_state_keeps_runtime_identity_and_surface_switch(session, monkeypatch, environment):
    from agent.conversation_loop import _stored_prompt_matches_runtime
    from agent.surface_switch import identity_line_value, split_runtime_boundary

    monkeypatch.setattr("agent.prompt_builder.build_environment_hints", lambda: environment)
    original = _agent(session, _skill())
    original.model, original.provider, original.platform = "model-a", "provider-a", "cli"
    original.pass_session_id = True
    _restore(original)
    stored = original._cached_system_prompt
    assert identity_line_value(stored, "Model") == "model-a"
    assert identity_line_value(stored, "Provider") == "provider-a"
    assert identity_line_value(stored, "Session ID") == "launch-session"
    assert identity_line_value(stored, "Platform") == "cli"
    if environment:
        assert environment in split_runtime_boundary(stored)[2]
    # A real route switch must reject stale model identity even with launch state.
    original.model = "model-b"
    assert not _stored_prompt_matches_runtime(original, stored)
    original.model, original.provider = "model-a", "provider-b"
    assert not _stored_prompt_matches_runtime(original, stored)
    # A surface switch keeps cached bytes and adds the normal tail correction.
    resumed = _agent(session)
    resumed.model, resumed.provider, resumed.platform = "model-a", "provider-a", "tui"
    resumed.pass_session_id = True
    _restore(resumed)
    assert resumed._cached_system_prompt == stored
    assert "tui" in resumed._surface_switch_note
