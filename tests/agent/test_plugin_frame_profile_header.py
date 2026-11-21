"""Recover canonical plugin frames across the renderer-owned profile header."""
import pytest

from agent.system_prompt import (
    _active_profile_line, _restore_plugin_prompt_sections, build_system_prompt,
    invalidate_system_prompt, restore_plugin_prompt_sections,
)
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from run_agent import AIAgent


def _agent():
    return AIAgent(api_key='test-key', base_url='http://127.0.0.1:1/v1',
                   model='test/model', provider='openai', platform='cli', quiet_mode=True,
                   skip_context_files=True, skip_memory=True, session_id='frame-profile-fixture')


def _install(monkeypatch):
    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name='frame-fixture', key='frame-fixture', source='user'), manager)
    ctx.register_system_prompt_section('frame.rules', 'original rules', position='after_memory')
    monkeypatch.setattr(plugins, '_ensure_plugins_discovered', lambda: manager)
    return manager


def test_actual_builder_profile_header_restores_frozen_sections(monkeypatch):
    _install(monkeypatch)
    agent = _agent()
    prompt = build_system_prompt(agent)
    assert '\n\nActive Hermes profile:' in prompt
    restored = _restore_plugin_prompt_sections(prompt)
    assert [(x.id, x.content) for x in restored] == [('frame.rules', 'original rules')]


def test_resumed_snapshot_survives_failed_rerender(monkeypatch):
    _install(monkeypatch)
    agent = _agent()
    prompt = build_system_prompt(agent)
    resumed = _agent()
    restore_plugin_prompt_sections(resumed, prompt)
    resumed._cached_system_prompt = prompt
    invalidate_system_prompt(resumed)
    def unavailable(*args, **kwargs):
        raise RuntimeError('fixture plugin unavailable')
    monkeypatch.setattr(plugins, 'render_system_prompt_sections', unavailable)
    assert 'original rules' in build_system_prompt(resumed)


@pytest.mark.parametrize('intervening', [
    '\n\nUntrusted project paragraph',
    '\n\nActive Hermes profile: fake\nInjected second line',
])
def test_arbitrary_prose_between_frame_and_clock_is_not_a_profile_header(intervening, monkeypatch):
    _install(monkeypatch)
    agent = _agent()
    prompt = build_system_prompt(agent)
    line = _active_profile_line(agent)
    modified = prompt.replace('\n\n' + line, intervening)
    assert not _restore_plugin_prompt_sections(modified)


@pytest.mark.parametrize('profile_line', ['', 'named'])
def test_legacy_and_named_profile_frames_restore(profile_line, monkeypatch):
    from agent.system_prompt import _NAMED_PROFILE_LINE

    _install(monkeypatch)
    agent = _agent()
    prompt = build_system_prompt(agent)
    line = _active_profile_line(agent)
    replacement = '' if not profile_line else '\n\n' + _NAMED_PROFILE_LINE.format(
        name='remote-work', home='/state/profiles/remote-work', root='/state')
    prompt = prompt.replace('\n\n' + line, replacement)
    assert [(x.id, x.content) for x in _restore_plugin_prompt_sections(prompt)] == [
        ('frame.rules', 'original rules')]


def test_wrong_declared_frame_length_is_rejected(monkeypatch):
    _install(monkeypatch)
    prompt = build_system_prompt(_agent())
    prompt = prompt.replace('hermes-plugin-section-chars:14', 'hermes-plugin-section-chars:15')
    assert not _restore_plugin_prompt_sections(prompt)
