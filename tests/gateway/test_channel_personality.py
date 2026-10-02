"""Named channel styles use the existing scoped prompt path and personality registry."""
import pytest
import yaml

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope


def make_runner(config):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig.from_dict(config['platforms']['telegram']),
    })
    return runner


def write_config(home, overrides, personalities=None):
    home.mkdir(parents=True, exist_ok=True)
    config = {
        'agent': {'system_prompt': 'Global', 'personalities': personalities or {'focused': 'Be focused.'}},
        'platforms': {'telegram': {'channel_overrides': overrides}},
    }
    (home / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf8')
    return config


@pytest.mark.parametrize(('override', 'expected'), [
    ({'personality': 'focused', 'system_prompt': 'Return JSON.'}, 'Be focused.\n\nReturn JSON.'),
    ({'personality': 'focused'}, 'Be focused.'),
    ({'system_prompt': '  Raw only.  '}, 'Raw only.'),
    ({'personality': 'neutral'}, ''),
    ({'personality': 'focused', 'system_prompt': '   '}, 'Be focused.'),
    ({'model': 'some-model'}, 'Global'),
])
def test_named_channel_personality_reaches_prompt(tmp_path, monkeypatch, override, expected):
    config = write_config(tmp_path, {'topic': override})
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: tmp_path)
    runner = make_runner(config)
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'chat', thread_id='topic') == expected
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'other') == 'Global'


def test_structured_personality_and_parent_lookup(tmp_path, monkeypatch):
    config = write_config(tmp_path, {'parent': {'personality': 'focused', 'system_prompt': 'Return JSON.'}},
                          {'focused': {'system_prompt': 'Focus.', 'tone': 'direct', 'style': 'terse'}})
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: tmp_path)
    runner = make_runner(config)
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'child', parent_id='parent') == (
        'Focus.\nTone: direct\nStyle: terse\n\nReturn JSON.')


@pytest.mark.parametrize(('prompt', 'expected'), [(None, 'Global'), ('Raw.', 'Raw.')])
def test_unknown_personality_warns_and_preserves_fallback(tmp_path, monkeypatch, caplog, prompt, expected):
    config = write_config(tmp_path, {'topic': {'personality': 'missing', 'system_prompt': prompt}})
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: tmp_path)
    runner = make_runner(config)
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'topic') == expected
    assert 'Unknown channel_overrides personality' in caplog.text
    assert 'missing' in caplog.text


def test_empty_named_definition_is_explicit_empty_overlay(tmp_path, monkeypatch):
    config = write_config(tmp_path, {'topic': {'personality': 'focused'}}, {'focused': ''})
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: tmp_path)
    assert make_runner(config)._get_system_prompt_for_channel(Platform.TELEGRAM, 'topic') == ''


def test_personality_roundtrips_with_platform_config():
    raw = {'channel_overrides': {'topic': {'personality': 'focused', 'system_prompt': 'Task.'}}}
    parsed = PlatformConfig.from_dict(raw)
    restored = PlatformConfig.from_dict(parsed.to_dict())
    assert restored.channel_overrides['topic'] == ChannelOverride(personality='focused', system_prompt='Task.')


def test_same_name_uses_active_profile_without_mutating_runner(tmp_path, monkeypatch):
    default, beta = tmp_path / 'default', tmp_path / 'beta'
    config = write_config(default, {'topic': {'personality': 'focused'}}, {'focused': 'Default style.'})
    write_config(beta, {}, {'focused': 'Beta style.'})
    monkeypatch.setattr('gateway.run._hermes_home', default)
    monkeypatch.setenv('HERMES_HOME', str(default))
    runner = make_runner(config)
    runner._ephemeral_system_prompt = 'Untouched'
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'topic') == 'Default style.'
    with _profile_runtime_scope(beta):
        assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'topic') == 'Beta style.'
    assert runner._get_system_prompt_for_channel(Platform.TELEGRAM, 'topic') == 'Default style.'
    assert runner._ephemeral_system_prompt == 'Untouched'


def test_turn_composition_and_cache_identity_preserve_named_style(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource

    config = write_config(tmp_path, {
        'one': {'personality': 'focused', 'system_prompt': 'Task.'},
        'two': {'system_prompt': 'Task.'},
    })
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: tmp_path)
    runner = make_runner(config)
    ctx = SimpleNamespace(context_prompt='Platform context.', channel_prompt='Channel hint.',
                          source=SessionSource(platform=Platform.TELEGRAM, chat_id='one'))
    turn = TurnRunner(runner, ctx)
    first = turn._combined_ephemeral_prompt()
    assert first == 'Platform context.\n\nChannel hint.\n\nBe focused.\n\nTask.'
    signature = runner._agent_config_signature('model', {}, [], first)
    assert runner._agent_config_signature('model', {}, [], turn._combined_ephemeral_prompt()) == signature
    ctx.source.chat_id = 'two'
    second = turn._combined_ephemeral_prompt()
    assert second == 'Platform context.\n\nChannel hint.\n\nTask.'
    assert runner._agent_config_signature('model', {}, [], second) != signature
    ctx.source.chat_id = 'one'
    assert turn._combined_ephemeral_prompt() == first
