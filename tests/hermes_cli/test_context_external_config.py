"""External context options extend canonical engine defaults (#48809)."""

from agent.prompt_builder import build_context_files_prompt
from hermes_cli.config import atomic_config_replace, load_config
from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_context_defaults_extend_engine_and_memory_trim(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    context = load_config()['context']
    assert context['engine'] == 'compressor'
    assert context['memory_trim'] == DEFAULT_CONFIG['context']['memory_trim']
    assert context['external_files'] == []
    assert context['ignore_hermes_md'] is False


def test_real_config_consumed_with_engine_override_and_owning_home(tmp_path, monkeypatch):
    launch = tmp_path / 'launch'
    launch.mkdir()
    owner = tmp_path / 'owner'
    owner.mkdir()
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(launch))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'runtime'))
    (project / 'HERMES.md').write_text('Compatibility contract.')
    (project / 'AGENTS.md').write_text('Automatic contract.')
    external = owner / 'rules.md'
    external.write_text('Owner external contract.')
    atomic_config_replace(owner / 'config.yaml', {'context': {
        'engine': 'custom-engine', 'memory_trim': {'enabled': False},
        'external_files': [str(external)], 'ignore_hermes_md': True,
    }})
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(owner)
    try:
        context = load_config()['context']
    finally:
        reset_hermes_home_override(token)
    assert context['engine'] == 'custom-engine'
    assert context['memory_trim']['enabled'] is False
    assert context['memory_trim']['cooldown_seconds'] == DEFAULT_CONFIG['context']['memory_trim']['cooldown_seconds']
    assert context['external_files'] == [str(external)]
    assert context['ignore_hermes_md'] is True

    result = build_context_files_prompt(cwd=str(project), skip_soul=True, home_override=owner)
    assert 'Owner external contract.' in result
    assert 'Automatic contract.' in result
    assert 'Compatibility contract.' not in result
    assert load_config()['context']['external_files'] == []
