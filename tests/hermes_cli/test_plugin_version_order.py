"""Plugin update consumers share strict version ordering and unknown results."""
import json
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins_updates import run_checks


def _installed_feed_plugin(plugins):
    plugin = plugins / 'feed-plugin'
    (plugin / '.git').mkdir(parents=True)
    feed_url = 'https://example.invalid/plugin.yml'
    metadata = plugins / '.install-metadata.json'
    metadata.write_text(json.dumps({'feed-plugin': {
        'source': 'https://example.invalid/plugin.git', 'revision': 'a' * 40,
        'update_url': feed_url, 'pinned': False,
    }}), encoding='utf-8')
    return plugin, metadata, feed_url


@pytest.mark.parametrize('current, offered, expected', [
    ('2.0', '1.0', False),
    ('1.0', '1.0.0', False),
    ('1.0rc1', '1.0', True),
    ('1.0', '1.0rc1', False),
    ('1.0', '2.0', True),
    ('1.0', 'garbage!', None),
    ('garbage!', '1.0', None),
])
def test_feed_and_pip_checks_use_the_same_version_order(tmp_path, current, offered, expected):
    plugins = tmp_path / 'plugins'
    plugin, metadata, feed_url = _installed_feed_plugin(plugins)
    manifest = plugin / 'plugin.yaml'
    manifest.write_text(yaml.safe_dump({'name': plugin.name, 'version': current,
                                       'update_url': feed_url}), encoding='utf-8')
    before = {file: file.read_bytes() for file in (manifest, metadata)}
    fetched = []
    results = run_checks(
        plugins,
        fetch=lambda url: fetched.append(url) or yaml.safe_dump({'version': offered}),
        ls_remote=lambda source: pytest.fail('a saved feed must not fall back to Git'),
        pip_entry_points=[SimpleNamespace(name='pip-plugin', dist_name='plugin-dist')],
        pip_installed_version=lambda name: current,
        pip_pypi_latest=lambda name: offered,
    )
    assert fetched == [feed_url]
    assert {result.name for result in results} == {'feed-plugin', 'pip-plugin'}
    for result in results:
        row = result.to_json()
        assert row['current'] == current and row['latest'] == offered
        assert row['update_available'] is expected
        assert ('cannot compare' in row['reason']) is (expected is None)
    assert {file: file.read_bytes() for file in before} == before


def test_cadence_never_applies_an_unparseable_version(tmp_path, monkeypatch):
    from hermes_cli.plugins_cadence import run_scheduled_check
    from pm import receipt

    home = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'tools'))
    plugins = home / 'plugins'
    plugin, metadata, feed_url = _installed_feed_plugin(plugins)
    (plugin / 'plugin.yaml').write_text(
        yaml.safe_dump({'name': plugin.name, 'version': '1.0', 'update_url': feed_url}), encoding='utf-8')
    before = metadata.read_bytes()
    applied = []
    results = run_scheduled_check(
        plugins_dir=plugins,
        run_checks_fn=lambda directory: run_checks(
            directory, include_pip=False,
            fetch=lambda url: 'version: "not-a-version"\n',
            ls_remote=lambda source: pytest.fail('feed version failure is not a Git update'),
        ),
        config_get=lambda section, key: True if key == 'auto_apply' else 24,
        apply_updates_fn=applied.append,
    )
    assert len(results) == 1 and results[0].update_available is None
    assert 'cannot compare' in results[0].reason
    assert applied == []
    assert receipt.latest()['outcome'] == 'ok'
    assert metadata.read_bytes() == before
