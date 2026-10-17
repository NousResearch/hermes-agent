"""Explicit catalog updates must consult the publisher, not just the fresh TTL cache."""
import json
from unittest.mock import Mock

import httpx
import pytest

from hermes_cli import plugin_catalog as catalog
from hermes_cli import plugins_cmd_catalog as updates
from hermes_cli.plugins_cmd import PluginOperationError


@pytest.fixture
def cached_catalog(tmp_path, monkeypatch):
    cache = tmp_path / 'cache' / 'plugin-catalog.json'
    cache.parent.mkdir()
    entry = {'name': 'refresh-test', 'repo': 'https://github.com/example/refresh-test',
             'sha': 'a' * 40, 'description': 'Test.', 'maintainer': 'example'}
    document = {'entries': [entry], 'removed': []}
    cache.write_text(json.dumps(document), encoding='utf-8')
    monkeypatch.setattr(catalog, '_live_cache_path', lambda: cache)
    monkeypatch.setattr(catalog, 'load_catalog', lambda: [])
    monkeypatch.setattr(catalog, 'in_tree_catalog_time', lambda: None)
    monkeypatch.setattr(catalog, '_live_fetch_failed_until', 0)
    sidecar = {'catalog_name': entry['name'], 'sha': entry['sha'], 'pin': entry['sha']}
    return cache, document, sidecar


def test_update_detects_published_removal_despite_fresh_cache(cached_catalog, tmp_path, monkeypatch):
    _, _, sidecar = cached_catalog
    response = httpx.Response(200, json={'entries': [], 'removed': []},
                              request=httpx.Request('GET', catalog.LIVE_CATALOG_URL))
    fetch = Mock(return_value=response)
    monkeypatch.setattr(httpx, 'get', fetch)
    with pytest.raises(PluginOperationError, match='no longer in the catalog'):
        updates.repin_catalog_plugin(tmp_path / 'refresh-test', sidecar)
    assert fetch.call_count == 1


def test_update_refreshes_even_when_pin_is_unchanged(cached_catalog, tmp_path, monkeypatch):
    _, doc, sidecar = cached_catalog
    response = httpx.Response(200, json=doc, request=httpx.Request('GET', catalog.LIVE_CATALOG_URL))
    fetch = Mock(return_value=response)
    monkeypatch.setattr(httpx, 'get', fetch)
    result = updates.repin_catalog_plugin(tmp_path / 'refresh-test', sidecar)
    assert result.changed is False
    assert fetch.call_count == 1


def test_update_keeps_offline_fallback(cached_catalog, tmp_path, monkeypatch):
    _, _, sidecar = cached_catalog
    fetch = Mock(side_effect=httpx.ConnectError('offline'))
    monkeypatch.setattr(httpx, 'get', fetch)
    result = updates.repin_catalog_plugin(tmp_path / 'refresh-test', sidecar)
    assert result.changed is False
    assert fetch.call_count == 1


def test_new_pin_reaches_update_preparation(cached_catalog, tmp_path, monkeypatch):
    cache, doc, sidecar = cached_catalog
    doc['entries'][0]['sha'] = 'b' * 40
    response = httpx.Response(200, json=doc, request=httpx.Request('GET', catalog.LIVE_CATALOG_URL))
    monkeypatch.setattr(httpx, 'get', Mock(return_value=response))

    def stop_before_install(target):
        assert catalog.get_live_catalog_entry('refresh-test').sha == 'b' * 40
        assert json.loads(cache.read_text())['entries'][0]['sha'] == 'b' * 40
        raise PluginOperationError('reached update preparation')

    monkeypatch.setattr(updates, '_local_changes', stop_before_install)
    with pytest.raises(PluginOperationError, match='reached update preparation'):
        updates.repin_catalog_plugin(tmp_path / 'refresh-test', sidecar)


def test_ordinary_catalog_reads_keep_ttl_cache(cached_catalog, monkeypatch):
    _, _, sidecar = cached_catalog
    fetch = Mock(side_effect=AssertionError('ordinary reads should stay cached'))
    monkeypatch.setattr(httpx, 'get', fetch)
    assert catalog.get_live_catalog_entry('refresh-test').sha == sidecar['sha']
    fetch.assert_not_called()
