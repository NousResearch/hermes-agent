"""Catalog discovery cannot block the ASGI loop or exceed its caller deadline."""
import asyncio
import threading
import time

import httpx
import pytest

pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']


def test_slow_discovery_leaves_concurrent_request_runnable(client, monkeypatch):
    from hermes_cli.skill_catalog import bootstrap
    from hermes_cli.web_server import app
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    workers = []

    def discovery(profile):
        workers.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return False

    monkeypatch.setattr(bootstrap, 'discovery_required', discovery)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test', headers=client.headers) as http:
            pending = asyncio.create_task(http.get('/api/skills/hub/official'))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                # This handler is independent of catalog discovery.
                response = await http.get('/api/skills/content', params={'name': 'absent'})
                assert response.status_code == 404
                assert workers == [workers[0]]
                assert workers[0] != loop_thread
            finally:
                release.set()
                await pending
    asyncio.run(run())


def test_no_consent_skips_manifest_discovery(client, monkeypatch):
    from hermes_cli.skill_catalog import bootstrap
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    atomic_config_write(get_hermes_home() / 'config.yaml', {'plugins': {'isolation': 'host', 'enabled': []}})
    calls = []
    monkeypatch.setattr(bootstrap, 'collect_directory_manifests', lambda: calls.append('manifests') or [])
    assert bootstrap.discovery_required(None) is False
    assert calls == []


def test_discovery_obeys_request_deadline(client, monkeypatch):
    from hermes_cli.skill_catalog import bootstrap, http
    from hermes_cli.skill_catalog.validation import CatalogError
    entered, release = threading.Event(), threading.Event()

    def discovery(profile):
        entered.set()
        assert release.wait(5)
        return False

    monkeypatch.setattr(bootstrap, 'discovery_required', discovery)

    async def run():
        try:
            with pytest.raises(CatalogError) as caught:
                await http._lookup_service(None, time.monotonic() + 0.05)
            assert caught.value.code == 'timeout'
        finally:
            release.set()
    asyncio.run(run())
