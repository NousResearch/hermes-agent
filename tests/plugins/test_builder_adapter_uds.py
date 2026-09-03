import asyncio
import hashlib
import hmac
import json
import os
import time
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from plugins.builder_adapter.auth import HMACAuthenticator, PrincipalKey
from plugins.builder_adapter.canonical import canonical_sha256, signed_material
from plugins.builder_adapter.service import (
    BuilderAdapterService,
    _strict_json,
    bind_unix_socket,
    serve_until,
)
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.runtime import (
    RuntimeSettings,
    _install_shutdown_handlers,
    _load_keys,
)
from plugins.builder_adapter.store import DispatchStore


class Adapter:
    cycle_registry = {}

    def dispatch(self, principal, payload):
        return {"principal": principal, "dispatch_id": payload["dispatch_id"]}


def _headers(secret, payload):
    timestamp = str(int(time.time()))
    nonce = "uds-integration-nonce-12345"
    digest = canonical_sha256(payload)
    signature = hmac.new(
        secret,
        signed_material("POST", "/v1/dispatches", timestamp, nonce, digest),
        hashlib.sha256,
    ).hexdigest()
    return {
        "X-Hermes-Timestamp": timestamp,
        "X-Hermes-Nonce": nonce,
        "X-Hermes-Key-Id": "test-key",
        "X-Hermes-Signature": signature,
    }


def test_health_proves_loaded_cycle_registry():
    adapter = Adapter()
    adapter.cycle_registry = {"CYCLE_ONE": {"revision": 1}}
    service = BuilderAdapterService(adapter, object(), peer_resolver=lambda _: (0, 0))
    app = service.application()
    route = next(
        item
        for item in app.router.routes()
        if item.method == "GET" and item.resource.canonical == "/v1/health"
    )

    response = asyncio.run(route.handler(None))
    payload = json.loads(response.body)

    assert payload["operational"] is True
    assert payload["process_id"] == os.getpid()
    assert payload["registered_cycles"] == 1
    assert payload["cycle_registry_sha256"] == canonical_sha256(
        adapter.cycle_registry
    )


def test_shutdown_signal_requests_orderly_socket_cleanup():
    callbacks = {}

    class Loop:
        def add_signal_handler(self, signum, callback):
            callbacks[signum] = callback

    stop = asyncio.Event()
    installed = _install_shutdown_handlers(Loop(), stop)

    assert installed
    callbacks[installed[-1]]()
    assert stop.is_set()


@pytest.mark.asyncio
async def test_real_uds_authentication_and_clean_shutdown(tmp_path):
    secret = b"isolated-test-secret".ljust(32, b"x")
    auth = HMACAuthenticator(
        [
            PrincipalKey(
                "orchestrator-mcp",
                "test-key",
                secret,
                os.getuid(),
                os.getgid(),
            )
        ],
        DispatchStore(tmp_path / "journal.db"),
    )
    service = BuilderAdapterService(
        Adapter(),
        auth,
        peer_resolver=lambda _: (os.getuid(), os.getgid()),
    )
    with tempfile.TemporaryDirectory(prefix="builder-uds-", dir="/tmp") as short:
        socket_path = Path(short) / "adapter.sock"
        stop = asyncio.Event()
        task = asyncio.create_task(
            serve_until(service.application(), socket_path, stop)
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.01)
        assert socket_path.exists()
        connector = aiohttp.UnixConnector(path=str(socket_path))
        payload = {"dispatch_id": "00000000-0000-0000-0000-000000000001"}
        async with aiohttp.ClientSession(connector=connector) as client:
            response = await client.post(
                "http://localhost/v1/dispatches",
                json=payload,
                headers=_headers(secret, payload),
            )
            assert response.status == 200
            assert (await response.json())["principal"] == "orchestrator-mcp"
        stop.set()
        await asyncio.wait_for(task, timeout=5)
        assert not socket_path.exists()


@pytest.mark.asyncio
async def test_binding_creates_only_a_unix_site(tmp_path):
    app = web.Application()

    async def health(_):
        return web.json_response({"ok": True})

    app.router.add_get("/", health)
    with tempfile.TemporaryDirectory(prefix="builder-uds-", dir="/tmp") as short:
        runner, site = await bind_unix_socket(app, Path(short) / "service.sock")
        try:
            assert isinstance(site, web.UnixSite)
            assert all(isinstance(item, web.UnixSite) for item in runner.sites)
        finally:
            await runner.cleanup()


def test_runtime_config_and_secret_source_are_owner_only(tmp_path, monkeypatch):
    config = tmp_path / "runtime.json"
    config.write_text(
        '{"socket_path":"/tmp/a.sock","state_path":"/tmp/a.db",'
        '"auth_file":"/tmp/auth.json","governance_repo":"/tmp/gov",'
        '"governance_commit":"' + "a" * 40
        + '","repository_allowlist":{},'
        '"validation_profile_id":"hermes-builder-adapter-strict.v1",'
        '"cycle_registry":{}}'
    )
    config.chmod(0o600)
    assert RuntimeSettings.from_file(config).governance_commit == "a" * 40
    config.chmod(0o644)
    with pytest.raises(AdapterError):
        RuntimeSettings.from_file(config)

    auth = tmp_path / "auth.json"
    auth.write_text(
        '{"keys":[{"principal":"p","key_id":"k",'
        '"secret_env":"OPENROUTER_API_KEY","allowed_uid":501}]}'
    )
    auth.chmod(0o600)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-read")
    with pytest.raises(AdapterError, match="unapproved secret source"):
        _load_keys(auth)


def test_socket_parent_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    app = web.Application()
    with pytest.raises(AdapterError, match="symlink"):
        asyncio.run(bind_unix_socket(app, linked / "adapter.sock"))


def test_duplicate_json_keys_are_rejected_before_authentication():
    with pytest.raises(AdapterError, match="duplicate JSON key"):
        _strict_json(b'{"dispatch_id":"one","dispatch_id":"two"}')
