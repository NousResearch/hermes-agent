"""The Telegram TLS workaround reaches every pool without weakening verification."""

import ssl

import httpx
import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.config_loader import load_yaml_layer
from plugins.platforms.telegram import adapter as tg


def _supported_groups(context):
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    connection = context.wrap_bio(incoming, outgoing, server_side=False, server_hostname="api.telegram.org")
    with pytest.raises(ssl.SSLWantReadError):
        connection.do_handshake()
    hello = outgoing.read()[9:]  # TLS record header + handshake header
    pos = 34
    pos += 1 + hello[pos]  # session ID
    pos += 2 + int.from_bytes(hello[pos:pos + 2], "big")  # cipher suites
    pos += 1 + hello[pos]  # compression methods
    pos += 2  # extensions length
    while pos < len(hello):
        kind = int.from_bytes(hello[pos:pos + 2], "big")
        size = int.from_bytes(hello[pos + 2:pos + 4], "big")
        data = hello[pos + 4:pos + 4 + size]
        if kind == 10:  # supported_groups
            return [int.from_bytes(data[i:i + 2], "big") for i in range(2, len(data), 2)]
        pos += 4 + size
    raise AssertionError("ClientHello did not advertise supported_groups")


async def _build_requests(tmp_path, monkeypatch, route, enabled):
    # Exercise the real YAML -> PlatformConfig -> adapter -> PTB/httpx chain.
    extra = {} if enabled is None else {"tls_classic_key_exchange": enabled}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"platforms": {"telegram": {"extra": extra}}}), encoding="utf-8"
    )
    data = {}
    load_yaml_layer(tmp_path, data)
    config = GatewayConfig.from_dict(data).platforms[Platform.TELEGRAM]
    adapter = tg.TelegramAdapter(config)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "false" if route == "fallback" else "true")
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: ["149.154.166.110"])
    monkeypatch.setattr(tg, "resolve_proxy_url", lambda *a, **k: "http://127.0.0.1:7890" if route == "proxy" else None)
    from plugins.platforms.telegram import telegram_network
    monkeypatch.setattr(telegram_network, "_resolve_proxy_url", lambda **k: None)
    return await adapter._build_ptb_requests()


async def _contexts(request, route):
    transport = request._client._transport
    if route == "fallback":
        yield transport._primary._pool._ssl_context
        yield (await transport._get_fallback("149.154.166.110"))._pool._ssl_context
        await transport._reset_primary(transport._primary)
        await transport._reset_fallback("149.154.166.110")
        yield transport._primary._pool._ssl_context
        yield (await transport._get_fallback("149.154.166.110"))._pool._ssl_context
    elif route == "proxy":
        for mounted in request._client._mounts.values():
            if mounted is not None:
                yield mounted._pool._ssl_context
    else:
        yield transport._pool._ssl_context


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["direct", "proxy", "fallback"])
async def test_classic_key_exchange_reaches_all_pools(tmp_path, monkeypatch, route):
    # A distinct one-root bundle catches accidental loss of the operator's CA override.
    root = httpx.create_ssl_context().get_ca_certs(binary_form=True)[0]
    bundle = tmp_path / "custom-ca.pem"
    bundle.write_text(ssl.DER_cert_to_PEM_cert(root), encoding="ascii")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    baseline = httpx.create_ssl_context()
    requests = await _build_requests(tmp_path, monkeypatch, route, True)
    try:
        for request in requests:
            contexts = [ctx async for ctx in _contexts(request, route)]
            assert contexts
            for context in contexts:
                assert _supported_groups(context) == [29]  # IANA X25519
                assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
                assert context.get_ca_certs(binary_form=True) == baseline.get_ca_certs(binary_form=True)
                assert context.minimum_version == baseline.minimum_version
                assert context.maximum_version == baseline.maximum_version
        assert _supported_groups(httpx.create_ssl_context()) == _supported_groups(baseline)
    finally:
        for request in requests:
            await request.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, None])
@pytest.mark.parametrize("route", ["direct", "proxy", "fallback"])
async def test_default_key_exchange_is_unchanged(tmp_path, monkeypatch, route, enabled):
    baseline = httpx.create_ssl_context()
    requests = await _build_requests(tmp_path, monkeypatch, route, enabled)
    try:
        for request in requests:
            async for context in _contexts(request, route):
                assert _supported_groups(context) == _supported_groups(baseline)
    finally:
        for request in requests:
            await request.shutdown()
