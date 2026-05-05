"""Tests for the Inkbox tunnel client (gateway/platforms/inkbox_tunnel.py)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.inkbox_tunnel import (
    InkboxTunnel,
    TunnelControlPlaneError,
    _decode_envelopes,
    _encode_envelope,
    _slug_for_identity,
    _tunnel_zone_for,
    derive_tunnel_name,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestNameSlug:
    def test_simple_lowercase_passes_through(self):
        assert _slug_for_identity("inkbox-on-call-agent") == "inkbox-on-call-agent"

    def test_uppercase_normalized(self):
        assert _slug_for_identity("MyAgent") == "myagent"

    def test_punctuation_collapsed_to_dashes(self):
        assert _slug_for_identity("foo.bar_baz!qux") == "foo-bar-baz-qux"

    def test_leading_trailing_dashes_stripped(self):
        assert _slug_for_identity("--abc--") == "abc"

    def test_too_short_padded(self):
        s = _slug_for_identity("a")
        assert len(s) >= 3
        assert s.startswith("a")

    def test_too_long_capped_at_63(self):
        s = _slug_for_identity("a" * 100)
        assert len(s) == 63

    def test_empty_falls_back(self):
        # Empty handle -> sentinel name; must satisfy server validation
        # (3..63 lowercase letters/digits/hyphens, alnum start + end).
        s = _slug_for_identity("")
        assert 3 <= len(s) <= 63
        assert s[0].isalnum() and s[-1].isalnum()
        assert "--" not in s


class TestZoneDerivation:
    def test_production_inkbox_ai_maps_to_root_zone(self):
        assert _tunnel_zone_for("https://inkbox.ai") == "inkboxwire.com"

    def test_beta_subdomain_maps_to_beta_zone(self):
        assert _tunnel_zone_for("https://beta.inkbox.ai") == "beta.inkboxwire.com"

    def test_dev_subdomain_maps_to_dev_zone(self):
        assert _tunnel_zone_for("https://development.inkbox.ai") == "development.inkboxwire.com"

    def test_explicit_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("INKBOX_TUNNEL_ZONE", "custom.example.com")
        assert _tunnel_zone_for("https://inkbox.ai") == "custom.example.com"

    def test_unknown_host_falls_back_to_dev(self):
        assert _tunnel_zone_for("https://localhost:9000") == "development.inkboxwire.com"


class TestDeriveName:
    def test_override_wins(self):
        assert derive_tunnel_name(identity_handle="X", override="my-name") == "my-name"

    def test_override_lowercased(self):
        assert derive_tunnel_name(identity_handle="X", override="My-Name") == "my-name"

    def test_override_blank_falls_through(self):
        assert (
            derive_tunnel_name(identity_handle="my-agent", override="")
            == "my-agent"
        )


# ---------------------------------------------------------------------------
# Envelope codec
# ---------------------------------------------------------------------------

class TestEnvelopeCodec:
    def test_roundtrip_text(self):
        env = {"type": "text", "data": "hello world"}
        wire = _encode_envelope(env)
        # 4-byte big-endian length prefix matches the JSON byte length.
        import struct
        (n,) = struct.unpack(">I", wire[:4])
        assert n == len(wire) - 4
        buf = bytearray(wire)
        out = _decode_envelopes(buf)
        assert out == [env]
        assert len(buf) == 0

    def test_roundtrip_partial_then_complete(self):
        a = _encode_envelope({"type": "text", "data": "one"})
        b = _encode_envelope({"type": "text", "data": "two"})
        wire = a + b
        buf = bytearray(wire[: len(a) + 2])  # only part of the second
        first = _decode_envelopes(buf)
        assert first == [{"type": "text", "data": "one"}]
        # Now feed the rest.
        buf.extend(wire[len(a) + 2 :])
        second = _decode_envelopes(buf)
        assert second == [{"type": "text", "data": "two"}]
        assert len(buf) == 0

    def test_no_complete_frame_returns_empty(self):
        buf = bytearray(b"\x00\x00\x00\x10short")  # claims 16 bytes, only 5
        out = _decode_envelopes(buf)
        assert out == []
        # Nothing consumed.
        assert len(buf) == 9


# ---------------------------------------------------------------------------
# ensure_tunnel — REST orchestration
# ---------------------------------------------------------------------------

def _make_tunnel(tmp_path) -> InkboxTunnel:
    state_path = str(tmp_path / "state.json")
    return InkboxTunnel(
        api_key="ApiKey_test",
        base_url="https://inkbox.ai",
        listen_host="127.0.0.1",
        listen_port=8765,
        tunnel_name="hermes-test",
        identity_handle="hermes-test",
        state_path=state_path,
    )


class _FakeRest:
    """Stand-in for _RestClient with scriptable return values."""

    def __init__(
        self,
        *,
        get_results: Dict[str, Dict[str, Any] | None] | None = None,
        list_results: List[Dict[str, Any]] | None = None,
        create_result: Tuple[Dict[str, Any], str] | None = None,
        rotate_result: str = "rotated_secret_xyz",
    ):
        self._get_results = get_results or {}
        self._list_results = list_results or []
        self._create_result = create_result
        self._rotate_result = rotate_result
        self.calls: List[Tuple[str, Any]] = []

    async def aclose(self) -> None:
        self.calls.append(("aclose", None))

    async def get_tunnel(self, tunnel_id: str):
        self.calls.append(("get", tunnel_id))
        return self._get_results.get(tunnel_id)

    async def list_tunnels(self):
        self.calls.append(("list", None))
        return list(self._list_results)

    async def create_tunnel(self, *, tunnel_name, description=""):
        self.calls.append(("create", tunnel_name))
        if self._create_result is None:
            raise TunnelControlPlaneError("no create configured")
        return self._create_result

    async def rotate_secret(self, tunnel_id):
        self.calls.append(("rotate", tunnel_id))
        return self._rotate_result


@pytest.mark.asyncio
async def test_ensure_tunnel_creates_when_no_state(tmp_path, monkeypatch):
    t = _make_tunnel(tmp_path)
    fake = _FakeRest(
        list_results=[],
        create_result=(
            {
                "id": "tun-1",
                "tunnel_name": "hermes-test",
                "status": "active",
            },
            "secret_abc",
        ),
    )
    monkeypatch.setattr(
        "gateway.platforms.inkbox_tunnel._RestClient",
        lambda **kw: fake,
    )

    record = await t.ensure_tunnel()
    assert record["id"] == "tun-1"
    assert t.tunnel_id == "tun-1"
    assert t.tunnel_name == "hermes-test"
    assert ("create", "hermes-test") in fake.calls

    saved = json.loads(open(t._state_path).read())
    assert saved["tunnel_id"] == "tun-1"
    assert saved["connect_secret"] == "secret_abc"


@pytest.mark.asyncio
async def test_ensure_tunnel_reuses_saved_state_when_active(tmp_path, monkeypatch):
    t = _make_tunnel(tmp_path)
    # Pre-seed state file.
    with open(t._state_path, "w") as fp:
        json.dump({
            "tunnel_id": "tun-saved",
            "tunnel_name": "hermes-test",
            "connect_secret": "secret_saved",
        }, fp)

    fake = _FakeRest(
        get_results={
            "tun-saved": {
                "id": "tun-saved",
                "tunnel_name": "hermes-test",
                "status": "active",
            },
        },
    )
    monkeypatch.setattr(
        "gateway.platforms.inkbox_tunnel._RestClient",
        lambda **kw: fake,
    )

    record = await t.ensure_tunnel()
    assert record["id"] == "tun-saved"
    # Should NOT have created — only verified via GET.
    assert ("get", "tun-saved") in fake.calls
    assert all(c[0] != "create" for c in fake.calls)


@pytest.mark.asyncio
async def test_ensure_tunnel_recreates_when_saved_is_deleted(tmp_path, monkeypatch):
    t = _make_tunnel(tmp_path)
    with open(t._state_path, "w") as fp:
        json.dump({
            "tunnel_id": "tun-stale",
            "tunnel_name": "hermes-test",
            "connect_secret": "stale_secret",
        }, fp)

    fake = _FakeRest(
        get_results={"tun-stale": None},  # GET 404 → None
        list_results=[],
        create_result=(
            {
                "id": "tun-fresh",
                "tunnel_name": "hermes-test",
                "status": "active",
            },
            "fresh_secret",
        ),
    )
    monkeypatch.setattr(
        "gateway.platforms.inkbox_tunnel._RestClient",
        lambda **kw: fake,
    )

    record = await t.ensure_tunnel()
    assert record["id"] == "tun-fresh"
    saved = json.loads(open(t._state_path).read())
    assert saved["connect_secret"] == "fresh_secret"


@pytest.mark.asyncio
async def test_ensure_tunnel_rotates_secret_when_org_owns_named_tunnel(tmp_path, monkeypatch):
    """
    First-run on a fresh box but the org already owns this tunnel name —
    the secret was shown once on creation and is unrecoverable.  The only
    safe path is rotate-secret + reuse the existing tunnel record.
    """
    t = _make_tunnel(tmp_path)
    fake = _FakeRest(
        list_results=[
            {
                "id": "tun-existing",
                "tunnel_name": "hermes-test",
                "status": "active",
            },
        ],
        rotate_result="new_rotated_secret",
    )
    monkeypatch.setattr(
        "gateway.platforms.inkbox_tunnel._RestClient",
        lambda **kw: fake,
    )

    record = await t.ensure_tunnel()
    assert record["id"] == "tun-existing"
    assert ("rotate", "tun-existing") in fake.calls
    assert all(c[0] != "create" for c in fake.calls)
    saved = json.loads(open(t._state_path).read())
    assert saved["connect_secret"] == "new_rotated_secret"


# ---------------------------------------------------------------------------
# Public URL
# ---------------------------------------------------------------------------

class TestPublicURL:
    def test_url_built_from_tunnel_name_and_zone(self, tmp_path):
        t = InkboxTunnel(
            api_key="ApiKey",
            base_url="https://beta.inkbox.ai",
            listen_host="127.0.0.1",
            listen_port=8765,
            tunnel_name="my-tunnel",
            state_path=str(tmp_path / "s.json"),
        )
        assert t.public_host == "my-tunnel.beta.inkboxwire.com"
        assert t.public_url == "https://my-tunnel.beta.inkboxwire.com"


# ---------------------------------------------------------------------------
# is_alive() / connected_seconds (used by the adapter watchdog)
# ---------------------------------------------------------------------------

class TestIsAlive:
    def test_fresh_tunnel_is_not_alive(self, tmp_path):
        t = _make_tunnel(tmp_path)
        # No supervisor task spawned yet → not alive.
        assert t.is_alive() is False
        assert t.connected_seconds == 0.0

    @pytest.mark.asyncio
    async def test_alive_when_supervisor_running(self, tmp_path):
        t = _make_tunnel(tmp_path)

        async def _forever() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return

        t._supervisor_task = asyncio.create_task(_forever())
        try:
            assert t.is_alive() is True
        finally:
            t._supervisor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t._supervisor_task

    @pytest.mark.asyncio
    async def test_not_alive_when_supervisor_done(self, tmp_path):
        t = _make_tunnel(tmp_path)

        async def _exit() -> None:
            return

        t._supervisor_task = asyncio.create_task(_exit())
        await t._supervisor_task  # let it finish
        assert t.is_alive() is False

    @pytest.mark.asyncio
    async def test_not_alive_after_stop(self, tmp_path):
        t = _make_tunnel(tmp_path)

        async def _forever() -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return

        t._supervisor_task = asyncio.create_task(_forever())
        try:
            assert t.is_alive() is True
            t._stop_evt.set()
            assert t.is_alive() is False
        finally:
            t._supervisor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t._supervisor_task

    def test_connected_seconds_increases_after_stamp(self, tmp_path):
        import time as _time
        t = _make_tunnel(tmp_path)
        t._connected_at = _time.time() - 42.0
        assert 41.0 <= t.connected_seconds <= 43.0

    def test_connected_seconds_zero_when_unstamped(self, tmp_path):
        t = _make_tunnel(tmp_path)
        t._connected_at = None
        assert t.connected_seconds == 0.0
