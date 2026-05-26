import pytest

from gateway.calls.browser_room import BrowserRoomConfig, BrowserRoomProvider, is_tailnet_url
from gateway.calls.models import CallError


def test_tailnet_url_accepts_ts_net_hostname():
    assert is_tailnet_url("https://bryans-mac-mini.tail670355.ts.net/call")


def test_tailnet_url_accepts_tailscale_cgnat_ip():
    assert is_tailnet_url("https://100.101.102.103:9443/call")


def test_tailnet_url_rejects_public_hostname():
    assert not is_tailnet_url("https://example.com/call")


def test_browser_room_provider_rejects_public_base_url_by_default():
    provider = BrowserRoomProvider(
        BrowserRoomConfig(base_url="https://example.com/call", public_exposure_enabled=False)
    )

    with pytest.raises(CallError) as exc:
        provider.create_room_url("call_123", "token")

    assert exc.value.code == "call_public_exposure_disabled"


def test_browser_room_provider_builds_tailnet_url():
    provider = BrowserRoomProvider(
        BrowserRoomConfig(base_url="https://bryans-mac-mini.tail670355.ts.net/call")
    )

    url = provider.create_room_url("call_123", "token-abc")

    assert url == "https://bryans-mac-mini.tail670355.ts.net/call/call_123?token=token-abc"
