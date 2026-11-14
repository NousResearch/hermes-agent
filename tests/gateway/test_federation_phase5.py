"""Tests for federation Phase 5 — mDNS discovery."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gateway.federation.federation_discovery import (
    DiscoveredPeer,
    FederationMDNS,
    HERMES_SERVICE_TYPE,
)


class TestDiscoveredPeer:
    def test_create(self):
        peer = DiscoveredPeer(
            device_id="dev-a",
            hostname="my-mac",
            ws_url="ws://192.168.1.10:18765",
        )
        assert peer.device_id == "dev-a"
        assert peer.ws_port == 18765
        assert peer.status == "online"

    def test_ws_port_extraction(self):
        peer = DiscoveredPeer(
            device_id="dev-b",
            hostname="other",
            ws_url="ws://10.0.0.5:19000",
        )
        assert peer.ws_port == 19000

    def test_ws_port_fallback(self):
        peer = DiscoveredPeer(
            device_id="dev-c",
            hostname="bad-url",
            ws_url="not-a-valid-url",
        )
        assert peer.ws_port == 18765  # fallback


class TestFederationMDNS:
    """mDNS-based federation peer discovery."""

    def test_init(self):
        mdns = FederationMDNS(
            device_id="dev-a",
            ws_port=18765,
        )
        assert mdns.device_id == "dev-a"
        assert mdns.ws_port == 18765
        assert mdns.peer_count == 0

    def test_socket_creation(self):
        """Test that mDNS socket can be created (may fail on CI without multicast)."""
        mdns = FederationMDNS(device_id="dev-a")
        try:
            sock = mdns._create_socket()
            assert sock is not None
            sock.close()
        except OSError:
            # Multicast may not be available in all environments
            pytest.skip("mDNS multicast not available")

    def test_get_peers_empty(self):
        mdns = FederationMDNS(device_id="dev-a")
        assert mdns.get_peers() == []

    def test_get_peer_not_found(self):
        mdns = FederationMDNS(device_id="dev-a")
        assert mdns.get_peer("nonexistent") is None

    def test_handle_announcement(self):
        """Test handling incoming mDNS announcement."""
        discovered = []
        mdns = FederationMDNS(
            device_id="dev-a",
            on_discover=lambda p: discovered.append(p),
        )

        payload = {
            "type": "announce",
            "device_id": "dev-b",
            "ws_port": 18766,
            "status": "online",
            "version": "2.0.0",
            "hostname": "other-mac",
        }

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mdns._handle_message(payload, ("192.168.1.10", 5353))
        )

        assert len(discovered) == 1
        peer = discovered[0]
        assert peer.device_id == "dev-b"
        assert peer.hostname == "other-mac"
        assert "192.168.1.10" in peer.ws_url

    def test_handle_self_announcement_ignored(self):
        """Announcements from self should be ignored."""
        discovered = []
        mdns = FederationMDNS(
            device_id="dev-a",
            on_discover=lambda p: discovered.append(p),
        )

        payload = {
            "type": "announce",
            "device_id": "dev-a",  # Same as self
            "ws_port": 18765,
        }

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mdns._handle_message(payload, ("127.0.0.1", 5353))
        )

        assert len(discovered) == 0

    def test_peer_active_filtering(self):
        """Peers not seen recently should be filtered out."""
        mdns = FederationMDNS(device_id="dev-a")

        # Add a stale peer
        import time
        stale_peer = DiscoveredPeer(
            device_id="stale",
            hostname="old",
            ws_url="ws://1.2.3.4:18765",
            last_seen=time.time() - 3600,  # 1 hour ago
        )
        mdns._peers["stale"] = stale_peer

        # Should not appear in active peers
        active = mdns.get_peers()
        assert len(active) == 0

    def test_peer_count(self):
        mdns = FederationMDNS(device_id="dev-a")

        mdns._peers["peer1"] = DiscoveredPeer(
            device_id="peer1", hostname="host1", ws_url="ws://1.1.1.1:18765",
        )
        mdns._peers["peer2"] = DiscoveredPeer(
            device_id="peer2", hostname="host2", ws_url="ws://2.2.2.2:18765",
        )

        assert mdns.peer_count == 2

    def test_get_specific_peer(self):
        mdns = FederationMDNS(device_id="dev-a")

        mdns._peers["peer1"] = DiscoveredPeer(
            device_id="peer1", hostname="host1", ws_url="ws://1.1.1.1:18765",
        )

        peer = mdns.get_peer("peer1")
        assert peer is not None
        assert peer.hostname == "host1"

        assert mdns.get_peer("nonexistent") is None
