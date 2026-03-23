from unittest.mock import patch

from gateway.kasia_identity import (
    canonicalize_kasia_address,
    kasia_target_matches,
    lookup_kasia_primary_name,
    normalize_kasia_kns_name,
    resolve_kasia_identity,
    resolve_kasia_kns_name,
)


def test_normalize_kasia_kns_name_accepts_bare_labels():
    assert normalize_kasia_kns_name("Peer") == "peer.kas"


def test_normalize_kasia_kns_name_rejects_bare_address_like_targets():
    assert normalize_kasia_kns_name("qpeeraddress") is None


def test_canonicalize_kasia_address_adds_mainnet_prefix(monkeypatch):
    monkeypatch.setenv("KASIA_NETWORK", "mainnet")
    assert canonicalize_kasia_address("qpeeraddress") == "kaspa:qpeeraddress"


def test_resolve_kasia_kns_name_uses_owner_endpoint(monkeypatch):
    monkeypatch.setenv("KASIA_NETWORK", "mainnet")

    def fake_fetch(url: str):
        assert url.endswith("/peer.kas/owner")
        return {
            "success": True,
            "data": {
                "asset": "peer.kas",
                "owner": "kaspa:qpeeraddress",
            },
        }

    assert resolve_kasia_kns_name("peer.kas", fetch_json=fake_fetch) == "kaspa:qpeeraddress"


def test_lookup_kasia_primary_name_uses_primary_name_endpoint(monkeypatch):
    monkeypatch.setenv("KASIA_NETWORK", "mainnet")

    def fake_fetch(url: str):
        assert url.endswith("/primary-name/kaspa%3Aqpeeraddress")
        return {
            "success": True,
            "data": {
                "domain": {"name": "peer.kas"},
                "ownerAddress": "kaspa:qpeeraddress",
            },
        }

    assert (
        lookup_kasia_primary_name("kaspa:qpeeraddress", fetch_json=fake_fetch)
        == "peer.kas"
    )


def test_resolve_kasia_identity_prefers_display_kns_over_lookup(monkeypatch):
    monkeypatch.setenv("KASIA_NETWORK", "mainnet")
    identity = resolve_kasia_identity(
        "kaspa:qpeeraddress",
        display_name="peer.kas",
        fetch_json=lambda _url: {"success": False, "data": {}},
    )

    assert identity.canonical_address == "kaspa:qpeeraddress"
    assert identity.kns_name == "peer.kas"
    assert identity.display_name == "peer.kas"


def test_kasia_target_matches_uses_display_kns_before_network_lookup():
    with patch(
        "gateway.kasia_identity.resolve_kasia_kns_name",
        side_effect=AssertionError("should not resolve"),
    ):
        assert kasia_target_matches(
            "kaspa:qpeeraddress",
            "peer.kas",
            display_name="peer.kas",
        ) is True
