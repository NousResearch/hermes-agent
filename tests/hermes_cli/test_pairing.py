from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import pairing as pairing_mod


def test_cmd_list_shows_kasia_kns_and_address(capsys):
    store = SimpleNamespace(
        list_pending=lambda: [
            {
                "platform": "kasia",
                "user_id": "kaspa:qpeeraddress",
                "canonical_address": "kaspa:qpeeraddress",
                "kns_name": "peer.kas",
                "display_name": "peer.kas",
                "age_minutes": 3,
            }
        ],
        list_approved=lambda: [],
    )

    pairing_mod._cmd_list(store)

    output = capsys.readouterr().out
    assert "Pending Kasia Contacts" in output
    assert "peer.kas" in output
    assert "kaspa:qpeeraddress" in output


def test_cmd_approve_uses_kasia_identity_path(capsys):
    store = SimpleNamespace(
        approve_identity=lambda platform, target: {
            "platform": platform,
            "user_id": "kaspa:qpeeraddress",
            "canonical_address": "kaspa:qpeeraddress",
            "kns_name": "peer.kas",
            "display_name": "peer.kas",
        },
    )

    with patch.object(
        pairing_mod,
        "_complete_live_kasia_approval",
        return_value={"status": "responded"},
    ) as complete_live:
        pairing_mod._cmd_approve(store, "kasia", "peer.kas")

    output = capsys.readouterr().out
    assert "Approved Kasia contact peer.kas" in output
    assert "kaspa:qpeeraddress" in output
    assert "Live Kasia transport is ready now." in output
    complete_live.assert_called_once()
