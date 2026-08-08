"""Regression: Discord's early pairing-admission gate must route to the
adapter's own multiplex profile, not the global default PairingStore.

``DiscordAdapter._is_pairing_approved_user`` (called from ``_is_allowed_user``,
the on_message admission gate) constructed a bare ``PairingStore()`` — always
the process-global/default-profile store. In a multiplex gateway, a secondary
profile is served by its own ``DiscordAdapter`` instance
(``gateway/run.py::_configure_profile_adapter`` stamps ``_own_profile`` on
it), and its own pairing whitelist lives in a separate, profile-scoped
``PairingStore(profile=name)``. A user approved only on that routed profile
was rejected here — before the message ever reached the correctly-scoped
downstream check in ``authz_mixin._pairing_store_for`` — because this earlier
gate consulted the wrong (default/global) whitelist.
"""

from unittest.mock import MagicMock, patch

from plugins.platforms.discord.adapter import DiscordAdapter


def _make_adapter(own_profile=None):
    """Build a minimal DiscordAdapter without running __init__."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    if own_profile is not None:
        adapter._own_profile = own_profile
    return adapter


def test_default_profile_uses_global_pairing_store():
    """No _own_profile stamped (default profile, or an adapter built without
    going through _configure_profile_adapter) -> PairingStore() with no
    profile kwarg, preserving pre-fix behavior exactly."""
    adapter = _make_adapter()
    assert getattr(adapter, "_own_profile", None) is None

    with patch("gateway.pairing.PairingStore") as mock_store_cls:
        mock_store_cls.return_value.is_approved.return_value = True
        assert adapter._is_pairing_approved_user("42") is True
        mock_store_cls.assert_called_once_with()
        mock_store_cls.return_value.is_approved.assert_called_once_with("discord", "42")


def test_secondary_profile_routes_to_its_own_pairing_store():
    """A secondary-profile adapter (own_profile='coder') must construct
    PairingStore(profile='coder'), not the global default store."""
    adapter = _make_adapter(own_profile="coder")

    with patch("gateway.pairing.PairingStore") as mock_store_cls:
        mock_store_cls.return_value.is_approved.return_value = True
        assert adapter._is_pairing_approved_user("42") is True
        mock_store_cls.assert_called_once_with(profile="coder")


def test_user_approved_only_on_routed_profile_is_not_rejected():
    """The regression scenario: a user approved on the 'coder' profile's
    pairing whitelist but NOT on the global default must be admitted when
    the adapter serving that profile checks."""
    adapter = _make_adapter(own_profile="coder")

    def _fake_store(profile=None):
        store = MagicMock()
        # Only the 'coder' profile's store has this user approved.
        store.is_approved.return_value = profile == "coder"
        return store

    with patch("gateway.pairing.PairingStore", side_effect=_fake_store):
        assert adapter._is_pairing_approved_user("42") is True


def test_is_allowed_user_admits_via_profile_scoped_pairing():
    """End-to-end through _is_allowed_user (the real on_message gate): a
    secondary-profile adapter with no configured allowlists must still admit
    a user approved only on its own profile's pairing store."""
    adapter = _make_adapter(own_profile="coder")

    with patch("gateway.pairing.PairingStore") as mock_store_cls:
        mock_store_cls.return_value.is_approved.return_value = True
        assert adapter._is_allowed_user("42", author=None, is_dm=True) is True
        mock_store_cls.assert_called_once_with(profile="coder")
