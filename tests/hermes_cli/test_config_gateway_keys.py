"""Registration contracts for gateway-scoped config keys the CLI can write."""

from hermes_cli.config import _validate_config_key


def test_gateway_session_store_max_age_days_is_a_registered_key():
    """``hermes config set gateway.session_store_max_age_days`` must be a known key.

    The loader bridge that honours it (``gateway/config_loader.py``) is contracted in
    ``tests/gateway/test_config.py``; this pins the CLI-side registration in
    ``DEFAULT_CONFIG["gateway"]`` so the key never regresses to "writable but unregistered"
    (unknown-key notices, no migration).
    """
    assert _validate_config_key("gateway.session_store_max_age_days") == (True, None)
