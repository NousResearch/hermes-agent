"""Tests for the MiniMax OAuth live model catalog (hermes_cli/models_minimax.py).

The ``minimax-oauth`` picker row used to be a hand-written snapshot, so a model MiniMax
ships (e.g. MiniMax-M3.1-Flash) was missing from every picker surface until Hermes
shipped a release. ``_minimax_oauth_catalog`` probes the account's real catalog at the
Anthropic-compatible ``/v1/models`` endpoint and merges it with the curated row.

Covers:
- live ids lead, curated-only extras are appended
- no credentials / no token -> None (the curated row is served)
- a failed probe -> None, and it never raises
- the picker never refreshes or rewrites the credential (read-only, #68004)
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.models_minimax import minimax_oauth_catalog as _minimax_oauth_catalog

MINIMAX_BASE = "https://api.minimax.io/anthropic"


def _state(**overrides):
    state = {
        "access_token": "sk-cp-test-token",
        "inference_base_url": MINIMAX_BASE,
    }
    state.update(overrides)
    return state


def _live_payload(*ids):
    return {"data": [{"id": i, "type": "model"} for i in ids]}


def _with_curated(curated):
    """Patch the curated row the fetcher merges against, so the test does not depend on
    the shipped catalog contents."""
    return patch.dict(
        "hermes_cli.models._PROVIDER_MODELS",
        {"minimax-oauth": list(curated)},
    )


def test_live_ids_lead_and_curated_extras_appended():
    live = ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    curated = ["MiniMax-M3", "MiniMax-M2.7"]

    with patch("hermes_cli.auth.get_provider_auth_state", return_value=_state()), \
         patch("hermes_cli.models._get_json", return_value=_live_payload(*live)) as get_json, \
         _with_curated(curated):
        rows = _minimax_oauth_catalog(False)

    assert rows is not None
    assert rows[0] == "MiniMax-M3.1-Flash"
    # Live leads; a curated model the endpoint omits still appears.
    assert "MiniMax-M2.7" in rows
    assert rows.index("MiniMax-M3.1-Flash") < rows.index("MiniMax-M2.7")
    # The probe hits the Anthropic-compatible catalog endpoint, not /models.
    args, _ = get_json.call_args
    assert args[0] == f"{MINIMAX_BASE}/v1/models"


def test_no_token_returns_none():
    with patch("hermes_cli.auth.get_provider_auth_state", return_value={}), \
         patch("hermes_cli.models._get_json") as get_json:
        assert _minimax_oauth_catalog(False) is None
    get_json.assert_not_called()


def test_no_auth_state_returns_none():
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=None), \
         patch("hermes_cli.models._get_json") as get_json:
        assert _minimax_oauth_catalog(False) is None
    get_json.assert_not_called()


def test_failed_probe_returns_none_without_raising():
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=_state()), \
         patch("hermes_cli.models._get_json", side_effect=OSError("boom")):
        assert _minimax_oauth_catalog(False) is None


def test_empty_live_payload_returns_none():
    with patch("hermes_cli.auth.get_provider_auth_state", return_value=_state()), \
         patch("hermes_cli.models._get_json", return_value={"data": []}):
        assert _minimax_oauth_catalog(False) is None


def test_probe_is_read_only_and_never_refreshes_the_token():
    """The picker must not import, refresh or persist a credential (#68004).

    ``resolve_minimax_oauth_runtime_credentials`` refreshes the token and rewrites
    auth.json, so the catalog probe must read the stored token directly instead. It is
    patched where it is actually consulted: ``hermes_cli.auth`` re-exports it at import
    and ``OAuthProviderFlow.resolve`` looks it up in that module's namespace.
    """
    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("catalog probe must not refresh the credential")

    with patch("hermes_cli.auth.resolve_minimax_oauth_runtime_credentials", _boom), \
         patch("hermes_cli.auth_minimax.resolve_minimax_oauth_runtime_credentials", _boom), \
         patch("hermes_cli.auth.get_provider_auth_state", return_value=_state()), \
         patch("hermes_cli.models._get_json", return_value=_live_payload("MiniMax-M3.1-Flash")), \
         _with_curated(["MiniMax-M3"]):
        rows = _minimax_oauth_catalog(False)

    assert rows and "MiniMax-M3.1-Flash" in rows
