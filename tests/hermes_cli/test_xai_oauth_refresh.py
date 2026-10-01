from __future__ import annotations
import auth.constants as _auth_auth_constants
import auth.providers.xai as _auth_auth_providers_xai


import base64
import json
import time

from hermes_cli import auth


def _jwt_with_exp(exp: int) -> str:
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode())
        .decode()
        .rstrip("=")
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
        .decode()
        .rstrip("=")
    )
    return f"{header}.{payload}.sig"




def test_xai_oauth_token_expiring_uses_one_hour_skew() -> None:
    token = _jwt_with_exp(int(time.time()) + 30 * 60)

    assert _auth_auth_providers_xai._xai_access_token_is_expiring(
        token,
        _auth_auth_constants.XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    )


def test_xai_proactive_refresh_skew_short_lived_token() -> None:
    token = _jwt_with_exp(int(time.time()) + 15 * 60)
    skew = _auth_auth_providers_xai._xai_proactive_refresh_skew_seconds(token)

    assert skew == 120
    assert not _auth_auth_providers_xai._xai_access_token_is_expiring(token, skew)


