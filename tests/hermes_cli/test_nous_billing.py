"""Regression test: nous_billing._request() must route its credentialed HTTP
call through hermes_cli.urllib_security.open_credentialed_url, not raw
urllib.request.urlopen, so a Bearer token cannot follow a cross-host redirect.

Sibling of the security(providers)/fix(security) sweep that migrated
providers/base.py, hermes_cli/models.py, hermes_cli/azure_detect.py, and
plugins/model-providers/anthropic/__init__.py to the same shared helper —
this call site read the exact same "portal_base_url" style configurable
base URL but was not one of them.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import hermes_cli.nous_billing as nb


def _fake_http_ok(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return cm


def test_request_uses_open_credentialed_url_not_raw_urlopen():
    """The billing request must go through the safe wrapper.

    Patching ``open_credentialed_url`` on pre-fix code (where it isn't
    imported into this module) raises AttributeError, so this test fails
    closed against the vulnerable code and only passes once the call site
    is migrated.
    """
    with patch.object(
        nb, "_resolve_token_and_base", return_value=("tok_abc", "https://portal.nousresearch.com")
    ), patch.object(
        nb, "open_credentialed_url", return_value=_fake_http_ok({"ok": True})
    ) as mock_open, patch.object(
        nb.urllib.request, "urlopen"
    ) as mock_urlopen:
        result = nb._request("GET", "/api/billing/status")

    assert result == {"ok": True}
    mock_open.assert_called_once()
    mock_urlopen.assert_not_called()
    req = mock_open.call_args.args[0]
    assert req.get_header("Authorization") == "Bearer tok_abc"
