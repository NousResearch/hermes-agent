"""Regression test: gateway_enroll._post_enroll() must route its credentialed
HTTP call through hermes_cli.urllib_security.open_credentialed_url, not raw
urllib.request.urlopen, so a Bearer token cannot follow a cross-host redirect
to a malicious relay connector.

Sibling of the security(providers)/fix(security) sweep that migrated
providers/base.py, hermes_cli/models.py, hermes_cli/azure_detect.py, and
plugins/model-providers/anthropic/__init__.py to the same shared helper —
this call site read a caller-supplied ``connector_base_url`` but was not
one of them.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import hermes_cli.gateway_enroll as ge


def _fake_http_ok(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return cm


def test_post_enroll_uses_open_credentialed_url_not_raw_urlopen():
    """Patching ``open_credentialed_url`` on pre-fix code (where it isn't
    imported into this module) raises AttributeError, so this test fails
    closed against the vulnerable code and only passes once the call site
    is migrated.
    """
    response = {"secret": "s3cr3t", "deliveryKey": "dk", "tenant": "t1", "gatewayId": "gw1"}
    with patch.object(
        ge, "open_credentialed_url", return_value=_fake_http_ok(response)
    ) as mock_open, patch.object(
        ge.urllib.request, "urlopen"
    ) as mock_urlopen:
        result = ge._post_enroll(
            connector_base_url="https://connector.example.com",
            access_token="tok_abc",
            enrollment_token="enroll_tok",
            gateway_id="gw1",
        )

    assert result == response
    mock_open.assert_called_once()
    mock_urlopen.assert_not_called()
    req = mock_open.call_args.args[0]
    assert req.get_header("Authorization") == "Bearer tok_abc"
    assert req.full_url == "https://connector.example.com/relay/enroll"
