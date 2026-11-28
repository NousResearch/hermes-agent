"""Shared guard for dashboard-auth provider tests: no *external* IDP HTTP.

The bundled providers (nous account enrichment, self-hosted OIDC discovery) call
``httpx.get`` in the request path. Tests that want a response patch
``<provider module>.httpx.get`` themselves; everything else aimed at a non-loopback
host gets a connect failure, which exercises the providers' fail-open paths
instead of reaching a real Portal over the network. Loopback URLs still pass —
several tests legitimately talk to a local HTTP server they spin up.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
import pytest

_REAL_GET = httpx.get
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


@pytest.fixture(autouse=True)
def _offline_external_idp_get(monkeypatch):
    def _guarded(url, *args, **kwargs):
        host = urlparse(str(url)).hostname or ""
        if host not in _LOOPBACK:
            raise httpx.ConnectError("dashboard-auth tests must mock httpx.get for non-loopback hosts")
        return _REAL_GET(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "get", _guarded)
