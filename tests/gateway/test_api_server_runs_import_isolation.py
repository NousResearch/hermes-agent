"""Import-time invariant: a missing ``RequestKey`` must not take ``web`` down with it."""

import importlib
import sys
import types

import aiohttp.web_request
from aiohttp import web


def test_missing_requestkey_keeps_web_module_bound():
    import gateway.platforms.api_server_runs as api_server_runs

    stub = types.ModuleType("aiohttp.web_request")
    real = sys.modules["aiohttp.web_request"]
    sys.modules["aiohttp.web_request"] = stub
    try:
        importlib.reload(api_server_runs)
        assert api_server_runs.web is web
        assert api_server_runs.RequestKey is None
    finally:
        sys.modules["aiohttp.web_request"] = real
        importlib.reload(api_server_runs)

    assert api_server_runs.RequestKey is aiohttp.web_request.RequestKey
