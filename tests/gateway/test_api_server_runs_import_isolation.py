"""Import-time invariant: a missing ``RequestKey`` must not take ``web`` down with it."""

import importlib
import sys
import types

import aiohttp.web_request
from aiohttp import web


def _with_missing_requestkey(api_server_runs):
    """Reload ``api_server_runs`` while ``aiohttp.web_request`` has no ``RequestKey``
    (as on aiohttp < 3.14). Returns the module reloaded under the stub."""
    stub = types.ModuleType("aiohttp.web_request")
    real = sys.modules["aiohttp.web_request"]
    sys.modules["aiohttp.web_request"] = stub
    try:
        return importlib.reload(api_server_runs)
    finally:
        sys.modules["aiohttp.web_request"] = real


def _restore(api_server_runs):
    importlib.reload(api_server_runs)


def test_missing_requestkey_keeps_web_module_bound():
    import gateway.platforms.api_server_runs as api_server_runs

    try:
        _with_missing_requestkey(api_server_runs)
        assert api_server_runs.web is web
        assert api_server_runs.RequestKey is None
    finally:
        _restore(api_server_runs)

    # aiohttp < 3.14 has no RequestKey; the module binds None there, so compare against getattr.
    assert api_server_runs.RequestKey is getattr(aiohttp.web_request, "RequestKey", None)


def test_missing_requestkey_still_serves_the_admission_response():
    """The reported symptom: ``web`` was None, so the 202 admission response raised
    ``AttributeError: 'NoneType' object has no attribute 'json_response'`` (500 on POST /v1/runs)."""
    import gateway.platforms.api_server_runs as api_server_runs

    try:
        _with_missing_requestkey(api_server_runs)
        response = api_server_runs._accepted_response(
            "run-1", "started", "session-key", replayed=False)
        assert isinstance(response, web.Response)
        assert response.status == 202
        assert response.headers["X-Hermes-Session-Key"] == "session-key"
        assert "Idempotency-Replayed" not in response.headers
        # Room-retention state stays usable without RequestKey (string-keyed fallback).
        assert api_server_runs._ROOM_RETENTION_REQUEST_KEY == "hermes.room_run_retention_until"
    finally:
        _restore(api_server_runs)


def test_web_and_requestkey_unchanged_when_aiohttp_provides_requestkey():
    """Protection surface: with the version-gated name present, nothing changes."""
    import gateway.platforms.api_server_runs as api_server_runs

    request_key = getattr(aiohttp.web_request, "RequestKey", None)
    assert api_server_runs.web is web
    assert api_server_runs.RequestKey is request_key
    response = api_server_runs._accepted_response("run-2", "started", None, replayed=True)
    assert response.status == 202
    assert response.headers["Idempotency-Replayed"] == "true"
    if request_key is not None:
        assert isinstance(api_server_runs._ROOM_RETENTION_REQUEST_KEY, request_key)

