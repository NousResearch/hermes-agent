"""One OAuth refresh attempt in its own process, for the two-process refresh race tests.

Run by ``tests/agent/test_oauth_root_refresh_race.py`` as
``python -I oauth_refresh_worker.py MODE PORT OUT_JSON NAME`` with ``HERMES_HOME`` set to a
profile of a temporary root. It is a script, not a test module, and imports Hermes from the
checkout it lives in.

MODE is ``pool-<provider>`` (load the credential pool, then ``select()``), ``pool-<provider>-401``
(select, then force-refresh the credential as a request that came back 401 would) or
``single-xai`` (``resolve_xai_oauth_runtime_credentials()``). The process:

1. confines itself to loopback and redirects every ``https`` token endpoint to the mock
   server on ``127.0.0.1:PORT`` (the host it meant to reach is sent as ``X-Orig-Host``);
2. imports Hermes and loads what it will refresh (the pool is loaded here, so it can be stale
   later; the xAI resolver is imported here), so only the refresh itself is left after the barrier;
3. creates ``OUT_JSON + ".ready"``, then blocks reading one line from stdin: the parent
   releases it with ``go``, which is how two workers start at the same instant;
4. performs the refresh and writes its outcome to ``OUT_JSON``.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import traceback
import urllib.parse
from pathlib import Path

mode, port, out, name = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _connect(self, address):
    if isinstance(address, tuple) and address[0] not in _LOOPBACK:
        raise OSError(f"blocked non-loopback connect to {address[0]}")
    return _real_connect(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if host not in _LOOPBACK:
        raise socket.gaierror(f"blocked DNS lookup for {host}")
    return _real_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.getaddrinfo = _getaddrinfo

import httpx

_real_handle_request = httpx.HTTPTransport.handle_request


def _handle_request(self, request):
    url = request.url
    headers = [(k, v) for k, v in request.headers.raw if k.lower() != b"host"]
    headers += [(b"X-Orig-Host", url.host.encode()), (b"X-Worker", name.encode())]
    local = httpx.Request(
        request.method,
        url.copy_with(scheme="http", host="127.0.0.1", port=port),
        headers=headers,
        content=request.read(),
    )
    return _real_handle_request(self, local)


httpx.HTTPTransport.handle_request = _handle_request

import urllib.request  # Anthropic's token calls go through urllib, not httpx

_real_urlopen = urllib.request.urlopen


def _urlopen(request, *args, **kwargs):
    if isinstance(request, urllib.request.Request) and request.full_url.startswith("https://"):
        host = request.host
        headers = dict(request.header_items())
        headers.update({"X-Orig-Host": host, "X-Worker": name})
        local_url = f"http://127.0.0.1:{port}{urllib.parse.urlsplit(request.full_url).path}"
        request = urllib.request.Request(local_url, data=request.data, method=request.get_method(), headers=headers)
    return _real_urlopen(request, *args, **kwargs)


urllib.request.urlopen = _urlopen

result: dict = {"name": name, "mode": mode}
try:
    pool = None
    if mode == "single-xai":
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
    else:
        from agent.credential_pool import load_pool

        pool = load_pool(mode[len("pool-"):].removesuffix("-401"))
    out.with_name(out.name + ".ready").write_text("ready", encoding="utf-8")
    sys.stdin.readline()  # the parent's start barrier
    if pool is not None:
        entry = pool.select()
        if mode.endswith("-401") and entry is not None:
            entry = pool.try_refresh_matching(api_key_hint=entry.runtime_api_key)
        result["selected"] = entry is not None
        result["refresh_token"] = getattr(entry, "refresh_token", None)
        result["access_token"] = getattr(entry, "access_token", None)
        result["status"] = getattr(entry, "last_status", None)
    else:
        creds = resolve_xai_oauth_runtime_credentials()
        result["selected"] = bool(creds.get("api_key"))
        result["access_token"] = creds.get("api_key")
    result["ok"] = True
except BaseException as exc:  # the parent reports whatever went wrong
    result["ok"] = False
    result["error"] = f"{type(exc).__name__}: {exc}"[:400]
    result["trace"] = traceback.format_exc()[-1500:]
out.write_text(json.dumps(result), encoding="utf-8")
