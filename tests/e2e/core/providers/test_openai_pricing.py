"""Model-picker pricing for a config-defined twin of a priced aggregator (#120757).

Tier (stated honestly): a REAL ``python -m tui_gateway.entry`` process — the backend the TUI
and Desktop model picker talk to — answering the ``model.options`` JSON-RPC exactly as the
picker calls it. The only fake is the vendor's price catalog: the aggregator catalog URL
(``https://openrouter.ai/api/v1/models``) is a hard-coded https constant with no override and
lives only in process memory, so a ``sitecustomize`` shim on the child's ``PYTHONPATH``
rewrites exactly that origin, at the ``urllib`` opener layer, to a loopback fake serving the
vendor's catalog shape. No Hermes function is patched. Every other egress is pinned to a
closed loopback proxy so nothing reaches a real vendor.

The config is the issue's: a ``providers:`` entry keyed ``openrouter`` pointing at the
aggregator's base URL. Its picker row is ``custom:openrouter``; the built-in ``openrouter``
row (credentialed from ``.env``) is the control that prices from the same catalog. Two relays
serving a catalog model id are decoys that must NOT borrow its prices: ``myrelay`` (a plain
``myrelay`` row) and one keyed ``deepseek`` — a built-in, non-aggregator provider name, so its
row takes the same ``custom:<key>`` slug path as the twin. Prices must follow the upstream a
row actually talks to, not the ``custom:`` slug shape.

The picker is driven with ``refresh: true`` (its explicit reload: catalogs are fetched
synchronously, so no background prewarm timing is involved). ``get_pricing_for_provider`` is
the single choke point for both that path and the cached-only normal open.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.providers._openai_helpers import (
    REPO_ROOT,
    Home,
    bug_assertions,
    known_marks,
    write_sitecustomize_shim,
)
from tests.e2e.core.providers._openai_tui import TuiGateway

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

KNOWN: dict[str, str] = {
    "custom_twin_priced": "#120757 custom:<key> row pointing at a priced aggregator gets no picker prices",
}


def known(name: str) -> list:
    return known_marks(KNOWN, name)


VENDOR_ORIGIN = "https://openrouter.ai"
# (id, $/token prompt, $/token completion) — the catalog the fake aggregator publishes.
CATALOG = [
    ("fakevendor/alpha-large", "0.000003", "0.000015"),
    ("fakevendor/beta-mini", "0.0000004", "0.0000016"),
    ("fakevendor/gamma-coder", "0.000002", "0.000008"),
]
TWIN_MODELS = [mid for mid, _p, _c in CATALOG]
RELAY_MODEL = TWIN_MODELS[-1]
# (config key, picker slug) of each decoy relay. ``deepseek`` collides with a built-in
# non-aggregator provider, so its row is ``custom:deepseek`` -- the twin's slug shape.
DECOYS = [("myrelay", "myrelay"), ("deepseek", "custom:deepseek")]

_SHIM = '''\
"""E2E vendor-boundary shim: route a hard-coded vendor origin to a loopback fake."""
import os

_pairs = [p.split("=", 1) for p in os.environ.get("HERMES_E2E_ORIGIN_REDIRECT", "").split(";") if "=" in p]
if _pairs:
    import urllib.request

    _open = urllib.request.OpenerDirector.open

    def open(self, fullurl, *args, **kwargs):
        req = fullurl if isinstance(fullurl, urllib.request.Request) else None
        url = req.full_url if req is not None else str(fullurl)
        for src, dst in _pairs:
            if url.startswith(src + "/"):
                url = dst + url[len(src):]
                if req is not None:
                    req.full_url = url
                else:
                    fullurl = url
                break
        return _open(self, fullurl, *args, **kwargs)

    urllib.request.OpenerDirector.open = open
'''


class FakeCatalog:
    """The aggregator's public ``GET /api/v1/models`` price catalog (vendor JSON shape), plus
    the decoy relay's plain ``GET /relay/v1/models`` listing on the same loopback server."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeCatalog":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:  # noqa: D401 - silence per-request logging
                pass

            def do_GET(self) -> None:  # noqa: N802
                server.requests.append((self.path, self.headers.get("Authorization", "")))
                path = self.path.split("?")[0].rstrip("/")
                if path == "/relay/v1/models":  # the decoy relay: plain OpenAI model list, no prices
                    status, body = 200, {"object": "list", "data": [{"id": RELAY_MODEL, "object": "model"}]}
                elif path == "/api/v1/models":
                    status, body = 200, {"data": [
                        {"id": mid, "name": mid, "context_length": 200000, "created": 1780000000,
                         "architecture": {"modality": "text->text", "input_modalities": ["text"],
                                          "output_modalities": ["text"]},
                         "pricing": {"prompt": p, "completion": c, "request": "0", "image": "0"},
                         "supported_parameters": ["tools", "reasoning", "max_tokens"]}
                        for mid, p, c in CATALOG]}
                else:
                    status, body = 404, {"error": {"message": "not found"}}
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


class _ShimHome(Home):
    """A Home whose child env carries the vendor-boundary shim and the egress guard."""

    extra: dict[str, str]

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return super().env({**self.extra, **(extra or {})})


def _closed_port() -> socket.socket:
    """A bound, never-listening loopback socket: connecting to it is refused at once."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock


def _home(tmp_path: Path, catalog: FakeCatalog, dead: socket.socket) -> _ShimHome:
    h = _ShimHome(tmp_path)
    write_sitecustomize_shim(tmp_path / "shim", _SHIM)
    proxy = f"http://127.0.0.1:{dead.getsockname()[1]}"
    h.extra = {"PYTHONPATH": f"{tmp_path / 'shim'}:{REPO_ROOT}",
               "HERMES_E2E_ORIGIN_REDIRECT": f"{VENDOR_ORIGIN}={catalog.origin}",
               **{k: proxy for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")},
               "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    h.write({
        "model": {"provider": "openrouter", "default": TWIN_MODELS[0]},
        "providers": {
            "openrouter": {"name": "OpenRouter-Full", "base_url": f"{VENDOR_ORIGIN}/api/v1",
                           "api_key": "sk-or-fake-twin", "models": TWIN_MODELS},
            # Decoys: serve a catalog model id, but are relays with no price catalog of their own.
            **{key: {"name": f"Relay {key}", "base_url": f"{catalog.origin}/relay/v1",
                     "api_key": "sk-relay-fake", "models": [RELAY_MODEL]} for key, _slug in DECOYS},
        },
    }, dotenv={"OPENROUTER_API_KEY": "sk-or-fake-builtin",
               # The picker lists a providers: entry keyed by a built-in name only while that
               # built-in is credentialed; this makes the custom:deepseek decoy row appear.
               "DEEPSEEK_API_KEY": "sk-ds-fake-builtin"})
    return h


def _row(payload: dict[str, Any], slug: str) -> dict[str, Any] | None:
    return next((r for r in payload.get("providers") or [] if r.get("slug") == slug), None)


def _priced(row: dict[str, Any] | None) -> dict[str, Any]:
    """``{model: price entry}`` for the row's models that carry a real price."""
    pricing = (row or {}).get("pricing") or {}
    return {m: p for m, p in pricing.items() if isinstance(p, dict) and (p.get("input") or p.get("output"))}


def _refreshed_options(tmp_path: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """One picker refresh (the picker's explicit reload: fetches price catalogs synchronously,
    so no background timing is involved) against a real gateway."""
    dead = _closed_port()
    try:
        with FakeCatalog() as catalog:
            h = _home(tmp_path, catalog, dead)
            gw = TuiGateway(h)
            try:
                payload = gw.call("model.options", {"refresh": True})
            finally:
                gw.close()
            return payload, list(catalog.requests)
    finally:
        dead.close()


@pytest.fixture(scope="module")
def picker(tmp_path_factory) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    return _refreshed_options(tmp_path_factory.mktemp("pricing"))


def _expected_price(mid: str) -> tuple[float, float]:
    """The catalog's own $/token for *mid*, as $/MTok."""
    _m, p, c = next(row for row in CATALOG if row[0] == mid)
    return float(p) * 1e6, float(c) * 1e6


def _dollars(text: str) -> float:
    return float(str(text).lstrip("$"))


def test_builtin_aggregator_row_prices_from_catalog(picker) -> None:
    """Control (green on main): the built-in aggregator row prices every catalog model it
    lists with the catalog's own $/MTok, read from the aggregator's catalog endpoint."""
    payload, requests = picker
    assert any(path.startswith("/api/v1/models") for path, _auth in requests), requests
    row = _row(payload, "openrouter")
    priced = _priced(row)
    listed = [m for m in (row or {}).get("models") or [] if m in TWIN_MODELS]
    assert listed and set(listed) <= set(priced), (listed, (row or {}).get("pricing"))
    for mid in listed:
        assert (_dollars(priced[mid]["input"]), _dollars(priced[mid]["output"])) == pytest.approx(
            _expected_price(mid)), (mid, priced[mid])


@pytest.mark.parametrize("slug", [slug for _key, slug in DECOYS])
def test_relay_without_catalog_gets_no_borrowed_prices(picker, slug: str) -> None:
    """Control (green on main): a config-defined relay that is NOT a priced aggregator serves
    a catalog model id but must not borrow the aggregator's prices -- including the
    ``custom:deepseek`` row, whose slug has the same ``custom:<key>`` shape as the twin."""
    payload, _requests = picker
    relay = _row(payload, slug)
    assert relay is not None and RELAY_MODEL in (relay.get("models") or []), payload.get("providers")
    assert _priced(relay) == {}, f"{slug} (upstream: the relay) borrowed prices: {relay.get('pricing')}"


@pytest.mark.parametrize("scenario", [pytest.param("custom_twin_priced", marks=known("custom_twin_priced"))])
def test_config_defined_twin_row_is_priced_like_the_aggregator(picker, scenario) -> None:
    """The ``custom:openrouter`` row serves the same upstream as the built-in row; its models
    must carry the aggregator's prices, not none."""
    payload, _requests = picker
    twin, builtin = _row(payload, "custom:openrouter"), _row(payload, "openrouter")
    assert twin is not None and set(TWIN_MODELS) <= set(twin.get("models") or []), payload.get("providers")
    priced = _priced(twin)
    with bug_assertions():
        assert set(priced) == set(TWIN_MODELS), (
            f"custom:openrouter row has {len(priced)}/{len(TWIN_MODELS)} models priced "
            f"(pricing={twin.get('pricing')!r})")
        for mid, entry in _priced(builtin).items():
            if mid in priced:
                assert priced[mid] == entry, (mid, priced[mid], entry)
