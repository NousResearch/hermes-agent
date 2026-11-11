"""Modal provider profile contracts: fetch_models() feeds the hot-path hooks,
and modal.direct / modal.run hosts claim the modal profile."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from providers import get_provider_profile, profile_for_host


class _ModalCatalogHandler(BaseHTTPRequestHandler):
    items: list = []

    def do_GET(self):
        if self.path.rstrip("/") == "/models":
            body = json.dumps({"data": self.items}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def _catalog():
    profile = get_provider_profile("modal")
    return sys.modules[type(profile).__module__]._CATALOG


@pytest.fixture(autouse=True)
def clear_modal_catalog():
    _catalog().clear()
    yield
    _catalog().clear()


def _serve(items):
    _ModalCatalogHandler.items = items
    server = HTTPServer(("127.0.0.1", 0), _ModalCatalogHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_fetch_models_feeds_reasoning_and_context_hooks():
    server, port = _serve(
        [
            {
                "id": "ws--ep-kimi.us-west.modal.direct",
                "context_length": 1048576,
                "reasoning_options": [
                    {"type": "effort", "values": ["low", "high", "max"]}
                ],
            },
            {"id": "ws--ep-plain.us-west.modal.direct", "context_length": 131072},
        ]
    )
    try:
        profile = get_provider_profile("modal")
        ids = profile.fetch_models(
            api_key="wk-x.ws-y", base_url=f"http://127.0.0.1:{port}"
        )
        assert ids == [
            "ws--ep-kimi.us-west.modal.direct",
            "ws--ep-plain.us-west.modal.direct",
        ]

        assert profile.supported_reasoning_efforts(
            "ws--ep-kimi.us-west.modal.direct"
        ) == ("low", "high", "max")
        assert (
            profile.supported_reasoning_efforts("ws--ep-plain.us-west.modal.direct")
            == ()
        )
        assert profile.supported_reasoning_efforts("never-fetched") is None

        assert (
            profile.get_model_context_length("ws--ep-kimi.us-west.modal.direct")
            == 1048576
        )
        assert (
            profile.get_model_context_length("ws--ep-plain.us-west.modal.direct")
            == 131072
        )
        assert profile.get_model_context_length("never-fetched") is None

        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="ws--ep-kimi.us-west.modal.direct",
        )
        assert top == {"reasoning_effort": "low"}
        _, top = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="ws--ep-kimi.us-west.modal.direct",
        )
        assert top == {"reasoning_effort": "none"}
        assert profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"},
            model="ws--ep-plain.us-west.modal.direct",
        ) == ({}, {})
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "url",
    [
        "https://inference.us-west.modal.direct/v1",
        "https://acme--qwen-chat.modal.run",
    ],
)
def test_profile_claims_modal_hosts(url):
    assert profile_for_host(url, metadata=True).name == "modal"


def test_unrelated_hosts_not_claimed():
    assert profile_for_host("https://example.com/v1", metadata=True) is None
    assert profile_for_host("https://modal.direct.evil/v1", metadata=True) is None
