"""``hermes webhook subscribe`` pins new routes to ``signature: generic`` and ``webhook test`` signs so the
real adapter accepts it under that pin (#84265)."""
import json
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from multidict import CIMultiDict

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.webhook import WebhookAdapter
from hermes_agent.hermes_cli.webhook import _load_subscriptions, _mutate_subscriptions, webhook_command


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_agent.hermes_cli.webhook._is_webhook_enabled", lambda: True)


def _args(**kwargs):
    base = dict(webhook_action="subscribe", name="hook", prompt="", events="", description="", skills="",
                deliver="log", deliver_chat_id="", secret="", route_profile=None, payload="", script="",
                signature=None)
    return Namespace(**{**base, **kwargs})


def _send_test(monkeypatch, name: str):
    """Run ``hermes webhook test`` and return the request it would have POSTed."""
    sent = []

    class _Resp:
        status = 202
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def _urlopen(req, timeout):
        sent.append(req)
        return _Resp()

    monkeypatch.setattr("hermes_agent.hermes_cli.webhook.urllib.request.urlopen", _urlopen)
    webhook_command(_args(webhook_action="test", name=name))
    (req,) = sent
    return req


def _adapter_accepts(route: dict, req) -> bool:
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {}}))
    request = MagicMock()
    request.headers = CIMultiDict(req.header_items())  # aiohttp headers are case-insensitive
    request.match_info = {"route_name": "hook"}
    return adapter._validate_signature(request, req.data, route["secret"], route.get("signature"))


def test_subscribe_pins_new_route_to_generic_and_test_signs_v2(monkeypatch):
    webhook_command(_args())
    route = _load_subscriptions()["hook"]
    assert route["signature"] == "generic"

    req = _send_test(monkeypatch, "hook")
    headers = {k.lower(): v for k, v in req.header_items()}
    assert "x-webhook-signature-v2" in headers and "x-webhook-timestamp" in headers
    assert "x-hub-signature-256" not in headers
    assert _adapter_accepts(route, req)


def test_provider_pin_is_kept_and_test_signs_with_the_provider_scheme(monkeypatch):
    webhook_command(_args(signature="github"))
    webhook_command(_args(description="updated"))  # an update without --signature keeps the pin
    route = _load_subscriptions()["hook"]
    assert route["signature"] == "github"

    req = _send_test(monkeypatch, "hook")
    assert req.get_header("X-hub-signature-256", "").startswith("sha256=")
    assert _adapter_accepts(route, req)


@pytest.mark.parametrize("pin", ["generic", "github", "gitlab", "linear"])
def test_webhook_test_signs_so_the_pinned_route_accepts_it(monkeypatch, pin):
    webhook_command(_args(signature=pin))
    route = _load_subscriptions()["hook"]
    assert _adapter_accepts(route, _send_test(monkeypatch, "hook"))


def test_update_of_pre_pin_route_stays_unpinned(monkeypatch):
    """A route created before pins existed may be a GitHub/Linear sender: updating it must not pin it
    to generic and start rejecting that provider."""
    _mutate_subscriptions(lambda subs: subs.update(hook={"secret": "s", "prompt": "", "deliver": "log"}))
    webhook_command(_args(description="updated"))
    route = _load_subscriptions()["hook"]
    assert "signature" not in route
    assert json.loads(json.dumps(route))["secret"] == "s"


@pytest.mark.parametrize("bad_pin", [False, ""])
def test_update_keeps_an_invalid_pin_so_the_gateway_fails_closed(bad_pin):
    """An update without --signature must not turn an invalid pin into "unpinned" (which accepts the
    body-only GitHub/Linear MACs again); the gateway refuses the route while the bad pin stays."""
    _mutate_subscriptions(lambda subs: subs.update(
        hook={"secret": "s", "prompt": "", "deliver": "log", "signature": bad_pin}))
    webhook_command(_args(description="updated"))
    route = _load_subscriptions()["hook"]
    assert route["signature"] == bad_pin
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {}}))
    assert adapter._dynamic_route_allowed("hook", route) is False
