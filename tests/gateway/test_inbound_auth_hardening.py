"""Inbound webhook/callback auth: every remote-supplied secret or signature goes through a
timing-safe compare that fails closed on hostile input, and a signed timestamp outside the
replay window is refused even when the signature is valid."""

import asyncio
import hashlib
import hmac
import json
import os
import time
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest

from gateway.config import PlatformConfig

SECRET = "s3cret"
_AES_KEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
# A byte that is not UTF-8 reaches handlers as a lone surrogate (aiohttp surrogate-escapes header
# bytes; JSON decodes "\udcff" escapes to the same), so it is the hostile input that matters.
_HOSTILE = "tök\udcff"


def _wecom_crypt():
    from plugins.platforms.wecom.wecom_crypto import WXBizMsgCrypt
    return WXBizMsgCrypt(SECRET, _AES_KEY, "ww1234567890")


def _wecom_accepts(timestamp: str, presented_signature=None) -> bool:
    from plugins.platforms.wecom.wecom_crypto import SignatureError
    crypt = _wecom_crypt()
    root = ET.fromstring(crypt.encrypt("<xml/>", nonce="n", timestamp=timestamp))
    try:
        crypt.decrypt(presented_signature or root.findtext("MsgSignature"), timestamp, "n", root.findtext("Encrypt"))
    except SignatureError:
        return False
    return True


def _feishu_accepts(timestamp: str) -> bool:
    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import FeishuAdapter
    env = {"FEISHU_APP_ID": "cli", "FEISHU_APP_SECRET": "sec", "FEISHU_ENCRYPT_KEY": SECRET,
           "HERMES_HOME": os.environ["HERMES_HOME"]}
    with patch.dict(os.environ, env, clear=True):
        adapter = FeishuAdapter(PlatformConfig())
    body = b'{"type":"event"}'
    sig = hashlib.sha256(f"{timestamp}n{SECRET}".encode() + body).hexdigest()
    headers = {"x-lark-request-timestamp": timestamp, "x-lark-request-nonce": "n", "x-lark-signature": sig}
    return adapter._is_webhook_signature_valid(headers, body)


@pytest.mark.parametrize("accepts", [_wecom_accepts, _feishu_accepts], ids=["wecom", "feishu"])
def test_validly_signed_request_outside_replay_window_is_refused(accepts):
    now = int(time.time())
    assert accepts(str(now))
    assert not accepts(str(now - 600))
    assert not accepts(str(now + 600))


def _bluebubbles(presented: str) -> bool:
    pytest.importorskip("aiohttp")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": SECRET}))

    class _Request:
        query = {"password": presented}
        headers: dict = {}

        async def read(self):
            return json.dumps({"type": "typing-indicator"}).encode()

    return asyncio.run(adapter._handle_webhook(_Request())).status != 401


def _google_meet(presented: str) -> bool:
    from plugins.google_meet.node import protocol
    msg = {"type": "ping", "id": "1", "token": presented, "payload": {}}
    return protocol.validate_request(msg, SECRET)[0]


def _wecom_signature(presented: str) -> bool:
    return _wecom_accepts(str(int(time.time())), None if presented == SECRET else presented)


def _a2a(presented: str) -> bool:
    from plugins.platforms.a2a.security import A2ASecurityContext
    ctx = A2ASecurityContext(bearer_token=SECRET, peer_tokens=(), trusted_peers=frozenset(),
                             allow_all_users=False, requested_host="0.0.0.0", push_secret=SECRET)
    return ctx.authenticate(f"Bearer {presented}", "10.0.0.1") is not None


def _dashboard_basic(presented: str) -> bool:
    from hermes_cli.dashboard_auth import InvalidCredentialsError
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    provider = BasicAuthProvider(username=SECRET, password_hash=hash_password("pw"), secret=b"k" * 16)
    try:
        provider.complete_password_login(username=presented, password="pw")
    except InvalidCredentialsError:
        return False
    return True


def _dashboard_drain(presented: str) -> bool:
    from plugins.dashboard_auth.drain import DrainSecretProvider
    strong = "Zq8-rT2vLk9wXy4pBn7mC3sDf6gHj1KaQ0eW5uI8oP7aS2dF4"  # the provider refuses a weak secret
    token = strong if presented == SECRET else presented
    return DrainSecretProvider(secret=strong).verify_token(token=token) is not None


def _loopback_login_state(module_name: str, run, mismatch_code: str):
    """A browser-login callback carrying *presented* as its ``state``; True when it got past the check."""
    def accepts(presented: str) -> bool:
        import importlib
        from types import SimpleNamespace
        from hermes_cli.auth_constants import AuthError
        module = importlib.import_module(module_name)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(module.secrets, "token_urlsafe", lambda _n=32: SECRET)  # the state the login mints
            for target in (module, importlib.import_module("hermes_cli.auth_device_flow")):
                mp.setattr(target, "_bind_loopback_callback_server",
                           lambda *a, **k: SimpleNamespace(server_address=("127.0.0.1", 1455)), raising=False)
                mp.setattr(target, "_serve_loopback_callback", lambda *a, **k: {"state": presented}, raising=False)
            try:
                run(module)
            except AuthError as exc:
                return exc.code != mismatch_code  # past the state check, it stops at the missing code
        return True
    return accepts


_codex_browser_state = _loopback_login_state(
    "hermes_cli.auth_codex_browser", lambda m: m._codex_browser_login(open_browser=False),
    "codex_browser_state_mismatch")
_oauth_pkce_state = _loopback_login_state(
    "hermes_cli.auth_oauth_pkce_plugin",
    lambda m: m.login("example-pkce", m.OAuthPKCEConfig(client_id="c", authorize_url="https://idp.example/authorize",
                                                         token_url="https://idp.example/token"), open_browser=False),
    "oauth_state_mismatch")


def _mcp_flow():
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
    flow = DashboardOAuthFlow(flow_id="f1", server_name="reports", profile=None, hermes_home=os.environ["HERMES_HOME"],
                              redirect_uri="https://agent.example/api/mcp/oauth/callback/reports")
    asyncio.run(flow.publish_authorization_url(f"https://idp.example/authorize?state={SECRET}"))
    return flow


def _mcp_dashboard_callback(presented: str) -> bool:
    import hermes_cli.web_server_mcp as flows
    from hermes_cli.web_routers.mcp import mcp_oauth_callback
    flows._mcp_oauth_flows.clear()
    flows._mcp_oauth_flows["f1"] = _mcp_flow()
    try:
        return asyncio.run(mcp_oauth_callback("reports", code="c", state=presented)).status_code != 404
    finally:
        flows._mcp_oauth_flows.clear()


def _mcp_flow_deliver(presented: str) -> bool:
    try:
        _mcp_flow().deliver_callback(code="c", state=presented, error=None)
    except ValueError:
        return False
    return True


def _pairing_request_id(presented: str) -> bool:
    from gateway.pairing import PairingStore
    store = PairingStore()
    store.generate_code("telegram", f"user-{len(store.list_pending('telegram'))}-{time.time_ns()}")
    request_id = store.list_pending("telegram")[0]["request_id"]
    return store.approve_request("telegram", request_id if presented == SECRET else presented) is not None


@pytest.mark.parametrize("accepts", [_bluebubbles, _google_meet, _wecom_signature, _a2a, _dashboard_basic,
                                     _dashboard_drain, _codex_browser_state, _oauth_pkce_state,
                                     _mcp_dashboard_callback, _mcp_flow_deliver, _pairing_request_id],
                         ids=["bluebubbles", "google_meet", "wecom", "a2a", "dashboard_basic", "dashboard_drain",
                              "codex_browser_state", "oauth_pkce_state", "mcp_dashboard_callback",
                              "mcp_flow_deliver", "pairing_request_id"])
def test_presented_secret_is_compared_timing_safe_and_fails_closed(accepts, monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))

    assert accepts(SECRET)
    assert calls, "the secret was not compared with hmac.compare_digest"
    assert not accepts(SECRET + "x")
    assert not accepts(_HOSTILE)
