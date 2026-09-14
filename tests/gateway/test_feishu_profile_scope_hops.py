"""Profile scope must survive the Feishu adapter's thread hops under a multiplexed gateway.

The adapter is constructed and connected inside ``_profile_runtime_scope`` (HERMES_HOME override +
secret scope as contextvars). Two hops used to start from an EMPTY context, so the work ran under
the LAUNCH profile: the drive-comment agent turn (a full ``AIAgent`` on a bare default executor)
and the lark WS client thread (every SDK callback, and its ``run_coroutine_threadsafe`` hop back
onto the adapter loop, inherits the WS thread's context). The lark SDK is optional, so a fake
client drives the REAL ``_connect_websocket`` / ``_run_official_feishu_ws_client``.
"""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    get_secret,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def _observe(routed_home: Path) -> tuple:
    try:
        secret = get_secret("FEISHU_PROBE_TOKEN")
    except UnscopedSecretError:
        secret = "<unscoped>"
    return ("routed" if Path(get_hermes_home()) == routed_home else "launch", secret)


@pytest.fixture
def routed_scope(tmp_path):
    """Enter a secondary profile's scope the way gateway/run.py::_profile_runtime_scope does."""
    routed_home = tmp_path / "profiles" / "b"
    routed_home.mkdir(parents=True)
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(routed_home))
    secret_token = set_secret_scope({"FEISHU_PROBE_TOKEN": "routed-value"})
    try:
        yield routed_home
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_drive_comment_agent_turn_runs_under_the_adapter_profile_scope(routed_scope):
    """The comment agent thread must resolve model/credentials/home for the routed profile,
    not the launch profile (bare run_in_executor drops the contextvar scope)."""
    from plugins.platforms.feishu import feishu_comment as fc
    from plugins.platforms.feishu import feishu_comment_rules as rules

    seen = {}

    def _fake_agent(prompt, client, session_key=""):
        seen["turn"] = _observe(routed_scope)
        return "NO_REPLY"

    event = SimpleNamespace(event={
        "comment_id": "c1", "reply_id": "", "is_mentioned": True, "timestamp": "1",
        "notice_meta": {"file_token": "docx_t", "file_type": "docx", "notice_type": "add_reply",
                        "from_user_id": {"open_id": "ou_user"}, "to_user_id": {"open_id": "ou_bot"}},
    })
    with patch.object(rules, "load_config", return_value=object()), \
         patch.object(rules, "resolve_rule", return_value=rules.ResolvedCommentRule(True, "allowlist", frozenset(), "top")), \
         patch.object(rules, "has_wiki_keys", return_value=False), \
         patch.object(rules, "is_user_allowed", return_value=True), \
         patch.object(fc, "query_document_meta", AsyncMock(return_value={"title": "T", "url": "u"})), \
         patch.object(fc, "batch_query_comment", AsyncMock(return_value={"is_whole": False, "quote": ""})), \
         patch.object(fc, "_local_comment_prompt", AsyncMock(return_value="prompt")), \
         patch.object(fc, "_run_comment_agent", _fake_agent):
        asyncio.run(fc.handle_drive_comment_event(object(), event, self_open_id="ou_bot"))

    assert seen["turn"] == ("routed", "routed-value")


def test_ws_client_thread_and_its_loop_callbacks_carry_the_adapter_profile_scope(routed_scope, monkeypatch):
    """Lark SDK callbacks run on the WS thread and re-enter the adapter loop through
    run_coroutine_threadsafe (which copies the caller's context): both must see the profile that
    connected the adapter, so media caching / markers / env reads before handle_message stay in it."""
    from plugins.platforms.feishu import adapter as fa

    client_mod = types.ModuleType("lark_oapi.ws.client")
    client_mod.loop = SimpleNamespace(name="sdk-default-loop")
    client_mod.websockets = SimpleNamespace(connect=lambda *a, **k: None)
    lark_ws = types.ModuleType("lark_oapi.ws")
    lark_ws.client = client_mod
    lark = types.ModuleType("lark_oapi")
    lark.ws = lark_ws
    for name, mod in (("lark_oapi", lark), ("lark_oapi.ws", lark_ws), ("lark_oapi.ws.client", client_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(fa, "_WS_ISOLATION_INSTALLED", False)
    monkeypatch.setattr(fa, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(fa, "lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO=1)))

    seen = {}

    async def scenario():
        adapter_loop = asyncio.get_running_loop()

        async def _on_loop():
            return _observe(routed_scope)

        class FakeWSClient:
            def __init__(self, **kwargs):
                pass

            def start(self):  # the SDK-owned thread every lark callback fires on
                seen["ws_thread"] = _observe(routed_scope)
                seen["loop_callback"] = asyncio.run_coroutine_threadsafe(_on_loop(), adapter_loop).result(5)

        monkeypatch.setattr(fa, "FeishuWSClient", FakeWSClient)
        stub = SimpleNamespace(
            _loop=adapter_loop, _app_id="cli_x", _app_secret="s", _domain_name="feishu", _event_handler=None,
            _ws_thread_loop=None, _ws_reconnect_nonce=None, _ws_reconnect_interval=None,
            _ws_ping_interval=None, _ws_ping_timeout=None, _ws_client=None, _ws_future=None,
            _prepare_client=lambda: "feishu-domain", _hydrate_bot_identity=AsyncMock(),
        )
        await fa.FeishuAdapter._connect_websocket(stub)
        await stub._ws_future

    asyncio.run(scenario())
    assert seen == {"ws_thread": ("routed", "routed-value"), "loop_callback": ("routed", "routed-value")}
