"""Regression for #120704: Slack OTP input stays out of model-visible chat.

Failure model covered through the real TurnRunner → vault tool → Slack adapter path:
- the prompt is bound to the user, chat, workspace, session, and a short lifetime;
- an authorized but different Slack user cannot open or submit it;
- the submitted code is single-use and appears only in the supervised page write;
- send failure/timeout releases the blocked tool and rejects late submissions;
- prompt completion/expiry removes the live Slack control without echoing the code.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock() -> None:
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

# Production defers the heavyweight gateway facade until a turn runs. Import it during collection,
# before the test suite's real-home I/O guard is armed, so worker callbacks are order-independent.
import gateway.run as _gateway_run  # noqa: E402,F401

from agent.vault_backends.unlock import get_code_prompt_callback  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from gateway.run_turn_runner import TurnRunner  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from tools import browser_vault_tool  # noqa: E402


class _FakeBoltApp:
    def __init__(self, client):
        self.client = client
        self.actions: dict[str, object] = {}
        self.views: dict[str, object] = {}

    @staticmethod
    def _decorator(store, key):
        def register(handler):
            if isinstance(key, str):
                store[key] = handler
            return handler
        return register

    def action(self, key):
        return self._decorator(self.actions, key)

    def view(self, key):
        return self._decorator(self.views, key)

    def view_closed(self, key):
        return self._decorator(self.views, f"{key}_closed")

    def event(self, key):
        return self._decorator({}, key)

    def command(self, key):
        return self._decorator({}, key)


class _SlackClient:
    def __init__(self):
        self.posted: list[dict] = []
        self.opened: list[dict] = []
        self.updated: list[dict] = []
        self.prompt_posted = threading.Event()
        self.post_started = threading.Event()
        self.post_error = False
        self.open_error = False
        self.post_release: threading.Event | None = None

    async def chat_postMessage(self, **kwargs):
        self.post_started.set()
        if self.post_error:
            raise RuntimeError("send failed")
        if self.post_release is not None:
            await asyncio.to_thread(self.post_release.wait)
        self.posted.append(kwargs)
        self.prompt_posted.set()
        return {"ok": True, "channel": kwargs["channel"], "ts": "171.001"}

    async def views_open(self, **kwargs):
        if self.open_error:
            raise RuntimeError("modal failed")
        self.opened.append(kwargs)
        return {"ok": True}

    async def chat_update(self, **kwargs):
        self.updated.append(kwargs)
        return {"ok": True}


class _GatewayRunner:
    @staticmethod
    def _consume_pending_native_image_paths(_session_key):
        return []


class _BrowserAgent:
    def __init__(self):
        self.callback_seen = False
        self.secret_expression = ""

    def run_conversation(self, _message, **_kwargs):
        from model_tools import handle_function_call

        self.callback_seen = get_code_prompt_callback() is not None
        raw = handle_function_call(
            "browser_vault_enter_code", {}, task_id="slack-login")
        return {"final_response": raw, "messages": [], "completed": True}


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def _adapter(client: _SlackClient) -> tuple[SlackAdapter, _FakeBoltApp]:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    app = _FakeBoltApp(client)
    adapter._app = app
    adapter._team_clients = {"T1": client}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    adapter.pause_typing_for_chat = MagicMock()
    adapter.resume_typing_for_chat = MagicMock()
    adapter._register_plugin_action_handlers = lambda: None
    adapter._wire_plugin_handlers = lambda _app: None
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._register_bolt_handlers()
    return adapter, app


def _turn(
    adapter: SlackAdapter,
    loop: asyncio.AbstractEventLoop,
    *,
    team_id: str = "T1",
    current: list[bool] | None = None,
    run_generation: int = 7,
) -> TurnRunner:
    current = current or [True]
    source = SimpleNamespace(user_id="U_OWNER", user_name="owner", scope_id=team_id)
    ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="C1",
        _status_thread_metadata={"team_id": team_id} if team_id else {},
        _loop_for_step=loop,
        session_key="agent:main:slack:group:T1:C1",
        session_id="session-1",
        run_generation=run_generation,
        process_task_id="gateway-turn-1",
        _run_still_current=lambda: current[0],
        source=source,
        title_user_message=None,
        message="sign in",
        persist_user_message=None,
        persist_user_display_kind=None,
        persist_user_display_metadata=None,
        persist_user_timestamp=None,
        moa_config=None,
        inbound_message_id="171.000",
        mute_notification_reply=False,
        stream_consumer_holder=[],
    )
    return TurnRunner(_GatewayRunner(), ctx)


def _patch_browser(monkeypatch, secret_expression: list[str]) -> None:
    controls = [{
        "index": 0,
        "type": "text",
        "name": "otp",
        "label": "Verification code",
        "autocomplete": "one-time-code",
    }]
    monkeypatch.setattr(browser_vault_tool, "_focus_bound_origin", lambda *_a, **_k: None)
    monkeypatch.setattr(
        browser_vault_tool, "_current_page_origin", lambda _task_id: "https://acme.test")
    monkeypatch.setattr(
        browser_vault_tool,
        "_eval_js",
        lambda _task_id, _expr: {"success": True, "result": json.dumps(controls)},
    )

    def fill(_task_id, expression):
        secret_expression.append(expression)
        return {"success": True, "result": json.dumps({"filled": 1})}

    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", fill)


async def _click(
    app: _FakeBoltApp, *, user_id: str, request_id: str, team_id: str = "T1"
):
    ack = AsyncMock()
    body = {
        "team": {"id": team_id},
        "trigger_id": f"trigger-{user_id}",
        "message": {"ts": "171.001", "blocks": []},
        "channel": {"id": "C1"},
        "user": {"id": user_id, "name": user_id.lower()},
    }
    action = {"action_id": "hermes_vault_code_open", "value": request_id}
    await app.actions["hermes_vault_code_open"](ack=ack, body=body, action=action)
    return ack


async def _submit(
    app: _FakeBoltApp,
    *,
    user_id: str,
    request_id: str,
    code: str,
    team_id: str = "T1",
    team_shape: str = "top_level",
):
    ack = AsyncMock()
    view = {
        "callback_id": "hermes_vault_code_submit",
        "private_metadata": request_id,
        "state": {"values": {
            "hermes_vault_code_block": {
                "hermes_vault_code_value": {"type": "plain_text_input", "value": code}
            }
        }},
    }
    if team_shape == "app_installed":
        view["app_installed_team_id"] = team_id
        body = {"team": None, "user": {"id": user_id, "name": user_id.lower()}, "view": view}
    elif team_shape == "user_team":
        body = {"team": None, "user": {"id": user_id, "name": user_id.lower(), "team_id": team_id}, "view": view}
    else:
        body = {"team": {"id": team_id}, "user": {"id": user_id, "name": user_id.lower()}, "view": view}
    await app.views["hermes_vault_code_submit"](ack=ack, body=body, view=view)
    return ack


async def _close(
    app: _FakeBoltApp,
    *,
    user_id: str,
    request_id: str,
    team_id: str = "T1",
    team_shape: str = "top_level",
):
    ack = AsyncMock()
    view = {"callback_id": "hermes_vault_code_submit", "private_metadata": request_id}
    if team_shape == "app_installed":
        view["app_installed_team_id"] = team_id
        body = {"team": None, "user": {"id": user_id, "name": user_id.lower()}, "view": view}
    elif team_shape == "user_team":
        body = {
            "team": None,
            "user": {"id": user_id, "name": user_id.lower(), "team_id": team_id},
            "view": view,
        }
    else:
        body = {"team": {"id": team_id}, "user": {"id": user_id, "name": user_id.lower()}, "view": view}
    await app.views["hermes_vault_code_submit_closed"](ack=ack, body=body, view=view)
    return ack


def test_slack_modal_code_reaches_browser_without_entering_model_context(
    gateway_loop, monkeypatch, caplog
):
    """One real synchronous tool call crosses the worker/async-adapter boundary and resumes."""
    client = _SlackClient()
    adapter, app = _adapter(client)
    turn = _turn(adapter, gateway_loop, run_generation=101)
    secret_expression: list[str] = []
    _patch_browser(monkeypatch, secret_expression)
    agent = _BrowserAgent()
    outcome: dict[str, object] = {}

    def run_turn():
        outcome["result"] = turn._run_conversation_with_approval(
            agent, [], None, None, None)
        outcome["callback_after"] = get_code_prompt_callback()

    worker = threading.Thread(target=run_turn)
    worker.start()
    assert client.prompt_posted.wait(timeout=5), "Slack secure-code prompt was not posted"

    button = client.posted[0]["blocks"][1]["elements"][0]
    request_id = button["value"]
    assert request_id.startswith("secure_")

    # Neither another authorized user nor the same Slack ids in another workspace may claim it.
    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OTHER", request_id=request_id), gateway_loop).result(timeout=5)
    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OWNER", request_id=request_id, team_id="T_OTHER"), gateway_loop
    ).result(timeout=5)
    assert client.opened == []

    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OWNER", request_id=request_id), gateway_loop).result(timeout=5)
    assert len(client.opened) == 1
    assert client.opened[0]["view"]["private_metadata"] == request_id
    # Claim is exclusive: repeated clicks cannot create two modals whose close events race.
    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OWNER", request_id=request_id), gateway_loop).result(timeout=5)
    assert len(client.opened) == 1

    # Even after the owner opens the modal, a forged submission by another user cannot resolve it.
    other_ack = asyncio.run_coroutine_threadsafe(
        _submit(app, user_id="U_OTHER", request_id=request_id, code="111111"), gateway_loop
    ).result(timeout=5)
    assert other_ack.call_args.kwargs.get("response_action") == "errors"
    assert worker.is_alive()

    asyncio.run_coroutine_threadsafe(
        _submit(
            app,
            user_id="U_OWNER",
            request_id=request_id,
            code="246810",
            team_shape="app_installed",
        ),
        gateway_loop,
    ).result(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive()

    result = outcome["result"]
    browser_result = json.loads(result["final_response"])
    assert agent.callback_seen is True
    assert outcome["callback_after"] is None
    assert browser_result["success"] is True
    assert browser_result["source"] == "user"
    assert "246810" in secret_expression[0]

    model_visible = json.dumps(result, ensure_ascii=False)
    outbound_slack = json.dumps(
        {"posted": client.posted, "opened": client.opened, "updated": client.updated},
        ensure_ascii=False,
    )
    assert "246810" not in model_visible
    assert "246810" not in outbound_slack
    assert "246810" not in caplog.text

    # The handle is single-use: a replay cannot wake or mutate anything.
    replay_ack = asyncio.run_coroutine_threadsafe(
        _submit(app, user_id="U_OWNER", request_id=request_id, code="999999"), gateway_loop
    ).result(timeout=5)
    assert replay_ack.call_args.kwargs.get("response_action") == "errors"


def test_slack_secure_prompt_cancel_and_timeout_clean_up_and_reject_late_code(
    gateway_loop, monkeypatch, tmp_path
):
    from gateway import secure_input

    client = _SlackClient()
    adapter, app = _adapter(client)

    # Slack workspace scope is mandatory; ambiguous routing must fail before posting a control.
    adapter._channel_team = {}
    adapter._team_clients = {"T1": client, "T2": client}
    unscoped_turn = _turn(adapter, gateway_loop, team_id="", run_generation=201)
    monkeypatch.setattr(secure_input, "SECURE_INPUT_TIMEOUT_SECONDS", 0.01)
    assert unscoped_turn._vault_code_callback_sync("acme.test", "") == ""
    assert client.posted == []
    adapter._team_clients = {"T1": client}
    adapter._channel_team = {"C1": "T1"}

    # Definitive send failure releases the broker and restores typing/input state.
    client.post_error = True
    failed_turn = _turn(adapter, gateway_loop, run_generation=202)
    assert failed_turn._vault_code_callback_sync("acme.test", "") == ""
    assert secure_input.pending_count() == 0
    assert adapter.pause_typing_for_chat.call_count == adapter.resume_typing_for_chat.call_count
    client.post_error = False
    adapter.pause_typing_for_chat.reset_mock()
    adapter.resume_typing_for_chat.reset_mock()

    # A displaced run revokes even a resolved-but-unconsumed code without touching its successor.
    session_key = "agent:main:slack:group:T1:C1"

    def register_request(run_generation: int, profile_home=tmp_path):
        return secure_input.register(
            session_key=session_key,
            session_id="session-1",
            run_generation=run_generation,
            task_id="gateway-turn",
            expected_user_id="U_OWNER",
            chat_id="C1",
            scope_id="T1",
            site="acme.test",
            profile_home=str(profile_home),
            timeout=5,
        )

    # An interrupt that wins the race before registration tombstones that run: a stale worker cannot
    # post a fresh secret prompt after /stop or /new has already returned.
    secure_input.clear_run(session_key, 206)
    with pytest.raises(RuntimeError, match="no longer active"):
        register_request(206)
    clock = secure_input.time.monotonic()
    with monkeypatch.context() as delayed_clock:
        delayed_clock.setattr(secure_input.time, "monotonic", lambda: clock + 86_400)
        with pytest.raises(RuntimeError, match="no longer active"):
            register_request(206)
    posts_before_stale_callback = len(client.posted)
    assert _turn(adapter, gateway_loop, run_generation=206)._vault_code_callback_sync(
        "acme.test", ""
    ) == ""
    assert len(client.posted) == posts_before_stale_callback

    displaced = register_request(207)
    successor = register_request(208)
    assert secure_input.claim(
        displaced.request_id, user_id="U_OWNER", chat_id="C1", scope_id="T1") is displaced
    assert secure_input.resolve(
        displaced.request_id, "135790", user_id="U_OWNER", scope_id="T1") is displaced
    secure_input.clear_run(session_key, 207)
    assert secure_input.wait(displaced) == ""
    assert secure_input.is_pending(successor)
    secure_input.cancel(successor)

    # Deferred modal rendering re-enters the request's profile, including A→B→A multiplex routing.
    import plugins.platforms.slack.adapter as slack_module
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = [tmp_path / "profile-a", tmp_path / "profile-b", tmp_path / "profile-a"]
    for home in set(homes):
        home.mkdir(parents=True)
    rendered_homes: list[str] = []
    terminal_homes: list[str] = []
    real_t = slack_module.t

    def tracked_t(key, **kwargs):
        if key == "platform.slack.secure_input.modal_title":
            rendered_homes.append(str(get_hermes_home()))
        elif key == "gateway.secure_input.expired":
            terminal_homes.append(str(get_hermes_home()))
        return real_t(key, **kwargs)

    monkeypatch.setattr(slack_module, "t", tracked_t)
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for index, home in enumerate(homes):
            request = register_request(220 + index, home)
            asyncio.run_coroutine_threadsafe(
                _click(app, user_id="U_OWNER", request_id=request.request_id), gateway_loop
            ).result(timeout=5)
            asyncio.run_coroutine_threadsafe(
                _close(app, user_id="U_OWNER", request_id=request.request_id), gateway_loop
            ).result(timeout=5)
    finally:
        set_multiplex_active(was_multiplex)
    assert rendered_homes == [str(home) for home in homes]

    # Slack modal-open failure is definitive: wake the tool immediately and retire the dead card.
    client.open_error = True
    client.prompt_posted.clear()
    monkeypatch.setattr(secure_input, "SECURE_INPUT_TIMEOUT_SECONDS", 30.0)
    failed_open: dict[str, str] = {}
    failed_open_worker = threading.Thread(
        target=lambda: failed_open.setdefault(
            "answer", _turn(adapter, gateway_loop, run_generation=230)._vault_code_callback_sync(
                "acme.test", "")))
    failed_open_worker.start()
    assert client.prompt_posted.wait(timeout=5)
    failed_open_request = client.posted[-1]["blocks"][1]["elements"][0]["value"]
    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OWNER", request_id=failed_open_request), gateway_loop
    ).result(timeout=5)
    failed_open_worker.join(timeout=2)
    if failed_open_worker.is_alive():
        secure_input.clear_session(session_key)
        failed_open_worker.join(timeout=5)
    assert not failed_open_worker.is_alive()
    assert failed_open["answer"] == ""
    client.open_error = False
    client.prompt_posted.clear()

    # A post whose ack arrives after the send window is ambiguous, not a failure. If the run is
    # invalidated meanwhile, its eventual Slack card is immediately retired instead of left live.
    from gateway import run_turn_runner as runner_module

    monkeypatch.setattr(runner_module, "t", tracked_t)
    monkeypatch.setattr(runner_module, "_SECURE_INPUT_SEND_ACK_SECONDS", 0.01)
    monkeypatch.setattr(secure_input, "SECURE_INPUT_TIMEOUT_SECONDS", 30.0)
    client.post_started.clear()
    client.post_release = threading.Event()
    current = [True]
    delayed_turn = _turn(adapter, gateway_loop, current=current, run_generation=231)
    delayed: dict[str, str] = {}

    def run_delayed() -> None:
        token = set_hermes_home_override(str(homes[1]))
        try:
            delayed["answer"] = delayed_turn._vault_code_callback_sync("acme.test", "")
        finally:
            reset_hermes_home_override(token)

    delayed_worker = threading.Thread(target=run_delayed)
    delayed_worker.start()
    assert client.post_started.wait(timeout=5)
    current[0] = False
    secure_input.clear_run(session_key, 231)
    delayed_worker.join(timeout=5)
    assert not delayed_worker.is_alive()
    assert delayed["answer"] == ""
    updates_before_late_post = len(client.updated)
    client.post_release.set()
    assert client.prompt_posted.wait(timeout=5)
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), gateway_loop).result(timeout=5)
    assert len(client.updated) == updates_before_late_post + 1
    assert terminal_homes and set(terminal_homes) == {str(homes[1])}
    client.post_release = None
    client.prompt_posted.clear()

    turn = _turn(adapter, gateway_loop, run_generation=240)
    monkeypatch.setattr(secure_input, "SECURE_INPUT_TIMEOUT_SECONDS", 1.0)

    cancelled: dict[str, str] = {}
    worker = threading.Thread(
        target=lambda: cancelled.setdefault("answer", turn._vault_code_callback_sync("acme.test", "")))
    worker.start()
    assert client.prompt_posted.wait(timeout=5)
    first_request_id = client.posted[-1]["blocks"][1]["elements"][0]["value"]
    asyncio.run_coroutine_threadsafe(
        _click(app, user_id="U_OWNER", request_id=first_request_id), gateway_loop).result(timeout=5)
    assert client.opened[-1]["view"]["notify_on_close"] is True
    asyncio.run_coroutine_threadsafe(
        _close(
            app,
            user_id="U_OWNER",
            request_id=first_request_id,
            team_shape="user_team",
        ),
        gateway_loop,
    ).result(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert cancelled["answer"] == ""
    assert secure_input.pending_count() == 0

    client.prompt_posted.clear()
    monkeypatch.setattr(secure_input, "SECURE_INPUT_TIMEOUT_SECONDS", 0.05)
    answer = turn._vault_code_callback_sync("acme.test", "")
    assert answer == ""
    assert secure_input.pending_count() == 0
    assert client.updated, "expired Slack prompt was not retired"

    timeout_request_id = client.posted[-1]["blocks"][1]["elements"][0]["value"]
    late_ack = asyncio.run_coroutine_threadsafe(
        _submit(app, user_id="U_OWNER", request_id=timeout_request_id, code="246810"), gateway_loop
    ).result(timeout=5)
    assert late_ack.call_args.kwargs.get("response_action") == "errors"
