"""Contracts for plugin-owned Feishu card actions and callback attestation.

Plugin-owned actions are a trusted transport edge. They must be dispatched to
one exact plugin handler, attested by the Feishu adapter, and never converted
into an LLM-visible synthetic message.
"""

from __future__ import annotations

import importlib.util
import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks() -> None:
    if "lark_oapi" not in sys.modules and importlib.util.find_spec("lark_oapi") is None:
        mod = MagicMock()
        for name in (
            "lark_oapi",
            "lark_oapi.api.im.v1",
            "lark_oapi.event",
            "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if "aiohttp" not in sys.modules and importlib.util.find_spec("aiohttp") is None:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig  # noqa: E402
from hermes_constants import (  # noqa: E402
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest  # noqa: E402
import plugins.platforms.feishu.adapter as feishu_module  # noqa: E402
from plugins.platforms.feishu.adapter import FeishuAdapter  # noqa: E402


PLUGIN_ACTION = "popaskin_ads.pending_write"
TEST_ENCRYPT_KEY = "encrypt-key-test"


def _signed_webhook_headers(body: bytes) -> dict[str, str]:
    timestamp = "1720000000"
    nonce = "nonce-test"
    content = f"{timestamp}{nonce}{TEST_ENCRYPT_KEY}{body.decode('utf-8')}"
    return {
        "Content-Type": "application/json",
        "x-lark-request-timestamp": timestamp,
        "x-lark-request-nonce": nonce,
        "x-lark-signature": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


class _FakeToast:
    def __init__(self):
        self.type = None
        self.content = None


class _FakeResponse:
    def __init__(self):
        self.card = None
        self.toast = None


class _FakeRequestContent:
    def __init__(self, body: bytes):
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


@pytest.fixture(autouse=True)
def _callback_types(monkeypatch):
    monkeypatch.setattr(feishu_module, "CallBackToast", _FakeToast, raising=False)
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeResponse)


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    return manager, PluginContext(manifest=manifest, manager=manager)


def _make_adapter(*, loop_ready: bool = False) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._app_id = "cli_test"
    adapter._app_secret = "app-secret-test"
    adapter._verification_token = "verify-token-test"
    adapter._encrypt_key = TEST_ENCRYPT_KEY
    if loop_ready:
        adapter._loop = MagicMock()
        adapter._loop.is_closed.return_value = False
    else:
        adapter._loop = None
    return adapter


def _make_card_action_data(
    *,
    route: str = PLUGIN_ACTION,
    event_id: str = "evt_1",
    event_token: str = "card-token-1",
    message_id: str = "om_card",
    app_id: str = "cli_test",
    extra_value: dict | None = None,
) -> SimpleNamespace:
    action_value = {
        "hermes_plugin_action": route,
        "action_id": "pa_public_123",
        "decision": "approve",
    }
    if extra_value:
        action_value.update(extra_value)
    return SimpleNamespace(
        header=SimpleNamespace(
            event_id=event_id,
            create_time="1720000000000000",
            event_type="card.action.trigger",
            tenant_key="tenant_1",
            app_id=app_id,
        ),
        event=SimpleNamespace(
            token=event_token,
            host="im_message",
            operator=SimpleNamespace(
                open_id="ou_owner",
                user_id="u_owner",
                union_id="on_owner",
                tenant_key="tenant_1",
            ),
            context=SimpleNamespace(
                open_message_id=message_id,
                open_chat_id="oc_dm",
            ),
            action=SimpleNamespace(tag="button", value=action_value),
        ),
    )


def _make_webhook_request(
    *,
    event_id: str,
    route: str | None = PLUGIN_ACTION,
) -> SimpleNamespace:
    action_value = {
        "action_id": "pa_public_123",
        "decision": "approve",
    }
    if route is not None:
        action_value["hermes_plugin_action"] = route
    payload = {
        "header": {
            "event_id": event_id,
            "create_time": "1720000000000000",
            "event_type": "card.action.trigger",
            "tenant_key": "tenant_1",
            "app_id": "cli_test",
            "token": "verify-token-test",
        },
        "event": {
            "token": f"card-token-{event_id}",
            "host": "im_message",
            "operator": {
                "open_id": "ou_owner",
                "user_id": "u_owner",
                "union_id": "on_owner",
                "tenant_key": "tenant_1",
            },
            "context": {
                "open_message_id": f"om_{event_id}",
                "open_chat_id": "oc_dm",
            },
            "action": {
                "tag": "button",
                "value": action_value,
            },
        },
    }
    body = json.dumps(payload).encode("utf-8")
    return SimpleNamespace(
        remote="127.0.0.1",
        content_length=None,
        headers=_signed_webhook_headers(body),
        content=_FakeRequestContent(body),
    )


class TestRegisterFeishuCardActionHandlerAPI:
    def test_exact_action_id_is_registered_and_disposable(self):
        manager, ctx = _make_ctx()
        callback = MagicMock()

        registration = ctx.register_feishu_card_action_handler(PLUGIN_ACTION, callback)

        assert manager.get_feishu_card_action_handlers() == [
            (PLUGIN_ACTION, callback, "test_plugin")
        ]
        assert registration.active is True

        registration.dispose()

        assert manager.get_feishu_card_action_handlers() == []
        assert registration.active is False

    def test_accessor_returns_a_copy(self):
        manager, ctx = _make_ctx()
        ctx.register_feishu_card_action_handler(PLUGIN_ACTION, MagicMock())

        handlers = manager.get_feishu_card_action_handlers()
        handlers.clear()

        assert len(manager.get_feishu_card_action_handlers()) == 1

    @pytest.mark.parametrize(
        "action_id",
        [None, "", "   ", 123, "../escape", "a" * 129],
    )
    def test_invalid_action_id_is_rejected(self, action_id):
        _manager, ctx = _make_ctx()
        with pytest.raises(ValueError, match="action_id"):
            ctx.register_feishu_card_action_handler(action_id, MagicMock())

    def test_non_callable_callback_is_rejected(self):
        _manager, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_feishu_card_action_handler(PLUGIN_ACTION, "not callable")

    def test_async_callback_is_rejected(self):
        _manager, ctx = _make_ctx()

        async def callback(**_kwargs):
            return {"ok": True}

        with pytest.raises(ValueError, match="synchronous"):
            ctx.register_feishu_card_action_handler(PLUGIN_ACTION, callback)

    def test_duplicate_action_id_is_rejected(self):
        manager, first = _make_ctx("first")
        second = PluginContext(
            PluginManifest(name="second", version="0.1.0", description="test"),
            manager,
        )
        first.register_feishu_card_action_handler(PLUGIN_ACTION, MagicMock())

        with pytest.raises(ValueError, match="already registered"):
            second.register_feishu_card_action_handler(PLUGIN_ACTION, MagicMock())


class TestFeishuPluginCardActionDispatch:
    def test_webhook_card_action_rejects_missing_auth_configuration(self):
        adapter = _make_adapter(loop_ready=False)
        adapter._verification_token = ""
        adapter._encrypt_key = ""

        response = asyncio.run(
            adapter._handle_webhook_request(_make_webhook_request(event_id="evt_no_auth"))
        )

        assert response.status == 503
        assert response.text == "Webhook authentication unavailable"

    def test_webhook_card_action_rejects_verification_token_only(self):
        adapter = _make_adapter(loop_ready=False)
        adapter._encrypt_key = ""

        response = asyncio.run(
            adapter._handle_webhook_request(
                _make_webhook_request(event_id="evt_token_only")
            )
        )

        assert response.status == 503
        assert response.text == "Card-action webhook authentication unavailable"

    @staticmethod
    def _manager_with(callback=None, route: str = PLUGIN_ACTION):
        manager = MagicMock()
        handlers = [] if callback is None else [(route, callback, "popaskin_ads")]
        manager.get_feishu_card_action_handlers.return_value = handlers
        return manager

    def test_matching_action_is_attested_without_adapter_loop_or_llm_dispatch(self):
        adapter = _make_adapter(loop_ready=False)
        callback = MagicMock(
            return_value={"ok": True, "toast": {"type": "success", "content": "已接收"}}
        )
        data = _make_card_action_data()

        with (
            patch(
                "hermes_cli.plugins.get_plugin_manager",
                return_value=self._manager_with(callback),
            ),
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            response = adapter._on_card_action_trigger(data)

        submit.assert_not_called()
        callback.assert_called_once()
        kwargs = callback.call_args.kwargs
        expected_body = (
            b'{"action":{"tag":"button","value":{"action_id":"pa_public_123",'
            b'"decision":"approve","hermes_plugin_action":"popaskin_ads.pending_write"}},'
            b'"context":{"open_chat_id":"oc_dm","open_message_id":"om_card"},'
            b'"header":{"app_id":"cli_test","create_time":"1720000000000000",'
            b'"event_id":"evt_1","event_type":"card.action.trigger",'
            b'"tenant_key":"tenant_1"},"host":"im_message",'
            b'"operator":{"open_id":"ou_owner","tenant_key":"tenant_1",'
            b'"union_id":"on_owner","user_id":"u_owner"},'
            b'"version":"hermes-feishu-card-action-v1"}'
        )
        assert kwargs["body"] == expected_body
        assert kwargs["signature"] == (
            "v1=5ddca6c9b5265a716642c4fbb0a7a0dd2761ba70c715d5c1fc390e59767751eb"
        )
        assert kwargs["envelope"] == json.loads(expected_body)
        assert response.toast.type == "success"
        assert response.toast.content == "已接收"

    def test_webhook_namespace_action_value_uses_trusted_plugin_path(self):
        adapter = _make_adapter(loop_ready=False)
        callback = MagicMock(return_value={"ok": True})
        data = _make_card_action_data()
        data.event.action.value["metadata"] = SimpleNamespace(
            source="feishu",
            nested=SimpleNamespace(attempt=1),
        )
        data.event.action.value = SimpleNamespace(**data.event.action.value)

        with (
            patch(
                "hermes_cli.plugins.get_plugin_manager",
                return_value=self._manager_with(callback),
            ),
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            response = adapter._on_card_action_trigger(data)

        callback.assert_called_once()
        submit.assert_not_called()
        assert callback.call_args.kwargs["envelope"]["action"]["value"]["metadata"] == {
            "source": "feishu",
            "nested": {"attempt": 1},
        }
        assert response.toast.type == "success"

    def test_webhook_returns_plugin_error_toast_instead_of_generic_ack(self):
        adapter = _make_adapter(loop_ready=False)
        callback_threads = []

        def _reject_callback(**_kwargs):
            callback_threads.append(threading.get_ident())
            return {"ok": False}

        callback = MagicMock(side_effect=_reject_callback)
        payload = {
            "header": {
                "event_id": "evt_webhook",
                "create_time": "1720000000000000",
                "event_type": "card.action.trigger",
                "tenant_key": "tenant_1",
                "app_id": "cli_test",
                "token": "verify-token-test",
            },
            "event": {
                "token": "card-token-webhook",
                "host": "im_message",
                "operator": {
                    "open_id": "ou_owner",
                    "user_id": "u_owner",
                    "union_id": "on_owner",
                    "tenant_key": "tenant_1",
                },
                "context": {
                    "open_message_id": "om_card",
                    "open_chat_id": "oc_dm",
                },
                "action": {
                    "tag": "button",
                    "value": {
                        "hermes_plugin_action": PLUGIN_ACTION,
                        "action_id": "pa_public_123",
                        "decision": "approve",
                    },
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            headers=_signed_webhook_headers(body),
            content=_FakeRequestContent(body),
        )

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._manager_with(callback),
        ):
            response = asyncio.run(adapter._handle_webhook_request(request))

        assert response.status == 503
        assert json.loads(response.text) == {
            "toast": {"type": "error", "content": "请求未确认，请重试"}
        }
        assert callback_threads and callback_threads != [threading.get_ident()]

    def test_webhook_success_is_acknowledged_only_after_plugin_acceptance(self):
        adapter = _make_adapter(loop_ready=False)
        callback = MagicMock(return_value={"ok": True})

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._manager_with(callback),
        ):
            response = asyncio.run(
                adapter._handle_webhook_request(
                    _make_webhook_request(event_id="evt_durable_accept")
                )
            )

        assert response.status == 200
        assert json.loads(response.text)["toast"]["type"] == "success"

    def test_webhook_non_plugin_action_fails_closed_before_memory_dispatch(self):
        adapter = _make_adapter(loop_ready=False)

        with patch.object(adapter, "_on_card_action_trigger") as legacy_dispatch:
            response = asyncio.run(
                adapter._handle_webhook_request(
                    _make_webhook_request(
                        event_id="evt_legacy_action",
                        route=None,
                    )
                )
            )

        legacy_dispatch.assert_not_called()
        assert response.status == 503
        assert json.loads(response.text) == {
            "toast": {
                "type": "error",
                "content": "Webhook 模式暂不支持此卡片操作",
            }
        }

    def test_webhook_timeout_and_busy_are_bounded_and_retryable(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter(loop_ready=False)
        started = threading.Event()
        release = threading.Event()
        callback_calls = []

        def _blocking_callback(**_kwargs):
            callback_calls.append(threading.get_ident())
            started.set()
            release.wait(1.0)
            return {"ok": True}

        manager = self._manager_with(_blocking_callback)
        monkeypatch.setattr(
            feishu_module,
            "_FEISHU_WEBHOOK_CARD_ACTION_TIMEOUT_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            feishu_module,
            "_FEISHU_WEBHOOK_CARD_ACTION_SLOTS",
            threading.BoundedSemaphore(1),
            raising=False,
        )

        timer = threading.Timer(0.15, release.set)
        timer.daemon = True
        timer.start()

        async def _run_two_requests():
            first = await adapter._handle_webhook_request(
                _make_webhook_request(event_id="evt_timeout_1")
            )
            assert started.is_set()
            second = await adapter._handle_webhook_request(
                _make_webhook_request(event_id="evt_timeout_2")
            )
            return first, second

        try:
            with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
                first, second = asyncio.run(_run_two_requests())
        finally:
            release.set()
            timer.cancel()

        assert len(callback_calls) == 1
        for response in (first, second):
            assert response.status == 503
            payload = json.loads(response.text)
            assert payload["toast"]["type"] == "warning"
            assert "未知" in payload["toast"]["content"]

    def test_webhook_serializer_preserves_resolved_card_and_drops_null_sdk_fields(self):
        response = _FakeResponse()
        response.toast = _FakeToast()
        response.toast.type = "success"
        response.toast.content = "已处理"
        response.card = SimpleNamespace(
            type="raw",
            data={"elements": [{"tag": "div", "text": "done"}]},
            optional=None,
        )

        assert FeishuAdapter._webhook_card_action_payload(response) == {
            "toast": {"type": "success", "content": "已处理"},
            "card": {
                "type": "raw",
                "data": {"elements": [{"tag": "div", "text": "done"}]},
            },
        }

    def test_handler_resolution_uses_adapter_profile_home(self, tmp_path):
        adapter_home = tmp_path / "profile-a"
        ambient_home = tmp_path / "profile-b"
        adapter_token = set_hermes_home_override(str(adapter_home))
        try:
            adapter = _make_adapter(loop_ready=False)
        finally:
            reset_hermes_home_override(adapter_token)

        callback = MagicMock(return_value={"ok": True})
        manager = self._manager_with(callback)
        observed_homes = []

        def _manager_for_current_home():
            observed_homes.append(get_hermes_home().resolve())
            return manager

        ambient_token = set_hermes_home_override(str(ambient_home))
        try:
            with patch(
                "hermes_cli.plugins.get_plugin_manager",
                side_effect=_manager_for_current_home,
            ):
                response = adapter._on_card_action_trigger(_make_card_action_data())
        finally:
            reset_hermes_home_override(ambient_token)

        callback.assert_called_once()
        assert response.toast.type == "success"
        assert observed_homes == [adapter_home.resolve()]

    def test_reserved_plugin_key_takes_precedence_over_builtin_action(self):
        adapter = _make_adapter(loop_ready=True)
        callback = MagicMock(return_value={"ok": True})
        data = _make_card_action_data(extra_value={"hermes_action": "approve_once"})

        with (
            patch(
                "hermes_cli.plugins.get_plugin_manager",
                return_value=self._manager_with(callback),
            ),
            patch.object(adapter, "_handle_approval_card_action") as builtin,
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            adapter._on_card_action_trigger(data)

        callback.assert_called_once()
        builtin.assert_not_called()
        submit.assert_not_called()

    def test_unknown_plugin_action_fails_closed_without_synthetic_dispatch(self):
        adapter = _make_adapter()

        with (
            patch(
                "hermes_cli.plugins.get_plugin_manager",
                return_value=self._manager_with(),
            ),
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            response = adapter._on_card_action_trigger(_make_card_action_data())

        submit.assert_not_called()
        assert response.toast.type == "error"

    @pytest.mark.parametrize(
        "mutation",
        [
            {"message_id": ""},
            {"app_id": "cli_other"},
        ],
    )
    def test_unattestable_callback_fails_closed(self, mutation):
        adapter = _make_adapter()
        callback = MagicMock()
        data = _make_card_action_data(**mutation)

        with (
            patch(
                "hermes_cli.plugins.get_plugin_manager",
                return_value=self._manager_with(callback),
            ),
            patch.object(adapter, "_submit_on_loop") as submit,
        ):
            response = adapter._on_card_action_trigger(data)

        submit.assert_not_called()
        callback.assert_not_called()
        assert response.toast.type == "error"

    def test_missing_app_secret_fails_closed(self):
        adapter = _make_adapter()
        adapter._app_secret = ""
        callback = MagicMock()

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._manager_with(callback),
        ):
            response = adapter._on_card_action_trigger(_make_card_action_data())

        callback.assert_not_called()
        assert response.toast.type == "error"

    def test_duplicate_callback_is_forwarded_for_durable_downstream_idempotency(self):
        adapter = _make_adapter()
        callback = MagicMock(return_value={"ok": True})
        manager = self._manager_with(callback)
        data = _make_card_action_data()

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            first = adapter._on_card_action_trigger(data)
            second = adapter._on_card_action_trigger(data)

        assert callback.call_count == 2
        assert first.toast.type == "success"
        assert second.toast.type == "success"

    def test_failed_forward_does_not_poison_same_event_retry(self):
        adapter = _make_adapter()
        callback = MagicMock(side_effect=[{"ok": False}, {"ok": True}])
        manager = self._manager_with(callback)
        data = _make_card_action_data()

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            first = adapter._on_card_action_trigger(data)
            second = adapter._on_card_action_trigger(data)

        assert callback.call_count == 2
        assert first.toast.type == "error"
        assert second.toast.type == "success"

    def test_plugin_exception_fails_closed_without_logging_exception_text(self, caplog):
        adapter = _make_adapter()
        callback = MagicMock(side_effect=RuntimeError("SENSITIVE_CALLBACK_TEXT"))

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._manager_with(callback),
        ):
            response = adapter._on_card_action_trigger(_make_card_action_data())

        assert response.toast.type == "error"
        assert "RuntimeError" in caplog.text
        assert "SENSITIVE_CALLBACK_TEXT" not in caplog.text

    def test_non_acknowledging_plugin_result_fails_closed(self):
        adapter = _make_adapter()
        callback = MagicMock(return_value=None)

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            return_value=self._manager_with(callback),
        ):
            response = adapter._on_card_action_trigger(_make_card_action_data())

        callback.assert_called_once()
        assert response.toast.type == "error"

    def test_non_plugin_card_action_keeps_existing_synthetic_path(self):
        adapter = _make_adapter(loop_ready=True)
        data = _make_card_action_data()
        data.event.action.value = {"ordinary": "value"}

        def close_coro(_loop, coro):
            coro.close()
            return True

        with patch.object(adapter, "_submit_on_loop", side_effect=close_coro) as submit:
            adapter._on_card_action_trigger(data)

        submit.assert_called_once()

    def test_ordinary_card_dedup_uses_adapter_lock(self):
        adapter = _make_adapter()
        guard = MagicMock()
        adapter._card_action_token_lock = guard

        assert adapter._is_card_action_duplicate("token") is False

        guard.__enter__.assert_called_once_with()
        guard.__exit__.assert_called_once()
