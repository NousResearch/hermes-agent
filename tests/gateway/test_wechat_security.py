import json
import stat
import inspect
from types import SimpleNamespace

import pytest

from gateway.platforms.wechat import WeChatAdapter
from gateway.platforms.wechat import WeChatSessionState, WeChatSessionStore


class TestWeChatSessionSecurity:
    def test_save_uses_0600_and_strips_filekeys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wechat.get_hermes_home", lambda: tmp_path)
        store = WeChatSessionStore(profile="sec")
        store.save(
            WeChatSessionState(
                bearer_token="bearer",
                context_tokens={
                    "context": "ok",
                    "cdn_filekey": "should-not-persist",
                    "nested": {"fileKey": "drop-me", "keep": "value"},
                    "items": [{"filekey": "drop-me-too", "keep": 1}],
                },
                cursor="c1",
            )
        )

        persisted = json.loads(store.path.read_text(encoding="utf-8"))
        assert "cdn_filekey" not in persisted["context_tokens"]
        assert "fileKey" not in persisted["context_tokens"]["nested"]
        assert "filekey" not in persisted["context_tokens"]["items"][0]
        assert persisted["context_tokens"]["nested"]["keep"] == "value"
        assert persisted["context_tokens"]["items"][0]["keep"] == 1
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    def test_load_strips_filekeys_from_existing_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wechat.get_hermes_home", lambda: tmp_path)
        store = WeChatSessionStore(profile="sec")
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            json.dumps(
                {
                    "bearer_token": "bearer",
                    "cursor": "c2",
                    "context_tokens": {
                        "filekey": "stale-secret",
                        "nested": {"cdn_fileKey": "stale-secret-2", "ok": True},
                    },
                }
            ),
            encoding="utf-8",
        )

        state = store.load()
        assert "filekey" not in state.context_tokens
        assert "cdn_fileKey" not in state.context_tokens["nested"]
        assert state.context_tokens["nested"]["ok"] is True

    def test_load_reads_legacy_baseurl_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wechat.get_hermes_home", lambda: tmp_path)
        store = WeChatSessionStore(profile="sec")
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            json.dumps(
                {
                    "bearer_token": "bearer",
                    "baseurl": "https://ilinkai.weixin.qq.com",
                }
            ),
            encoding="utf-8",
        )

        state = store.load()
        assert state.base_url == "https://ilinkai.weixin.qq.com"


class TestWeChatAdapterSecurity:
    @pytest.mark.asyncio
    async def test_connect_rejects_non_localhost_ilink_url(self):
        adapter = object.__new__(WeChatAdapter)
        adapter.platform = SimpleNamespace(value="wechat")
        adapter._settings = SimpleNamespace(ilink_url="https://example.com:8787", ilink_token="token")
        adapter._session_state = SimpleNamespace(bearer_token=None)
        adapter._api_base_url = adapter._settings.ilink_url
        adapter.fatal = {}

        def _capture_fatal_error(code, message, *, retryable):
            adapter.fatal = {"code": code, "message": message, "retryable": retryable}

        adapter._set_fatal_error = _capture_fatal_error  # type: ignore[method-assign]

        ok = await adapter.connect()
        assert ok is False
        assert adapter.fatal["code"] == "wechat_non_local_url"

    @pytest.mark.asyncio
    async def test_connect_accepts_ilink_allowlist_hostname(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wechat.AIOHTTP_AVAILABLE", True)
        adapter = object.__new__(WeChatAdapter)
        adapter.platform = SimpleNamespace(value="wechat")
        adapter._settings = SimpleNamespace(ilink_url="https://ilinkai.weixin.qq.com", ilink_token="")
        adapter._session_state = SimpleNamespace(bearer_token=None)
        adapter._api_base_url = adapter._settings.ilink_url
        adapter.fatal = {}

        def _capture_fatal_error(code, message, *, retryable):
            adapter.fatal = {"code": code, "message": message, "retryable": retryable}

        adapter._set_fatal_error = _capture_fatal_error  # type: ignore[method-assign]

        ok = await adapter.connect()
        assert ok is False
        assert adapter.fatal["code"] == "wechat_missing_token"

    @pytest.mark.asyncio
    async def test_connect_rejects_ilink_allowlist_hostname_without_https(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.wechat.AIOHTTP_AVAILABLE", True)
        adapter = object.__new__(WeChatAdapter)
        adapter.platform = SimpleNamespace(value="wechat")
        adapter._settings = SimpleNamespace(ilink_url="http://ilinkai.weixin.qq.com", ilink_token="token")
        adapter._session_state = SimpleNamespace(bearer_token=None)
        adapter._api_base_url = adapter._settings.ilink_url
        adapter.fatal = {}

        def _capture_fatal_error(code, message, *, retryable):
            adapter.fatal = {"code": code, "message": message, "retryable": retryable}

        adapter._set_fatal_error = _capture_fatal_error  # type: ignore[method-assign]

        ok = await adapter.connect()
        assert ok is False
        assert adapter.fatal["code"] == "wechat_insecure_remote_url"

    def test_runtime_base_url_prefers_session_value(self):
        adapter = object.__new__(WeChatAdapter)
        adapter._session_state = SimpleNamespace(base_url="https://ilinkai.weixin.qq.com")
        adapter._settings = SimpleNamespace(ilink_url="http://127.0.0.1:8787")

        assert adapter._resolve_runtime_base_url() == "https://ilinkai.weixin.qq.com"

    def test_auth_headers_match_bind_compat_requirements(self):
        adapter = object.__new__(WeChatAdapter)
        adapter._settings = SimpleNamespace(ilink_token="")
        adapter._session_state = SimpleNamespace(bearer_token="bind-token")

        headers = adapter._auth_headers()

        assert headers["Authorization"] == "Bearer bind-token"
        assert headers["AuthorizationType"] == "ilink_bot_token"
        assert headers["iLink-App-Id"] == "openclaw-weixin"
        assert headers["iLink-App-ClientVersion"] == "hermes-gateway"
        assert headers["X-WECHAT-UIN"] == "aGVybWVz"
        assert headers["X-ILink-Token"] == "bind-token"

    def test_merge_session_tokens_persists_to_wechat_store_after_session_store_injection(self):
        adapter = object.__new__(WeChatAdapter)
        adapter._session_state = WeChatSessionState(
            bearer_token="old-token",
            context_tokens={},
            cursor=None,
            base_url="https://ilinkai.weixin.qq.com",
        )

        class _WeChatStore:
            def __init__(self):
                self.saved = []

            def save(self, state):
                self.saved.append(state)

        adapter._wechat_session_store = _WeChatStore()
        adapter._session_store = object()
        adapter._api_base_url = "https://ilinkai.weixin.qq.com"

        adapter._merge_session_tokens(
            {
                "cursor": "cursor-1",
                "bearer_token": "new-token",
                "context_tokens": {"chat": "ctx"},
                "baseurl": "https://ilinkai.weixin.qq.com",
            }
        )

        assert adapter._session_state.cursor == "cursor-1"
        assert adapter._session_state.bearer_token == "new-token"
        assert adapter._session_state.context_tokens == {"chat": "ctx"}
        assert len(adapter._wechat_session_store.saved) == 1


class TestWeChatGatewayRegistration:
    def test_wechat_in_adapter_factory(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._create_adapter)
        assert "Platform.WECHAT" in source

    def test_wechat_in_allowed_users_map(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        assert "WECHAT_ALLOWED_USERS" in source

    def test_wechat_in_allow_all_map(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        assert "WECHAT_ALLOW_ALL_USERS" in source
