"""Regression tests: recovery from stale lark-oapi tenant access tokens.

The lark-oapi token cache (``TokenManager.cache``) is a process-global singleton.
When its ``self_tenant_token:<app_id>`` entry goes stale, every SDK call fails with a
*response* carrying a token error code (not an exception), so neither the exception-based
send retry nor a client rebuild can recover — the gateway stayed broken until manually
restarted. ``_run_blocking`` now detects token error codes, evicts the cache entry, and
retries the single failed operation once.

Covers: PR #28608 (reviewer requirements — token codes, no duplicate sends across
chunks, sibling client operations, SDK-missing tolerance).
"""

import asyncio
import time
from types import SimpleNamespace as NS

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

FEISHU = load_plugin_adapter("feishu")


def _bare_adapter(app_id: str = "cli_test_app") -> FEISHU.FeishuAdapter:
    import threading

    adapter = object.__new__(FEISHU.FeishuAdapter)
    adapter._sdk_executor_lock = threading.Lock()
    adapter._sdk_executor = None
    adapter._sdk_executor_closing = False
    adapter._app_id = app_id
    return adapter


def _resp(code: int, msg: str = "", message_id: str = None):
    data = NS(message_id=message_id) if message_id is not None else None
    return NS(success=lambda: code == 0, code=code, msg=msg, data=data)


class _FlakyAPI:
    """Fails the first N calls with a token code, then succeeds."""

    def __init__(self, fail_times: int = 1, code: int = 99991664):
        self.fail_times = fail_times
        self.code = code
        self.calls = 0

    def create(self, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            return _resp(self.code, "access token invalid")
        return _resp(0, message_id="om_recovered")


class TestIsTokenError:
    @pytest.mark.parametrize("code", [99991400, 99991663, 99991664, 99991679])
    def test_token_codes_detected(self, code):
        assert FEISHU.FeishuAdapter._is_token_error(_resp(code))

    @pytest.mark.parametrize("code", [230001, 99992402, 230002, 11232])
    def test_non_token_codes_ignored(self, code):
        assert not FEISHU.FeishuAdapter._is_token_error(_resp(code))

    def test_success_response_not_token_error(self):
        assert not FEISHU.FeishuAdapter._is_token_error(_resp(0))

    def test_non_response_objects_ignored(self):
        assert not FEISHU.FeishuAdapter._is_token_error(None)
        assert not FEISHU.FeishuAdapter._is_token_error("not-a-response")
        assert not FEISHU.FeishuAdapter._is_token_error(NS(code=99991664))  # no .success()


class TestInvalidateTokenCache:
    def test_evicts_process_global_cache_entries(self, monkeypatch):
        monkeypatch.setattr(FEISHU, "_FEISHU_AVAILABLE", True, raising=False)
        from lark_oapi.core.token import TokenManager

        adapter = _bare_adapter("cli_evict")
        future = int(time.time() + 3600)
        TokenManager.cache.set("self_tenant_token:cli_evict", "stale", future)
        TokenManager.cache.set("self_app_token:cli_evict", "stale-app", future)
        assert TokenManager.cache.get("self_tenant_token:cli_evict") == "stale"

        adapter._invalidate_token_cache()
        # LocalCache.get drops entries whose expire < now, so eviction reads back None.
        assert TokenManager.cache.get("self_tenant_token:cli_evict") is None
        assert TokenManager.cache.get("self_app_token:cli_evict") is None

    def test_missing_sdk_is_silent(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_lark(name, *args, **kwargs):
            if name.startswith("lark_oapi"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_lark)
        adapter = _bare_adapter("cli_nosdk")
        adapter._invalidate_token_cache()  # must not raise


class TestRunBlockingRecovery:
    @pytest.mark.asyncio
    async def test_token_error_retried_once_and_succeeds(self):
        adapter = _bare_adapter()
        api = _FlakyAPI(fail_times=1)
        adapter._client = NS(im=NS(v1=NS(message=api)))
        response = await adapter._run_blocking(api.create, "req")
        assert api.calls == 2  # one initial + exactly one retry
        assert response.success()
        assert response.data.message_id == "om_recovered"

    @pytest.mark.asyncio
    async def test_persistent_token_failure_retried_exactly_once(self):
        adapter = _bare_adapter()
        api = _FlakyAPI(fail_times=10)
        adapter._client = NS(im=NS(v1=NS(message=api)))
        response = await adapter._run_blocking(api.create, "req")
        assert api.calls == 2  # no retry loop despite still failing
        assert not response.success()
        assert response.code == 99991664

    @pytest.mark.asyncio
    async def test_non_token_error_not_retried(self):
        adapter = _bare_adapter()
        api = _FlakyAPI(fail_times=1, code=230001)
        adapter._client = NS(im=NS(v1=NS(message=api)))
        response = await adapter._run_blocking(api.create, "req")
        assert api.calls == 1
        assert not response.success()

    @pytest.mark.asyncio
    async def test_sibling_operation_edit_message_recovered(self):
        """The same recovery covers edit_message(): the choke point wraps every op."""

        class _UpdateAPI:
            def __init__(self):
                self.calls = 0

            def update(self, request):
                self.calls += 1
                if self.calls == 1:
                    return _resp(99991400, "tenant access token expired")
                return _resp(0)

        adapter = _bare_adapter()
        api = _UpdateAPI()
        adapter._client = NS(im=NS(v1=NS(message=api)))

        # Minimal edit path mirroring production edit_message(): one SDK call
        # (im.v1.message.update) dispatched through the real _run_blocking.
        async def _edit():
            request = NS(message_id="om_1")
            response = await adapter._run_blocking(adapter._client.im.v1.message.update, request)
            return response

        result = await _edit()
        assert api.calls == 2  # initial token failure + one recovered retry
        assert result.success()

    @pytest.mark.asyncio
    async def test_multi_chunk_send_does_not_duplicate_accepted_chunks(self):
        """Reviewer scenario: chunk 1 accepted, chunk 2 token-fails. Only chunk 2's
        operation may retry — chunk 1 must not be re-sent (its message_id appears once)."""
        from gateway.config import PlatformConfig

        sent_ids = []
        chunk_results = [
            _resp(0, message_id="om_chunk1"),   # accepted
            _resp(99991664),                    # token failure
            _resp(0, message_id="om_chunk2"),   # accepted after eviction+retry
        ]
        calls = {"n": 0}

        class _MessageAPI:
            def create(self, request):
                idx = min(calls["n"], len(chunk_results) - 1)
                resp = chunk_results[idx]
                calls["n"] += 1
                if resp.success():
                    sent_ids.append(resp.data.message_id)
                return resp

        adapter = _bare_adapter()
        adapter._client = NS(im=NS(v1=NS(message=_MessageAPI())))

        # Two chunks; each chunk = one _run_blocking(message.create).
        chunk1 = await adapter._run_blocking(adapter._client.im.v1.message.create, "chunk-1")
        chunk2 = await adapter._run_blocking(adapter._client.im.v1.message.create, "chunk-2")
        assert chunk1.success() and chunk2.success()
        assert calls["n"] == 3  # chunk1 (1) + chunk2 fail (1) + chunk2 retry (1)
        assert sent_ids == ["om_chunk1", "om_chunk2"]  # chunk1 sent exactly once
