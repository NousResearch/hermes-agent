"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import io
import json
import threading
import time
import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Fake Mem0Backend for provider-level tests."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or []
        self._all_results = all_results or {"results": [], "count": 0}
        self.captured = []

    def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank}))
        return self._search_results

    def get_all(self, *, filters, page=1, page_size=100):
        self.captured.append(("get_all", {"filters": filters, "page": page, "page_size": page_size}))
        return self._all_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append((
            "add",
            messages,
            {"user_id": user_id, "agent_id": agent_id, "infer": infer, "metadata": metadata},
        ))
        return {"status": "PENDING", "event_id": "evt-test-123"}

    def update(self, memory_id, text):
        self.captured.append(("update", memory_id, text))
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self.captured.append(("delete", memory_id))
        return {"result": "Memory deleted.", "memory_id": memory_id}


class TestMem0V3Tools:
    """Test v3 tool names and response handling."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_search_returns_ids(self, monkeypatch):
        backend = FakeBackend(search_results=[{"id": "mem-1", "memory": "foo", "score": 0.9}])
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        assert result["results"][0]["id"] == "mem-1"


    def test_add_uses_content_param(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["infer"] is False
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert "event_id" in result


    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0UpdateDelete:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "mem-1", "text": "updated fact"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert backend.captured[0][2] == "updated fact"
        assert result["result"] == "Memory updated."
        assert result["memory_id"] == "mem-1"


    def test_delete_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert result["result"] == "Memory deleted."


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider


class TestMem0V3Internal:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_sync_turn_explicit_kwargs(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert call[2]["infer"] is True


class TestSyncTurnTruncation:
    """sync_turn must cap messages before ingestion so small-context embedding
    backends (OSS Ollama bge-small-zh-v1.5: 512 tokens; jina-embeddings-v3 token
    limits) don't fail the whole extraction — a failure _try only logs."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._config = {"mode": "platform", "oss": {}}  # pin config so a local mem0.json can't skew caps
        provider._mode = "platform"
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_short_messages_pass_through_unchanged(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert backend.captured[0][1] == [
            {"role": "user", "content": "user said"},
            {"role": "assistant", "content": "assistant replied"},
        ]

    def test_oversized_messages_truncated_at_sentence_boundary(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        max_len = mem0_plugin._SYNC_MSG_MAX_CHARS
        long_user = ("Important fact about the project. " * 40).strip()
        long_assistant = "这是很长的一段回答。" * 80
        provider.sync_turn(long_user, long_assistant, session_id="s1")
        provider._sync_thread.join(timeout=2)
        sent = backend.captured[0][1]
        assert len(sent[0]["content"]) <= max_len
        assert sent[0]["content"].endswith(".")
        assert len(sent[1]["content"]) <= max_len
        assert sent[1]["content"].endswith("。")

    def test_small_context_backend_never_sees_oversized_input(self, monkeypatch):
        """Regression for #106235/#37421: an OSS embedding backend with a small
        context window raises on oversized input; truncation up front keeps the
        extraction from being silently dropped (no breaker failures)."""

        class SmallContextBackend(FakeBackend):
            def add(self, messages, **kwargs):
                if any(len(m["content"]) > mem0_plugin._SYNC_MSG_MAX_CHARS for m in messages):
                    raise RuntimeError("HTTP 500: embedding input exceeds model context")
                return super().add(messages, **kwargs)

        backend = SmallContextBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("x" * 5000, "y" * 5000, session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(backend.captured) == 1
        assert all(len(m["content"]) <= mem0_plugin._SYNC_MSG_MAX_CHARS for m in backend.captured[0][1])
        assert provider._consecutive_failures == 0

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("short text", "short text"),  # under the cap: untouched
            ("这是很短的句子。", "这是很短的句子。"),  # CJK under the cap: untouched
            # no sentence boundary anywhere: hard cut at the cap
            ("x" * 600, "x" * mem0_plugin._SYNC_MSG_MAX_CHARS),
        ],
    )
    def test_truncate_for_sync_pass_through_and_hard_cut(self, text, expected):
        assert mem0_plugin._truncate_for_sync(text) == expected

    def test_truncate_for_sync_keeps_last_complete_sentence(self):
        max_len = mem0_plugin._SYNC_MSG_MAX_CHARS
        text = "".join(f"Sentence {i}. " for i in range(100))
        out = mem0_plugin._truncate_for_sync(text)
        assert len(out) <= max_len
        assert out.endswith(".")
        assert out.startswith("Sentence 0. ")
        # boundary only inside the first third of the window: hard cut instead
        assert mem0_plugin._truncate_for_sync("One. " + "x" * 600) == ("One. " + "x" * 600)[:max_len]


class TestSyncCapResolution:
    """The cap is model-aware: explicit config → Ollama probe → known-model table
    → conservative default, resolved once and cached (the probe is a network call)."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._config = {"mode": "platform", "oss": {}}
        provider._mode = "platform"
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def _oss(self, provider, embedder_cfg):
        provider._mode = "oss"
        provider._config["oss"] = {"embedder": {"config": embedder_cfg}}

    def test_explicit_config_overrides_everything(self, monkeypatch):
        provider = self._make_provider(monkeypatch, FakeBackend())
        self._oss(provider, {"model": "jina-embeddings-v3"})
        provider._config["sync_max_chars"] = 100
        assert provider._sync_max_chars() == 100

    def test_invalid_config_value_falls_through(self, monkeypatch):
        provider = self._make_provider(monkeypatch, FakeBackend())
        provider._config["sync_max_chars"] = "not-a-number"
        assert provider._sync_max_chars() == mem0_plugin._SYNC_MSG_MAX_CHARS

    def test_oss_probe_large_context_lifts_cap(self, monkeypatch):
        provider = self._make_provider(monkeypatch, FakeBackend())
        self._oss(provider, {"model": "bge-m3:latest", "ollama_base_url": "http://127.0.0.1:11434"})
        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", lambda b, m: 8192)
        assert provider._sync_max_chars() == mem0_plugin._chars_for_context(8192)
        assert provider._sync_max_chars() > 6900  # 8192-token models must not be stuck at the 450 default

    def test_oss_probe_small_context_keeps_default(self, monkeypatch):
        # szicely's measurement on bge-small-zh-v1.5:f16 (512 tokens): 450 chars OK, 600 → HTTP 500
        provider = self._make_provider(monkeypatch, FakeBackend())
        self._oss(provider, {"model": "bge-small-zh-v1.5:f16"})
        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", lambda b, m: 512)
        assert provider._sync_max_chars() == mem0_plugin._SYNC_MSG_MAX_CHARS

    def test_known_model_table_when_probe_fails(self, monkeypatch):
        provider = self._make_provider(monkeypatch, FakeBackend())
        self._oss(provider, {"model": "jina-embeddings-v3:latest"})
        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", lambda b, m: None)
        assert provider._sync_max_chars() == mem0_plugin._chars_for_context(8192)

    def test_unknown_model_falls_back_to_default(self, monkeypatch):
        provider = self._make_provider(monkeypatch, FakeBackend())
        self._oss(provider, {"model": "custom-finetune:q4"})
        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", lambda b, m: None)
        assert provider._sync_max_chars() == mem0_plugin._SYNC_MSG_MAX_CHARS

    def test_platform_mode_never_probes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", lambda b, m: calls.append((b, m)) or 8192)
        provider = self._make_provider(monkeypatch, FakeBackend())
        assert provider._sync_max_chars() == mem0_plugin._SYNC_MSG_MAX_CHARS
        assert calls == []

    def test_probe_resolved_once_and_cached_across_syncs(self, monkeypatch):
        calls = []

        def fake_probe(base_url, model):
            calls.append((base_url, model))
            return 8192

        monkeypatch.setattr(mem0_plugin, "_probe_ollama_context", fake_probe)
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        self._oss(provider, {"model": "bge-m3", "ollama_base_url": "http://127.0.0.1:11434"})
        provider.sync_turn("x" * 9000, "y" * 9000, session_id="s1")
        provider._sync_thread.join(timeout=2)
        provider.sync_turn("x" * 9000, "y" * 9000, session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(calls) == 1  # cached: the second sync must not re-probe
        cap = mem0_plugin._chars_for_context(8192)
        assert all(len(m["content"]) <= cap for m in backend.captured[0][1])

    def test_sync_turn_uses_resolved_cap(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider._config["sync_max_chars"] = 120
        provider.sync_turn("Sentence one. " * 30, "ok", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(backend.captured[0][1][0]["content"]) <= 120


class TestProbeOllamaContext:
    """The probe reads the context window from Ollama's /api/show payload and
    fails soft on any error (it runs inside the sync path)."""

    def _fake_urlopen(self, monkeypatch, payload, sent=None):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_open(req, timeout=5):
            if sent is not None:
                sent.append((req.full_url, json.loads(req.data.decode())))
            return _Resp(json.dumps(payload).encode())

        monkeypatch.setattr(mem0_plugin.urllib.request, "urlopen", fake_open)

    def test_reads_bert_context_length(self, monkeypatch):
        sent = []
        self._fake_urlopen(monkeypatch, {"model_info": {"general.architecture": "bert", "bert.context_length": 512}}, sent)
        assert mem0_plugin._probe_ollama_context("http://localhost:11434/", "bge-small-zh-v1.5:f16") == 512
        url, body = sent[0]
        assert url == "http://localhost:11434/api/show"  # trailing slash normalized
        assert body == {"model": "bge-small-zh-v1.5:f16"}

    def test_reads_arch_prefixed_and_bare_context_length(self, monkeypatch):
        self._fake_urlopen(monkeypatch, {"model_info": {"general.architecture": "jina", "jina.context_length": 8192}})
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "jina-embeddings-v3") == 8192
        self._fake_urlopen(monkeypatch, {"model_info": {"context_length": 4096}})
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "weird-model") == 4096

    def test_rejects_non_http_schemes_and_empty_inputs(self, monkeypatch):
        opened = []
        monkeypatch.setattr(mem0_plugin.urllib.request, "urlopen", lambda req, timeout=5: opened.append(req))
        assert mem0_plugin._probe_ollama_context("ftp://host", "m") is None
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "") is None
        assert mem0_plugin._probe_ollama_context("", "m") is None
        assert opened == []  # the scheme/netloc guard fires before any request

    def test_returns_none_on_errors_and_garbage_payloads(self, monkeypatch):
        def boom(req, timeout=5):
            raise OSError("connection refused")

        monkeypatch.setattr(mem0_plugin.urllib.request, "urlopen", boom)
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "m") is None
        self._fake_urlopen(monkeypatch, {"model_info": {"general.architecture": "bert", "bert.context_length": "bogus"}})
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "m") is None
        self._fake_urlopen(monkeypatch, {})  # no model_info at all
        assert mem0_plugin._probe_ollama_context("http://127.0.0.1:11434", "m") is None


class TestMem0Prefetch:
    """prefetch() must recall on the CURRENT question, synchronously.

    The old implementation ignored its ``query`` and returned whatever a
    background ``queue_prefetch`` had warmed from the PREVIOUS turn — so the
    first turn injected nothing and later turns injected stale, off-topic
    memories. These lock the corrected behaviour.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_prefetch_searches_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "user prefers dark mode"}])
        provider = self._make_provider(backend)
        result = provider.prefetch("what theme do I like?")
        kind, query, opts = backend.captured[0]
        assert kind == "search"
        assert query == "what theme do I like?"
        assert opts["filters"] == {"user_id": "u123"}
        assert opts["top_k"] == 10
        assert opts["rerank"] is False
        assert "## Mem0 Memory" in result
        assert "user prefers dark mode" in result


    def test_on_turn_start_queues_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        provider = self._make_provider(backend)
        provider.on_turn_start(1, "where do I live?")
        provider._prefetch_thread.join(timeout=1)
        result = provider.prefetch("where do I live?")
        assert "lives in Berlin" in result
        assert len([c for c in backend.captured if c[0] == "search"]) == 1

    def test_slow_prefetch_returns_quickly(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()
        search_returned = threading.Event()

        class SlowBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                entered.set()
                try:
                    release.wait(30)
                    return super().search(
                        query, filters=filters, top_k=top_k, rerank=rerank
                    )
                finally:
                    search_returned.set()

        monkeypatch.setattr(mem0_plugin, "_PREFETCH_WAIT_SECS", 0.01)
        provider = self._make_provider(
            SlowBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        )
        # DETERMINISTIC non-blocking witness — replaces `assert elapsed < 0.1`.
        #
        # The old form slept 0.2s in the backend and asserted prefetch returned
        # in under 0.1s. That makes the OS scheduler part of the assertion: on
        # a loaded box thread startup alone can eat the 100ms budget, so the
        # inequality flips with nothing wrong in the code under test. Observed
        # failing in a full-directory run of tests/plugins/memory.
        #
        # The real contract is that prefetch gives up on the slow backend
        # instead of waiting for it. Assert it directly: the backend search is
        # STILL PARKED (release unset, so `search_returned` cannot be set). If
        # prefetch ever waited for the backend, the search would have returned
        # first and this fails. No wall-clock constant.
        assert provider.prefetch("where do I live?") == ""
        assert entered.wait(30), "prefetch never reached the backend"
        assert not search_returned.is_set(), (
            "prefetch blocked on the slow backend: the backend search had "
            "already returned by the time prefetch did"
        )

        release.set()
        provider._prefetch_thread.join(timeout=30)
        assert "lives in Berlin" in provider.prefetch("where do I live?")


    def test_queue_prefetch_fires_no_search(self):
        # prefetch is synchronous now, so the post-turn warm is redundant and
        # must not fire a wasted backend search.
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "x"}])
        provider = self._make_provider(backend)
        provider.queue_prefetch("previous turn text")
        assert backend.captured == []


class TestMem0V3Config:

    def test_tool_schemas_four_tools(self):
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["mem0_search", "mem0_add", "mem0_update", "mem0_delete"]

    def test_system_prompt_new_tool_names(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_add" in block
        assert "mem0_update" in block
        assert "mem0_delete" in block
        assert "mem0_list" not in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block


class TestMem0ModeSwitch:

    def test_default_mode_is_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"

    def test_missing_mode_key_defaults_platform(self, monkeypatch, tmp_path):
        """Backward compat: old mem0.json without mode key works."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"user_id": "old-user"}')
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"
        assert provider._user_id == "old-user"

    def test_is_available_platform_needs_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False


class TestMem0UserIdResolution:
    """user_id resolution: configured override > gateway-native id > placeholder.

    Same human across CLI / Telegram / Discord / Slack / etc. should map to
    the same memory store when MEM0_USER_ID is set, and only fall back to the
    gateway-native id when it isn't.
    """

    def _provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        # Skip backend instantiation — we only care about identity resolution.
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        return provider

    def test_env_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_USER_ID", "ryan@example.com")
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_file_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "ryan@example.com"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_unset_falls_back_to_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


    def test_legacy_placeholder_in_config_does_not_override_kwargs(self, monkeypatch, tmp_path):
        # Setup wizard historically wrote {"user_id": "hermes-user"} as the
        # suggested default. Treat that placeholder as unset so users on
        # gateways still get gateway-native ids — not silent collisions.
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "hermes-user"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


class TestMem0WriteMetadata:
    """Writes carry metadata.channel so per-channel filtered views are possible
    without coupling identity to the channel.
    """

    def _make_provider(self, channel: str = "cli"):
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._channel = channel
        provider._backend = FakeBackend()
        return provider


class _SentinelBackend:
    def __init__(self, *args):
        self.args = args


class TestCreateBackendRouting:
    """_create_backend() must pick the backend matching the configured mode/host."""

    def _provider(self, monkeypatch, *, mode="platform", api_key="k", host=""):
        # Neutralize lazy-install so the routing decision is all we exercise.
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None, raising=False)
        provider = Mem0MemoryProvider()
        provider._mode = mode
        provider._api_key = api_key
        provider._host = host
        provider._config = {"oss": {"vector_store": {"provider": "qdrant"}}}
        return provider

    def test_routes_to_selfhosted_when_host_set(self, monkeypatch):
        captured = {}

        class SH(_SentinelBackend):
            def __init__(self, api_key, host):
                captured["args"] = (api_key, host)

        monkeypatch.setattr("plugins.memory.mem0._backend.SelfHostedBackend", SH)
        provider = self._provider(monkeypatch, host="http://sh:8888", api_key="adminkey")
        backend = provider._create_backend()
        assert isinstance(backend, SH)
        assert captured["args"] == ("adminkey", "http://sh:8888")


    def test_oss_mode_takes_precedence_over_host(self, monkeypatch):
        class OB(_SentinelBackend):
            def __init__(self, cfg):
                pass

        monkeypatch.setattr("plugins.memory.mem0._backend.OSSBackend", OB)
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        assert isinstance(provider._create_backend(), OB)

    def test_prompt_label_matches_routing_when_oss_and_host_both_set(self, monkeypatch):
        # system_prompt_block must mirror _create_backend precedence: with both
        # mode=oss and host set, OSS wins the routing, so the prompt must label
        # OSS — not "self-hosted (HTTP API)". Guards the prompt-vs-routing lie.
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "OSS" in block
        assert "HTTP API" not in block


class TestSelfHostedConfig:
    """Config plumbing for self-hosted (MEM0_HOST env + is_available)."""

    def test_load_config_reads_mem0_host_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_HOST", "http://localhost:8888")
        assert mem0_plugin._load_config()["host"] == "http://localhost:8888"


