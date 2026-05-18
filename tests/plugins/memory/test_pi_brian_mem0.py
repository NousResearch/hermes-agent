import json

from plugins.memory.pi_brian_mem0 import PiBrianMem0MemoryProvider


def make_provider() -> PiBrianMem0MemoryProvider:
	provider = PiBrianMem0MemoryProvider()
	provider._base_url = "http://127.0.0.1:8000"
	provider._user_id = "tg-user-1"
	provider._agent_id = "hermes-brian"
	provider._prefetch_limit = 5
	provider._prefetch_chars = 1200
	provider._sync_turns = True
	provider._agent_context = "primary"
	return provider


def test_gateway_user_id_overrides_default(monkeypatch, tmp_path):
	monkeypatch.setenv("HERMES_HOME", str(tmp_path))
	monkeypatch.setenv("MEM0_BASE_URL", "http://127.0.0.1:8000")
	monkeypatch.setenv("MEM0_USER_ID", "fallback-user")
	provider = PiBrianMem0MemoryProvider()
	provider.initialize("session-1", user_id="telegram-42")
	assert provider._user_id == "telegram-42"


def test_mem0_profile_uses_get_memories(monkeypatch):
	provider = make_provider()

	def fake_request(method, path, **kwargs):
		assert method == "GET"
		assert path == "/memories"
		assert kwargs["query"] == {"user_id": "tg-user-1"}
		return {"results": [{"memory": "prefers concise morning digest"}]}

	monkeypatch.setattr(provider, "_request_json", fake_request)
	result = json.loads(provider.handle_tool_call("mem0_profile", {}))
	assert result["count"] == 1
	assert "morning digest" in result["result"]


def test_mem0_search_returns_bounded_results(monkeypatch):
	provider = make_provider()

	def fake_request(method, path, **kwargs):
		assert method == "POST"
		assert path == "/search"
		assert kwargs["payload"] == {"query": "digest", "user_id": "tg-user-1"}
		return {"results": [{"memory": "a", "score": 0.9}, {"memory": "b", "score": 0.8}]}

	monkeypatch.setattr(provider, "_request_json", fake_request)
	result = json.loads(provider.handle_tool_call("mem0_search", {"query": "digest", "top_k": 1}))
	assert result["count"] == 1
	assert result["results"][0]["memory"] == "a"


def test_queue_prefetch_formats_cached_results(monkeypatch):
	provider = make_provider()

	def fake_request(method, path, **kwargs):
		return {"results": [{"memory": "poseidon runs assistant deploys", "score": 0.42}]}

	monkeypatch.setattr(provider, "_request_json", fake_request)
	provider.queue_prefetch("poseidon deploys")
	provider._prefetch_thread.join(timeout=2)
	result = provider.prefetch("poseidon deploys")
	assert "Relevant semantic memory" in result
	assert "poseidon runs assistant deploys" in result


def test_sync_turn_posts_user_and_assistant_messages(monkeypatch):
	provider = make_provider()
	posted = []

	def fake_request(method, path, **kwargs):
		posted.append((method, path, kwargs["payload"]))
		return {"results": []}

	monkeypatch.setattr(provider, "_request_json", fake_request)
	provider.sync_turn("user said", "assistant replied", session_id="s-1")
	provider._sync_thread.join(timeout=2)
	assert posted[0][0] == "POST"
	assert posted[0][1] == "/memories"
	assert posted[0][2]["user_id"] == "tg-user-1"
	assert posted[0][2]["messages"][0]["content"] == "user said"
	assert posted[0][2]["messages"][1]["content"] == "assistant replied"


def test_on_memory_write_mirrors_builtin_memory(monkeypatch):
	provider = make_provider()
	posted = []

	def fake_request(method, path, **kwargs):
		posted.append(kwargs["payload"])
		return {"results": []}

	monkeypatch.setattr(provider, "_request_json", fake_request)
	provider.on_memory_write("add", "user", "Brian likes concise digests")
	assert posted
	assert posted[0]["metadata"]["target"] == "user"
	assert posted[0]["messages"][0]["content"] == "[user] Brian likes concise digests"
