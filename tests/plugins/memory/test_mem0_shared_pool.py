"""Tests for the mem0 agent-scoped shared company pool (mem0_search_shared / mem0_add_shared).

Covers:
- off-by-default registration (shared tools absent unless shared_pool.enabled)
- agent-scoped read returns the full agent-wide view via agent_id filter
- config-driven submitter allowlist (hard authorization gate on writes)
- default allow-everything when authorized_submitters is empty
- system-prompt guidance appears only when the pool is enabled
"""

import json
import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Fake Mem0Backend capturing calls (mirrors test_mem0_v3.py)."""

    def __init__(self, search_results=None):
        self._search_results = search_results or []
        self.captured = []

    def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank}))
        return self._search_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append((
            "add", messages,
            {"user_id": user_id, "agent_id": agent_id, "infer": infer, "metadata": metadata},
        ))
        return {"status": "PENDING", "event_id": "evt-shared"}


def _provider(backend=None, *, user_id="u123", agent_id="hermes",
              shared_enabled=False, submitters=None, channel="cli",
              mode="oss", host=""):
    """Build a provider with explicit shared-pool state, no backend side effects."""
    provider = Mem0MemoryProvider()
    provider._user_id = user_id
    provider._agent_id = agent_id
    provider._channel = channel
    provider._mode = mode
    provider._host = host
    provider._shared_pool_enabled = shared_enabled
    provider._shared_pool_submitters = list(submitters or [])
    provider._shared_pool_resolved = True
    provider._backend = backend or FakeBackend()
    return provider


class TestSharedPoolRegistration:
    def test_disabled_by_default_registers_no_shared_tools(self):
        provider = _provider()
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert names == ["mem0_search", "mem0_add", "mem0_update", "mem0_delete"]
        assert "mem0_search_shared" not in names
        assert "mem0_add_shared" not in names

    def test_enabled_registers_shared_tools(self):
        provider = _provider(shared_enabled=True)
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert "mem0_search_shared" in names
        assert "mem0_add_shared" in names

    def test_pool_honors_config_from_initialize(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "k")
        (tmp_path / "mem0.json").write_text(json.dumps({
            "shared_pool": {"enabled": True, "authorized_submitters": ["u123", "u456"]},
        }))
        provider = Mem0MemoryProvider()
        provider._create_backend = lambda: FakeBackend()  # type: ignore[method-assign]
        provider.initialize("test-session", user_id="u123")
        assert provider._shared_pool_enabled is True
        assert provider._shared_pool_submitters == ["u123", "u456"]

    def test_shared_tools_registered_before_initialize(self, monkeypatch, tmp_path):
        """Regression: add_provider() calls get_tool_schemas() BEFORE
        initialize_all(), so the shared tools must be resolvable from config
        straight off the constructor — otherwise they are advertised in the
        system prompt but never registered for dispatch ("Unknown tool")."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "k")
        (tmp_path / "mem0.json").write_text(json.dumps({
            "shared_pool": {"enabled": True, "authorized_submitters": ["a", "b"]},
        }))
        provider = Mem0MemoryProvider()  # NOT initialized intentionally
        names = [s["name"] for s in provider.get_tool_schemas()]
        assert "mem0_search_shared" in names
        assert "mem0_add_shared" in names
        assert provider._shared_pool_enabled is True
        assert provider._shared_pool_submitters == ["a", "b"]


class TestSharedPoolSearch:
    def test_search_uses_agent_id_only_filter(self):
        backend = FakeBackend(
            search_results=[{"id": "s1", "memory": "All remote access requires MFA", "score": 0.9,
                             "metadata": {"scope": "shared"}}]
        )
        provider = _provider(backend, shared_enabled=True)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "remote access"}))
        assert result["results"][0]["memory"] == "All remote access requires MFA"
        # Agent-scoped read keyword source is the same agent_id-only filter, but
        # per-user records are filtered out of the returned items (see below).
        assert backend.captured[0][0] == "search"
        assert backend.captured[0][2]["filters"] == {"agent_id": "hermes"}
        assert "user_id" not in backend.captured[0][2]["filters"]

    def test_search_excludes_per_user_records(self):
        # Shared records are written with no user_id AND positively tagged
        # metadata.scope="shared"; per-user records carry user_id. Only shared
        # records (both belts) may appear in the company view.
        backend = FakeBackend(
            search_results=[
                {"id": "share1", "memory": "Company policy: MFA required", "score": 0.9,
                 "metadata": {"scope": "shared"}},   # shared — tagged, no user_id
                {"id": "priv1", "memory": "Kyle likes early meetings", "score": 0.8,
                 "user_id": "kyle", "metadata": {"scope": "shared"}},  # per-user — user_id wins
                {"id": "priv2", "memory": "Bob's client is secret", "score": 0.7,
                 "metadata": {"user_id": "bob"}},  # per-user — untagged, has user_id
            ]
        )
        provider = _provider(backend, shared_enabled=True)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "company"}))
        ids = [item["id"] for item in result["results"]]
        assert ids == ["share1"]  # per-user records filtered out
        assert result["count"] == 1

    def test_search_drops_untagged_agent_records(self):
        # A record with no positive shared marker is NOT admitted — even if it
        # has no user_id — because it is indistinguishable from a private note
        # (isolation-by-intersection, not isolation-by-drop).
        backend = FakeBackend(
            search_results=[
                {"id": "mystery", "memory": "Some agent-scoped note without a marker", "score": 0.9},
            ]
        )
        provider = _provider(backend, shared_enabled=True)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "note"}))
        assert result.get("result") == "No shared memories found."

    def test_search_drops_shared_tagged_record_that_also_has_user_id(self):
        # Belt 2: even a positively shared-tagged record is excluded if it also
        # carries a per-user scope.
        backend = FakeBackend(
            search_results=[
                {"id": "tagged", "memory": "Company policy: MFA", "score": 0.9,
                 "metadata": {"scope": "shared", "user_id": "kyle"}},
            ]
        )
        provider = _provider(backend, shared_enabled=True)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "MFA"}))
        assert result.get("result") == "No shared memories found."

    def test_search_returns_no_shared_when_all_results_are_per_user(self):
        backend = FakeBackend(
            search_results=[
                {"id": "priv1", "memory": "Kyle likes early meetings", "score": 0.8, "user_id": "kyle"},
            ]
        )
        provider = _provider(backend, shared_enabled=True)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "meetings"}))
        assert result.get("result") == "No shared memories found."

    def test_search_refused_when_pool_disabled(self):
        provider = _provider(shared_enabled=False)
        result = json.loads(provider.handle_tool_call("mem0_search_shared", {"query": "x"}))
        assert "not enabled" in result.get("error", "")


class TestSharedPoolAddAuthorization:
    def test_add_writes_agent_scoped_no_user_id(self):
        backend = FakeBackend()
        provider = _provider(backend, shared_enabled=True, user_id="u123")
        result = json.loads(provider.handle_tool_call(
            "mem0_add_shared", {"content": "Company policy: MFA required"}
        ))
        assert result.get("result")  # stored
        assert backend.captured[0][0] == "add"
        # Agent-scoped write: no per-user principal.
        assert backend.captured[0][2]["user_id"] is None
        assert backend.captured[0][2]["agent_id"] == "hermes"
        assert backend.captured[0][2]["infer"] is False
        # Positively tagged as shared so the read path can identify it without
        # relying only on the absence of a user_id.
        assert backend.captured[0][2]["metadata"]["scope"] == "shared"

    def test_unlisted_operator_refused_when_allowlist_set(self):
        backend = FakeBackend()
        provider = _provider(backend, shared_enabled=True, user_id="intruder",
                             submitters=["u123", "u456"])
        result = json.loads(provider.handle_tool_call(
            "mem0_add_shared", {"content": "should not land"}
        ))
        assert "not authorized" in result.get("error", "")
        assert backend.captured == []  # no write reached the backend

    def test_listed_operator_allowed(self):
        backend = FakeBackend()
        provider = _provider(backend, shared_enabled=True, user_id="u456",
                             submitters=["u123", "u456"])
        result = json.loads(provider.handle_tool_call(
            "mem0_add_shared", {"content": "approved fact"}
        ))
        assert result.get("result")
        assert len(backend.captured) == 1

    def test_empty_allowlist_allows_any_operator(self):
        # Default: empty authorized_submitters => anyone may contribute.
        backend = FakeBackend()
        provider = _provider(backend, shared_enabled=True, user_id="someone-else")
        result = json.loads(provider.handle_tool_call(
            "mem0_add_shared", {"content": "open fact"}
        ))
        assert result.get("result")
        assert len(backend.captured) == 1

    def test_add_refused_when_pool_disabled(self):
        provider = _provider(shared_enabled=False)
        result = json.loads(provider.handle_tool_call("mem0_add_shared", {"content": "x"}))
        assert "not enabled" in result.get("error", "")


class TestSharedPoolSystemPrompt:
    def test_prompt_mentions_shared_only_when_enabled(self):
        off = _provider(shared_enabled=False)
        assert "mem0_search_shared" not in off.system_prompt_block()

        on = _provider(shared_enabled=True)
        block = on.system_prompt_block()
        assert "mem0_search_shared" in block
        assert "mem0_add_shared" in block
        assert "SHARED company-knowledge" in block


class TestSharedPoolHelpers:
    def test_authorized_returns_true_with_empty_allowlist(self):
        provider = _provider(submitters=[])
        assert provider._shared_pool_authorized() is True

    def test_authorized_checks_user_id_against_allowlist(self):
        assert _provider(user_id="a", submitters=["a", "b"])._shared_pool_authorized() is True
        assert _provider(user_id="c", submitters=["a", "b"])._shared_pool_authorized() is False

    def test_authorized_matches_case_insensitively(self):
        # Operator identifiers can arrive from any gateway with case variance
        # (email-style ids, username-cased CLI ids, etc.). The allowlist must
        # not silently refuse a differently-cased (but otherwise equal) id.
        assert _provider(user_id="Admin@Example.com",
                         submitters=["admin@example.com"])._shared_pool_authorized() is True
        assert _provider(user_id="admin@example.com",
                         submitters=["ADMIN@EXAMPLE.COM"])._shared_pool_authorized() is True
        assert _provider(user_id="nobody@example.com",
                         submitters=["admin@example.com"])._shared_pool_authorized() is False

    def test_filter_is_agent_id_only(self):
        assert _provider(agent_id="foxtrot")._shared_submitters_filter() == {"agent_id": "foxtrot"}