"""Regression tests for #117790 — ``fact_store.add`` stored whatever ``content``
the model sent, and read ``category`` through ``args.get("category", "general")``.

The model therefore re-stored the raw JSON of tool calls it had just made
(``{"name": ..., "arguments": ...}``) as if it were a fact about the user, and
every deliberate classification collapsed into the same 'general' bucket, so
genuine facts drowned in search/probe/reason results.

The guard under test is deliberately narrow: only the unambiguous tool-call
shape is refused, so a fact written as prose — or as any other JSON — still
stores. A refused ``add`` must also leave the store untouched.
"""

import json

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _add(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", {"action": "add", **args}))


def _contents(provider):
    return [f["content"] for f in provider._store.list_facts(limit=100)]


class TestAddRefusesPayloadShapedContent:
    def test_raw_tool_call_json_is_refused_and_not_stored(self, tmp_path):
        provider = _make_provider(tmp_path)
        raw = json.dumps({"name": "terminal", "arguments": {"command": "ls -la"}})
        result = _add(provider, content=raw, category="tool")
        assert "tool-call payload" in result["error"]
        assert _contents(provider) == []

    def test_other_json_is_not_refused(self, tmp_path):
        """Narrow by design: only a 'name'+'arguments' payload trips the guard."""
        provider = _make_provider(tmp_path)
        result = _add(provider, content=json.dumps({"deploy_window": "fridays"}),
                      category="project")
        assert result["status"] == "added"
        assert len(_contents(provider)) == 1


class TestAddRequiresExplicitCategory:
    def test_missing_category_is_refused_and_not_stored(self, tmp_path):
        provider = _make_provider(tmp_path)
        result = _add(provider, content="the user prefers dark mode")
        assert "category is required" in result["error"]
        assert _contents(provider) == []

    def test_category_is_what_the_caller_chose(self, tmp_path):
        provider = _make_provider(tmp_path)
        _add(provider, content="the user prefers dark mode", category="user_pref")
        stored = provider._store.list_facts(limit=10)
        assert [f["category"] for f in stored] == ["user_pref"]
