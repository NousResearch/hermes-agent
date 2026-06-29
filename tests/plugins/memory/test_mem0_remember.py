"""Tests for the mem0_remember background-review write helper + dedup ladder.

Covers spec 2026-06-27_mem0-in-background-review (registry-tool path per Phase-0
probe 0.2): the helper writes infer=False stamped write_origin=background_review,
and the dedup ladder (Tier-1 exact-hash, Tier-2 two-band cosine) gates the write.
"""

import json
import urllib.request
from urllib.parse import urlparse

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider, REMEMBER_SCHEMA


class _HTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if self._payload is None:
            return b""
        return json.dumps(self._payload).encode("utf-8")


def _json_body(request):
    if not request.data:
        return None
    return json.loads(request.data.decode("utf-8"))


def _provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_HOST", "http://mem0.test")
    monkeypatch.setenv("MEM0_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("MEM0_USER_ID", "ace")
    monkeypatch.setenv("MEM0_AGENT_ID", "apollo")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    p = Mem0MemoryProvider()
    p.initialize("test-session")
    return p


# ---------------------------------------------------------------------------
# Task 1.1 — schema shape
# ---------------------------------------------------------------------------

def test_remember_schema_shape():
    assert REMEMBER_SCHEMA["name"] == "mem0_remember"
    props = REMEMBER_SCHEMA["parameters"]["properties"]
    assert "fact" in props
    assert REMEMBER_SCHEMA["parameters"]["required"] == ["fact"]
    # Must carry the salience rubric so the model knows WHEN to save.
    desc = REMEMBER_SCHEMA["description"].lower()
    assert "durable" in desc
    assert "work-narration" in desc or "do not save" in desc


# ---------------------------------------------------------------------------
# Task 1.2 — dispatch: infer=False + write_origin=background_review
# ---------------------------------------------------------------------------

def test_remember_writes_infer_false_with_review_origin(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(request, timeout=0, context=None):
        path = urlparse(request.full_url).path
        body = _json_body(request)
        calls.append((request.get_method(), path, body))
        if request.get_method() == "POST" and path == "/search":
            # No prior dup -> empty results so the write proceeds.
            return _HTTPResponse({"results": []})
        if request.get_method() == "POST" and path == "/memories":
            return _HTTPResponse({"results": [{"id": "m-new", "memory": "stored"}]})
        raise AssertionError(f"unexpected {request.get_method()} {path}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = _provider(monkeypatch, tmp_path)

    out = json.loads(p.handle_tool_call("mem0_remember", {"fact": "Ace's backup brokerage is Fidelity."}))
    assert out.get("result") in ("Fact stored.", "stored")

    writes = [c for c in calls if c[0] == "POST" and c[1] == "/memories"]
    assert len(writes) == 1, f"expected exactly one write, got {calls}"
    body = writes[0][2]
    assert body.get("infer") is False
    meta = body.get("metadata", {})
    assert meta.get("write_origin") == "background_review"
    assert meta.get("write_kind") == "deliberate"


# ---------------------------------------------------------------------------
# Task 1.3 — registry-tool registration (denied-not-absent)
# ---------------------------------------------------------------------------

def test_mem0_remember_fails_closed_without_pin_user_id():
    """Greptile P1: the shared cached provider is only safe single-tenant. The tool
    must REFUSE to write when pin_user_id is off (multi-tenant → would cross
    namespaces), not silently write to the default user."""
    import tools.mem0_remember_tool as trt

    class _ProvPinOff:
        _pin_user_id = False
        def handle_tool_call(self, *a, **k):
            raise AssertionError("must not reach write when pin_user_id is off")

    class _ProvPinOn:
        _pin_user_id = True
        def handle_tool_call(self, name, args):
            return '{"result": "Fact stored.", "dedup": "wrote"}'

    trt._provider = _ProvPinOff()
    out = trt.mem0_remember_tool(fact="Ace prefers X")
    assert "disabled" in out.lower() and "pin_user_id" in out

    trt._provider = _ProvPinOn()
    out2 = trt.mem0_remember_tool(fact="Ace prefers X")
    assert "stored" in out2.lower()
    trt._provider = None  # reset cache for other tests


def test_mem0_remember_is_registry_tool_in_memory_write_toolset():
    """mem0_remember registers in its OWN toolset 'memory_write' (NOT 'memory'),
    so it's parent-resident (cache-stable) but denied in the fork whose whitelist
    is built from memory+skills. And it must be top-level auto-discoverable."""
    import os
    os.environ.setdefault("MEM0_HOST", "http://mem0.test")
    os.environ.setdefault("MEM0_ADMIN_API_KEY", "admin-key")
    import tools.mem0_remember_tool  # noqa: F401 (triggers registration)
    from tools.registry import registry, _module_registers_tools
    from pathlib import Path

    # Auto-discovery must see it (top-level register call, not nested).
    tool_file = Path(tools.mem0_remember_tool.__file__)
    assert _module_registers_tools(tool_file), "register() must be a top-level statement"

    assert registry.get_toolset_for_tool("mem0_remember") == "memory_write"
    assert "mem0_remember" not in registry.get_tool_names_for_toolset("memory"), (
        "must NOT be in the 'memory' toolset, or the fork would auto-whitelist it"
    )


def test_mem0_remember_is_RESIDENT_in_default_parent_tools():
    """REGRESSION GUARD (2026-06-29): the feature shipped SILENTLY DARK because
    mem0_remember was registered in toolset 'memory_write' but that toolset was
    in no enabled meta-toolset, so it never reached the parent's model-visible
    tools[] — and the fork inherits the parent's tools[], so the model could
    never SEE the tool no matter what the prompt clause or whitelist said. The
    'memory_write' + not-in-'memory' assertions above prove the NECESSARY half
    (won't auto-whitelist); this proves the SUFFICIENT half: with the DEFAULT
    enabled toolset, mem0_remember is actually present in the built tool
    definitions. If this fails, the bg-review mem0 write path is dark again.
    """
    import os
    os.environ.setdefault("MEM0_HOST", "http://mem0.test")
    os.environ.setdefault("MEM0_ADMIN_API_KEY", "admin-key")
    import tools.mem0_remember_tool  # noqa: F401 (triggers registration)
    # It must live in the shared core tool list so it's resident in tools[]
    # for every standing surface (CLI + every messaging gateway share _HERMES_CORE_TOOLS).
    from toolsets import _HERMES_CORE_TOOLS
    assert "mem0_remember" in _HERMES_CORE_TOOLS, (
        "mem0_remember must be in _HERMES_CORE_TOOLS so it is resident in the "
        "parent's tools[] and inheritable by the background-review fork"
    )
    # And it must resolve as resident for the default hermes-cli toolset.
    from toolsets import resolve_multiple_toolsets
    assert "mem0_remember" in set(resolve_multiple_toolsets(["hermes-cli"])), (
        "mem0_remember must resolve into the default hermes-cli toolset (resident)"
    )


def test_fork_whitelist_excludes_mem0_remember_by_default():
    """The review fork builds its whitelist from memory+skills toolsets; the
    memory_write tool must be ABSENT there (denied-not-absent default)."""
    import os
    os.environ.setdefault("MEM0_HOST", "http://mem0.test")
    os.environ.setdefault("MEM0_ADMIN_API_KEY", "admin-key")
    import tools.mem0_remember_tool  # noqa: F401
    from toolsets import resolve_multiple_toolsets
    fork_tools = set(resolve_multiple_toolsets(["memory", "skills"]))
    assert "mem0_remember" not in fork_tools


def test_dedup_norm_hash_normalizes():
    from plugins.memory.mem0 import _dedup_norm_hash
    assert _dedup_norm_hash("  Ace USES Schwab ") == _dedup_norm_hash("ace uses schwab")
    assert _dedup_norm_hash("a b") != _dedup_norm_hash("a c")


def test_tier1_exact_hash_skip(monkeypatch, tmp_path):
    """A row already carrying the candidate's dedup_hash -> skip, no write."""
    calls = []

    def fake_urlopen(request, timeout=0, context=None):
        path = urlparse(request.full_url).path
        body = _json_body(request)
        calls.append((request.get_method(), path, body))
        if request.get_method() == "POST" and path == "/search":
            # Tier-1 uses search_meta_filtered -> nested filters:{dedup_hash:...}.
            from plugins.memory.mem0 import _dedup_norm_hash
            h = _dedup_norm_hash("Ace uses Schwab")
            if (body or {}).get("filters", {}).get("dedup_hash") == h:
                return _HTTPResponse({"results": [{"id": "m-old", "memory": "Ace uses Schwab", "score": 1.0}]})
            return _HTTPResponse({"results": []})
        if request.get_method() == "POST" and path == "/memories":
            return _HTTPResponse({"results": [{"id": "m-new"}]})
        raise AssertionError(f"unexpected {request.get_method()} {path}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = _provider(monkeypatch, tmp_path)

    out = json.loads(p.handle_tool_call("mem0_remember", {"fact": "  ace uses schwab  "}))
    assert out.get("dedup") == "skipped_exacthash", out
    writes = [c for c in calls if c[0] == "POST" and c[1] == "/memories"]
    assert len(writes) == 0, "Tier-1 must skip the write on an exact-hash dup"


# ---------------------------------------------------------------------------
# Task D2 — Tier 2 two-band cosine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_cos,expect_write,expect_tag", [
    (0.997, False, "skipped_identical"),  # >= IDENTICAL (0.995) -> near-verbatim skip
    (0.97, True, "wrote_ambiguous"),      # ambiguous band -> WRITE (DD-1)
    (0.50, True, "wrote"),                # below floor -> plain write
])
def test_tier2_two_band(monkeypatch, tmp_path, top_cos, expect_write, expect_tag):
    calls = []

    def fake_urlopen(request, timeout=0, context=None):
        path = urlparse(request.full_url).path
        body = _json_body(request)
        calls.append((request.get_method(), path, body))
        if request.get_method() == "POST" and path == "/search":
            if "filters" in (body or {}):
                return _HTTPResponse({"results": []})  # Tier-1 (meta-filtered) miss
            # Tier-2 candidate retrieval -> return one candidate text.
            return _HTTPResponse({"results": [{"id": "m-near", "memory": "a near candidate fact"}]})
        if request.get_method() == "POST" and path == "/memories":
            return _HTTPResponse({"results": [{"id": "m-new"}]})
        raise AssertionError(f"unexpected {request.get_method()} {path}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = _provider(monkeypatch, tmp_path)
    # Deterministic, offline: force the client-side cosine to the parametrized value.
    monkeypatch.setattr(p, "_dedup_embed", lambda texts: [[1.0, 0.0]] + [[1.0, 0.0]] * (len(texts) - 1))
    monkeypatch.setattr(type(p), "_dedup_cos", staticmethod(lambda a, b: top_cos))

    out = json.loads(p.handle_tool_call("mem0_remember", {"fact": "some candidate fact"}))
    writes = [c for c in calls if c[0] == "POST" and c[1] == "/memories"]
    assert (len(writes) == 1) == expect_write, f"cos={top_cos} writes={len(writes)} out={out}"
    assert out.get("dedup") == expect_tag, out
