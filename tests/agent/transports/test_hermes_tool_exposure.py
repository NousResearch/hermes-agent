"""Tests for the shared Hermes-tool-exposure layer used by both the codex
stdio backend and the Claude Agent SDK in-process backend."""

from agent.transports import hermes_tool_exposure as hte


# ---------------------------------------------------------------------------
# Curated list
# ---------------------------------------------------------------------------
def test_curated_excludes_agent_level_and_builtins():
    curated = set(hte.CURATED_STATELESS_TOOLS)
    # The 4 agent-level tools need a live agent — must not be in the
    # stateless-safe curation.
    for name in ("todo", "memory", "delegate_task", "session_search"):
        assert name not in curated
    # Things the external engine has its own builtins for.
    for name in ("terminal", "read_file", "write_file", "patch", "search_files"):
        assert name not in curated
    # A few we DO expose.
    for name in ("web_search", "web_extract", "vision_analyze", "kanban_complete"):
        assert name in curated


# ---------------------------------------------------------------------------
# normalize_tool_spec
# ---------------------------------------------------------------------------
def test_normalize_openai_format():
    spec = {"type": "function", "function": {
        "name": "web_search", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
    name, desc, schema = hte.normalize_tool_spec(spec)
    assert name == "web_search"
    assert desc == "Search the web"
    assert schema["properties"] == {"q": {"type": "string"}}


def test_normalize_anthropic_format_and_mcp_prefix():
    spec = {"name": "mcp__read_file", "description": "Read", "input_schema": {"type": "object"}}
    name, desc, schema = hte.normalize_tool_spec(spec)
    assert name == "read_file"          # mcp__ stripped
    assert schema == {"type": "object"}


def test_normalize_missing_schema_defaults():
    name, desc, schema = hte.normalize_tool_spec({"name": "foo"})
    assert name == "foo"
    assert schema == {"type": "object", "properties": {}}


def test_normalize_bad_input():
    assert hte.normalize_tool_spec("nope") is None
    assert hte.normalize_tool_spec({"description": "no name"}) is None
    assert hte.normalize_tool_spec({"type": "function", "function": {}}) is None


# ---------------------------------------------------------------------------
# resolve_curated_specs
# ---------------------------------------------------------------------------
def test_resolve_curated_specs_orders_and_filters():
    defs = [
        {"type": "function", "function": {"name": "web_search", "description": "s",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "vision_analyze", "description": "v",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "not_registered_extra", "description": "x",
                                          "parameters": {"type": "object"}}},
    ]
    out = hte.resolve_curated_specs(defs, names=("vision_analyze", "web_search", "image_generate"))
    # image_generate not in defs → dropped; order follows `names`.
    assert list(out.keys()) == ["vision_analyze", "web_search"]
    assert out["web_search"][0] == "s"


def test_resolve_curated_specs_empty():
    assert hte.resolve_curated_specs(None) == {}
    assert hte.resolve_curated_specs([]) == {}


# ---------------------------------------------------------------------------
# error envelope (producer + detector agree — the historical 1-vs-2-key bug)
# ---------------------------------------------------------------------------
def test_error_envelope_round_trips():
    env = hte.make_error_envelope(ValueError("boom"), "web_search")
    assert hte.looks_like_tool_error(env)          # producer + detector agree
    assert "web_search" in env and "boom" in env


def test_error_envelope_single_key():
    import json
    env = hte.make_error_envelope("oops")
    parsed = json.loads(env)
    assert parsed == {"error": "oops"}             # single key, so detector matches


def test_looks_like_tool_error_negatives():
    assert not hte.looks_like_tool_error("plain text")
    assert not hte.looks_like_tool_error('{"data": 1}')
    # A legit result that merely contains an "error" field among others is not
    # a bare error envelope.
    assert not hte.looks_like_tool_error('{"error": "x", "data": 1}')
    assert not hte.looks_like_tool_error(123)


# ---------------------------------------------------------------------------
# wrap_untrusted (delegates to the native promptware defense)
# ---------------------------------------------------------------------------
def test_wrap_untrusted_wraps_untrusted_tool():
    long_text = "a" * 100
    wrapped = hte.wrap_untrusted("web_search", long_text)
    assert "<untrusted_tool_result" in wrapped


def test_wrap_untrusted_passes_trusted_through():
    # vision_analyze is not in the untrusted set → unchanged.
    text = "some analysis result that is definitely longer than the wrap threshold"
    assert hte.wrap_untrusted("vision_analyze", text) == text
