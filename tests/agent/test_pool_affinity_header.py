"""x-hermes-session pool affinity header stamp (reset-weighted router, 2026-07-05).

Proves the stamp is (a) present ONLY for the pool providers, (b) read per-request
off the live agent.session_id (so it rotates with compaction's child session id),
(c) absent for direct Anthropic / third-party / non-pool providers, so it never
egresses.
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import _pool_affinity_headers


def _agent(provider, session_id="20260705_120000_abc123"):
    return SimpleNamespace(provider=provider, session_id=session_id)


def test_stamp_present_for_claude_app():
    h = _pool_affinity_headers(_agent("claude-app"))
    assert h == {"x-hermes-session": "20260705_120000_abc123"}


def test_stamp_absent_for_claude_bpp_out_of_scope():
    # Greptile #205: claude-bpp resolves to api_mode chat_completions (a different
    # build_api_kwargs branch, separate daemon + affinity map) — deliberately OUT of
    # scope so the helper only claims what the anthropic_messages wiring stamps.
    assert _pool_affinity_headers(_agent("claude-bpp")) == {}


def test_stamp_absent_for_direct_anthropic():
    # a direct anthropic / OAuth provider must NOT receive the header (no egress)
    assert _pool_affinity_headers(_agent("anthropic")) == {}
    assert _pool_affinity_headers(_agent("claude-api-proxy-f2")) == {}


def test_stamp_absent_for_third_party():
    assert _pool_affinity_headers(_agent("openai-codex")) == {}
    assert _pool_affinity_headers(_agent("xai")) == {}


def test_stamp_case_insensitive_provider():
    assert _pool_affinity_headers(_agent("Claude-App")) == {
        "x-hermes-session": "20260705_120000_abc123"}


def test_stamp_absent_when_no_session_id():
    assert _pool_affinity_headers(_agent("claude-app", session_id=None)) == {}
    assert _pool_affinity_headers(_agent("claude-app", session_id="")) == {}


def test_stamp_reads_live_session_id_per_call():
    # THE compaction-rotation property: the header reflects the CURRENT session_id,
    # so when compaction mints a child id the next call carries the new key.
    a = _agent("claude-app", session_id="parent_sid")
    assert _pool_affinity_headers(a)["x-hermes-session"] == "parent_sid"
    a.session_id = "child_sid_after_compaction"   # compaction rotated it
    assert _pool_affinity_headers(a)["x-hermes-session"] == "child_sid_after_compaction"


def test_stamp_missing_provider_attr_safe():
    assert _pool_affinity_headers(SimpleNamespace(session_id="x")) == {}
