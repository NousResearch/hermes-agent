"""x-hermes-session + x-hermes-lane pool routing headers (reset-weighted router +
lanes, 2026-07-05).

Proves the stamps are (a) present ONLY for the api-proxy pool — canonical claude-apr (never egress to a direct/
third-party endpoint), (b) read per-request off the live agent (session id rotates
with compaction; lane reflects the live delegate_depth/platform), (c) the lane
classifier splits interactive/background correctly incl. the critical-aux (B1) case.
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import (
    _pool_affinity_headers,
    _pool_lane,
    _pool_lane_src,
)


def _agent(provider="claude-apr", session_id="20260705_120000_abc123",
           delegate_depth=0, platform="discord"):
    return SimpleNamespace(provider=provider, session_id=session_id,
                           _delegate_depth=delegate_depth, platform=platform)


# --------------------------------------------------------------------------- #
# session id (unchanged contract) + lane presence
# --------------------------------------------------------------------------- #
def test_headers_present_for_claude_apr():
    # canonical name (2026-07-08 rename): the gate MUST fire for claude-apr —
    # a stale single-literal gate silently killed stamping post-rename.
    h = _pool_affinity_headers(_agent("claude-apr"))
    assert h["x-hermes-session"] == "20260705_120000_abc123"
    assert h["x-hermes-lane"] == "interactive"          # top-level main turn
    assert "delegate_depth=0" in h["x-hermes-lane-src"]


def test_headers_absent_for_retired_claude_app_alias():
    # 2026-07-08: the legacy `claude-app` alias was fully RETIRED (killed at the
    # root, not just hidden). It is no longer a resolvable provider, so the gate
    # must NOT fire for it — nothing live pins claude-app anymore.
    assert _pool_affinity_headers(_agent("claude-app")) == {}


def test_headers_absent_for_claude_bpp_out_of_scope():
    # Greptile #205: claude-bpp is chat_completions (a different build_api_kwargs
    # branch) — deliberately OUT of scope so the helper only claims what's wired.
    assert _pool_affinity_headers(_agent("claude-bpp")) == {}
    assert _pool_affinity_headers(_agent("claude-bpr")) == {}


def test_headers_absent_for_direct_anthropic():
    assert _pool_affinity_headers(_agent("anthropic")) == {}
    assert _pool_affinity_headers(_agent("claude-api-proxy-f2")) == {}
    assert _pool_affinity_headers(_agent("claude-apx-2")) == {}


def test_headers_absent_for_third_party():
    assert _pool_affinity_headers(_agent("openai-codex")) == {}
    assert _pool_affinity_headers(_agent("xai")) == {}


def test_case_insensitive_provider():
    h = _pool_affinity_headers(_agent("Claude-Apr"))
    assert h["x-hermes-session"] == "20260705_120000_abc123"


def test_session_absent_but_lane_still_stamped():
    # no session id -> no session header, but the lane still classifies (pool-scoped)
    h = _pool_affinity_headers(_agent("claude-apr", session_id=None))
    assert "x-hermes-session" not in h
    assert h["x-hermes-lane"] == "interactive"


def test_session_reads_live_id_per_call():
    a = _agent("claude-apr", session_id="parent_sid")
    assert _pool_affinity_headers(a)["x-hermes-session"] == "parent_sid"
    a.session_id = "child_sid_after_compaction"   # compaction rotated it
    assert _pool_affinity_headers(a)["x-hermes-session"] == "child_sid_after_compaction"


def test_missing_provider_attr_safe():
    assert _pool_affinity_headers(SimpleNamespace(session_id="x")) == {}


# --------------------------------------------------------------------------- #
# lane classifier
# --------------------------------------------------------------------------- #
def test_lane_main_turn_interactive():
    assert _pool_lane(_agent(delegate_depth=0, platform="discord")) == "interactive"


def test_lane_subagent_background():
    assert _pool_lane(_agent(delegate_depth=1, platform="discord")) == "background"


def test_lane_cron_background():
    assert _pool_lane(_agent(delegate_depth=0, platform="cron")) == "background"


def test_lane_headless_cli_background():
    # a headless CLI / systemd / docker run: platform is empty/cli, NOT a live
    # messaging surface -> background (Greptile #206: must not eat interactive headroom).
    assert _pool_lane(_agent(delegate_depth=0, platform="cli")) == "background"
    assert _pool_lane(_agent(delegate_depth=0, platform="")) == "background"
    assert _pool_lane(_agent(delegate_depth=0, platform=None)) == "background"


def test_lane_unknown_platform_is_background():
    # any source that isn't a known interactive messaging surface -> background
    assert _pool_lane(_agent(platform="scheduler")) == "background"
    assert _pool_lane(_agent(platform="batch-job")) == "background"


def test_lane_session_source_env_fallback(monkeypatch):
    # platform empty but HERMES_SESSION_SOURCE=cron -> background (codebase idiom)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cron")
    assert _pool_lane(_agent(platform=None)) == "background"
    # HERMES_SESSION_SOURCE names a live surface -> interactive
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "discord")
    assert _pool_lane(_agent(platform=None)) == "interactive"


def test_lane_interactive_surfaces_stay_interactive():
    for p in ("discord", "telegram", "slack", "whatsapp", "imessage", "tui", "desktop"):
        assert _pool_lane(_agent(platform=p)) == "interactive", p


def test_lane_critical_aux_is_interactive():
    # B1: compaction/title/vision of a live top-level turn is ON the critical path
    # (the user turn blocks on it) -> interactive, must NOT be damped.
    a = _agent(delegate_depth=0, platform="discord")
    assert _pool_lane(a, aux_task="compression") == "interactive"
    assert _pool_lane(a, aux_task="title") == "interactive"


def test_lane_offpath_aux_is_background():
    # aux issued by a subagent or cron principal is not on a live human's critical path
    assert _pool_lane(_agent(delegate_depth=1), aux_task="compression") == "background"
    assert _pool_lane(_agent(platform="cron"), aux_task="vision") == "background"


def test_lane_src_carries_inputs():
    src = _pool_lane_src(_agent(delegate_depth=2, platform="telegram"), aux_task="session_search")
    assert "platform=telegram" in src
    assert "delegate_depth=2" in src
    assert "aux_task=session_search" in src


def test_lane_headers_stamped_in_affinity_for_pool_only():
    # the lane headers ride the same pool-only gate as the session header
    assert _pool_affinity_headers(_agent("claude-apr")).get("x-hermes-lane") == "interactive"
    assert "x-hermes-lane" not in _pool_affinity_headers(_agent("anthropic"))
    assert "x-hermes-lane" not in _pool_affinity_headers(_agent("openai-codex"))
