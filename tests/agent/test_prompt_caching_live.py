"""Live E2E: long-lived prefix caching on Claude via OpenRouter.

Run only when LIVE_OR_KEY env var is set. Skipped under the normal hermetic
test suite (which unsets credentials).
"""
import os, sys, tempfile, time, shutil, pytest


# Probe for the key BEFORE conftest unsets it
_LIVE_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LIVE_OR_KEY")
if not _LIVE_KEY:
    # Try to read directly from .env
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    _LIVE_KEY = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break


pytestmark = pytest.mark.skipif(
    not _LIVE_KEY,
    reason="set OPENROUTER_API_KEY (or LIVE_OR_KEY) to run live cache test",
)


def test_long_lived_prefix_cache_e2e_openrouter(tmp_path, monkeypatch):
    """Two AIAgent runs in fresh sessions: call 1 writes cache, call 2 reads it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The hermetic conftest unsets OPENROUTER_API_KEY — restore for this test
    monkeypatch.setenv("OPENROUTER_API_KEY", _LIVE_KEY)

    # Minimal config — but with enough toolset/guidance to exceed Anthropic's
    # ~1024-token minimum-cacheable-prefix threshold. Anthropic silently
    # ignores cache_control markers on small blocks.
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "anthropic/claude-haiku-4.5"},
        "prompt_caching": {"long_lived_prefix": True, "long_lived_ttl": "1h", "cache_ttl": "5m"},
        "agent": {"tool_use_enforcement": True},   # adds substantial guidance text
        "memory": {"provider": ""},
        "compression": {"enabled": False},
    }))

    from run_agent import AIAgent

    def make_agent():
        return AIAgent(
            api_key=_LIVE_KEY,
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            api_mode="chat_completions",
            # Use the default toolset roster — the tools array (~13k tokens
            # for ~35 tools) is what carries the bulk of the cross-session
            # cache value. With a tiny toolset the cached prefix can fall
            # below Anthropic Haiku's 2048-token minimum cacheable size and
            # the marker is silently ignored.
            enabled_toolsets=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )

    a1 = make_agent()
    assert a1._use_prompt_caching is True, "policy should enable caching for Claude on OR"
    assert a1._use_long_lived_prefix_cache is True, "long-lived path should activate"
    parts = a1._build_system_prompt_parts()
    print(f"\nstable={len(parts['stable']):,} ctx={len(parts['context']):,} volatile={len(parts['volatile']):,} chars")
    print(f"tool count: {len(a1.tools or [])}")

    # Use distinct user messages each call so OpenRouter's response cache
    # doesn't short-circuit the upstream Anthropic call (we need real
    # Anthropic billing visibility to verify cache_creation/cache_read).
    USER_1 = "Reply with the single word ALPHA."
    USER_2 = "Reply with the single word BRAVO."

    print("\n--- Call 1 (cold) ---")
    r1 = a1.run_conversation(USER_1, conversation_history=[])
    print(f"final_response[:80]: {(r1.get('final_response') or '')[:80]!r}")
    cr1 = a1.session_cache_read_tokens
    cw1 = a1.session_cache_write_tokens
    print(f"call1: cache_read={cr1} cache_write={cw1}")

    # Wait so cache settles, then fresh agent (NEW SESSION) for cross-session read
    time.sleep(2)
    a2 = make_agent()
    assert a2.session_id != a1.session_id, "second agent must have a new session"

    print("\n--- Call 2 (warm, NEW session, different user msg) ---")
    r2 = a2.run_conversation(USER_2, conversation_history=[])
    print(f"final_response[:80]: {(r2.get('final_response') or '')[:80]!r}")
    cr2 = a2.session_cache_read_tokens
    cw2 = a2.session_cache_write_tokens
    print(f"call2: cache_read={cr2} cache_write={cw2}")

    print(f"\n=== VERDICT ===")
    print(f"  call1 wrote {cw1:,} cache tokens, read {cr1:,}")
    print(f"  call2 wrote {cw2:,} cache tokens, read {cr2:,}")
    if cw1:
        print(f"  cross-session read fraction: cr2/cw1 = {cr2/cw1:.2%}")

    # Assertions
    assert cw1 > 0, f"call 1 must write cache (got {cw1}); long-lived layout not reaching wire"
    assert cr2 > 0, (
        f"call 2 must read cache cross-session (got {cr2}); "
        f"stable prefix is not byte-stable across sessions"
    )
    assert cr2 >= 1000, f"cache_read on call 2 ({cr2}) too small to indicate real reuse"
