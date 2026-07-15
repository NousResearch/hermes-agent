"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
from typing import Any
from unittest.mock import patch

from agent import background_review as br


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model
        self._credential_pool: Any = None
        self.request_overrides = {}
        self.max_tokens: int | None = None

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "codex_responses"  # downgraded so agent-loop tools dispatch


def test_routing_to_different_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter", "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
        "credential_pool": "routed-pool",
        "request_overrides": {"extra_body": {"store": False}},
        "max_output_tokens": 2048,
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"
    assert rt["credential_pool"] == "routed-pool"
    assert rt["request_overrides"] == {"extra_body": {"store": False}}
    assert rt["max_tokens"] == 2048


def test_unrouted_runtime_keeps_parent_pool_and_overrides():
    agent = _FakeAgent()
    agent._credential_pool = "parent-pool"
    agent.request_overrides = {"service_tier": "priority"}
    agent.max_tokens = 4096
    with patch("hermes_cli.config.load_config", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["credential_pool"] == "parent-pool"
    assert rt["request_overrides"] == {"service_tier": "priority"}
    assert rt["max_tokens"] == 4096


def test_routing_same_model_as_parent_is_not_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False  # same model/provider → keep full-replay path


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"


# ---------------------------------------------------------------------------
# _digest_history — routed-path compact replay
# ---------------------------------------------------------------------------

def test_digest_under_tail_returns_full():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert br._digest_history(msgs, tail=24) == msgs


def test_digest_collapses_old_keeps_tail_verbatim():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 50))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 50))
    out = br._digest_history(msgs, tail=10)
    # First message is the synthetic digest (user role → alternation preserved).
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith("[Earlier conversation digest")
    # Recent tail preserved verbatim.
    assert out[-1] == msgs[-1]
    assert len(out) == 11  # 1 digest + 10 tail


def test_digest_does_not_open_tail_on_a_tool_message():
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 50))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}}]))
        msgs.append({"role": "tool", "content": "result " + "w" * 50})
    out = br._digest_history(msgs, tail=2)
    # The verbatim tail (after the digest) must not begin on a bare tool message.
    assert out[1]["role"] != "tool"


def test_digest_records_tool_names_in_arc():
    old = [
        _msg("user", "do the thing"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "skill_view", "arguments": "{}"}},
            {"function": {"name": "patch", "arguments": "{}"}}]),
    ]
    msgs = old + [_msg("user", f"tail{i}") for i in range(30)]
    out = br._digest_history(msgs, tail=10)
    digest = out[0]["content"]
    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest


# ---------------------------------------------------------------------------
# _should_skip_same_model_review — local single-server OOM guard (#54115)
#
# On a single local inference server the main turn and the review fork share
# one fixed n_ctx KV budget. When live usage is already high, a concurrent
# full-transcript replay can push combined usage past the window and the
# server rejects it ("Context size has been exceeded"). The guard skips the
# same-model review in that case; routed (separate-server) reviews are exempt.
# ---------------------------------------------------------------------------

_LOCAL = "http://127.0.0.1:8088/v1"   # llama.cpp default endpoint
_CLOUD = "https://api.openai.com/v1"


class _FakeCompressor:
    def __init__(self, context_length=0, last_total_tokens=0):
        self.context_length = context_length
        self.last_total_tokens = last_total_tokens


class _GuardAgent:
    def __init__(self, base_url="", *, context_length=0, last_total_tokens=0,
                 has_compressor=True):
        self.base_url = base_url
        self.context_compressor = (
            _FakeCompressor(context_length, last_total_tokens)
            if has_compressor else None
        )


def test_guard_skips_local_server_when_usage_high():
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=150_000)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is True


def test_guard_allows_local_server_when_usage_low():
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=20_000)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is False


def test_guard_never_skips_cloud_endpoint():
    # Cloud providers serve each request independently — no shared-slot OOM.
    agent = _GuardAgent(_CLOUD, context_length=200_000, last_total_tokens=190_000)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _CLOUD}) is False


def test_guard_does_not_skip_when_context_unknown():
    agent = _GuardAgent(_LOCAL, context_length=0, last_total_tokens=150_000)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is False


def test_guard_does_not_skip_when_usage_unknown():
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=0)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is False


def test_guard_does_not_skip_without_compressor():
    agent = _GuardAgent(_LOCAL, has_compressor=False)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is False


def test_guard_uses_agent_base_url_when_runtime_lacks_one():
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=150_000)
    with patch("agent.background_review._review_local_skip_fraction", return_value=0.45):
        assert br._should_skip_same_model_review(agent, {}) is True


def test_guard_disabled_when_fraction_out_of_range():
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=190_000)
    for disabled in (0.0, 1.0, -0.2, 1.5):
        with patch("agent.background_review._review_local_skip_fraction", return_value=disabled):
            assert br._should_skip_same_model_review(agent, {"base_url": _LOCAL}) is False


def test_run_review_skips_fork_when_guard_trips():
    """The fork AIAgent is never constructed when the guard trips — no replay
    request reaches the local server."""
    agent = _GuardAgent(_LOCAL, context_length=200_000, last_total_tokens=180_000)
    agent._safe_print = lambda *a, **k: None
    agent.background_review_callback = None
    rt = {"routed": False, "base_url": _LOCAL, "model": "m", "provider": "llamacpp"}
    with patch("agent.background_review._resolve_review_runtime", return_value=rt), \
         patch("agent.background_review._should_skip_same_model_review", return_value=True), \
         patch("run_agent.AIAgent") as fake_aiagent:
        br._run_review_in_thread(agent, [], "review prompt")
    fake_aiagent.assert_not_called()


# _review_local_skip_fraction — config reader

def test_fraction_defaults_when_unset():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert br._review_local_skip_fraction() == br._DEFAULT_LOCAL_SKIP_FRACTION


def test_fraction_reads_config_value():
    cfg = {"auxiliary": {"background_review": {"local_skip_context_fraction": 0.6}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert br._review_local_skip_fraction() == 0.6


def test_fraction_falls_back_on_bad_value():
    cfg = {"auxiliary": {"background_review": {"local_skip_context_fraction": "nope"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert br._review_local_skip_fraction() == br._DEFAULT_LOCAL_SKIP_FRACTION


def test_fraction_registered_in_default_config():
    """DEFAULT_CONFIG must declare the knob so `config check` discovers it.

    hermes-sweeper review on #54255: the option lived only in
    cli-config.yaml.example + the runtime reader fallback, so recursive
    DEFAULT_CONFIG walks (get_missing_config_fields) never surfaced it.
    """
    from hermes_cli.config import DEFAULT_CONFIG

    bg = DEFAULT_CONFIG["auxiliary"]["background_review"]
    assert "local_skip_context_fraction" in bg
    assert bg["local_skip_context_fraction"] == br._DEFAULT_LOCAL_SKIP_FRACTION
    assert bg["local_skip_context_fraction"] == 0.45


def test_fraction_merged_config_default_path(tmp_path, monkeypatch):
    """load_config deep-merge supplies the DEFAULT_CONFIG value when user yaml omits it.

    Covers the real merged-config path (not only the mocked reader).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # Minimal user config that does NOT set local_skip_context_fraction.
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    provider: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.config import load_config

    # Clear any process-level load_config cache keyed on path.
    try:
        from hermes_cli import config as cfg_mod
        cache = getattr(cfg_mod, "_CONFIG_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()
        for attr in ("_load_config_cache", "_config_cache"):
            if hasattr(cfg_mod, attr):
                setattr(cfg_mod, attr, None)
    except Exception:
        pass

    cfg = load_config()
    bg = cfg.get("auxiliary", {}).get("background_review", {})
    assert bg.get("local_skip_context_fraction") == 0.45
    # Reader should observe the merged default without a mock.
    assert br._review_local_skip_fraction() == 0.45
