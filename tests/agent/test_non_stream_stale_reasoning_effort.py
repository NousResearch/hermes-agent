"""Regression tests for reasoning-effort-aware non-stream stale timeout.

Before this fix: ``AIAgent._compute_non_stream_stale_timeout`` returned
the 90s base (or context-tier bumped value) regardless of
reasoning_effort. GPT-5+ models with effort=medium routinely spend
200-280s in server-side thinking before emitting any response on
non-streaming Codex Responses paths (Copilot fallback after streaming
failures). The 90s stale detector killed those connections after every
attempt — even though they would have succeeded on their own.

User-visible symptom (oneshot mode)::

    API call failed after 3 retries: Non-streaming API call timed out
    after 90s with no response (threshold: 90s)

After this fix: the timeout multiplies by 5.0× for high/xhigh effort
and 3.5× for medium effort, reading the effort from
``api_payload['reasoning']['effort']`` (Responses API) or
``api_payload['reasoning_effort']`` (Chat Completions). The agent-level
``self.reasoning_config`` is checked as a fallback — but NOT relied on,
because oneshot / cron / many default chat paths never populate it.
"""

from __future__ import annotations

from pathlib import Path


def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── reads effort from api_payload (Responses API shape) ──────────────────


def test_responses_api_high_effort_applies_5x_multiplier(monkeypatch, tmp_path):
    """api_payload['reasoning']['effort']='high' -> 90s × 5.0 = 450s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "instructions": "",
        "reasoning": {"effort": "high"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0 * 5.0


def test_responses_api_xhigh_effort_applies_5x_multiplier(monkeypatch, tmp_path):
    """api_payload['reasoning']['effort']='xhigh' -> same 5.0× multiplier."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "instructions": "",
        "reasoning": {"effort": "xhigh"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0 * 5.0


def test_responses_api_medium_effort_applies_35x_multiplier(monkeypatch, tmp_path):
    """api_payload['reasoning']['effort']='medium' -> 90s × 3.5 = 315s.

    This is the case that caused the council failures: gpt-5.5 defaults
    to medium effort and the bare 90s killed legitimate 200-280s reasoning.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "instructions": "",
        "reasoning": {"effort": "medium"},
    }
    # 90.0 * (5.0 * 0.7) = 315.0
    assert agent._compute_non_stream_stale_timeout(payload) == 315.0


def test_responses_api_low_effort_keeps_base(monkeypatch, tmp_path):
    """Low effort should NOT trigger the multiplier — 90s base preserved."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "instructions": "",
        "reasoning": {"effort": "low"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


def test_responses_api_minimal_effort_keeps_base(monkeypatch, tmp_path):
    """Minimal effort should NOT trigger the multiplier."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "instructions": "",
        "reasoning": {"effort": "minimal"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


def test_responses_api_no_reasoning_keeps_base(monkeypatch, tmp_path):
    """No reasoning field at all -> base 90s, no multiplier."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "hi", "instructions": ""}
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


# ── reads effort from api_payload (Chat Completions shape) ───────────────


def test_chat_completions_reasoning_effort_high(monkeypatch, tmp_path):
    """Chat Completions uses flat ``reasoning_effort`` field (not nested)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(
        tmp_path,
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.4",
    )
    payload = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0 * 5.0


# ── env-var override ─────────────────────────────────────────────────────


def test_env_var_overrides_multiplier(monkeypatch, tmp_path):
    """HERMES_API_CALL_STALE_REASONING_MULTIPLIER env var overrides default."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", "10.0")
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "high"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0 * 10.0


def test_env_var_disables_multiplier_when_one(monkeypatch, tmp_path):
    """Setting multiplier to 1.0 effectively disables the scaling."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", "1.0")
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "high"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


def test_medium_effort_override_one_does_not_shrink_below_base(monkeypatch, tmp_path):
    """The medium sub-multiplier must never reduce the stale timeout.

    HERMES_API_CALL_STALE_REASONING_MULTIPLIER=1.0 is the documented way to
    disable scaling. Since medium uses a lower multiplier than high, clamp it
    at 1.0 instead of making the stale detector more aggressive than default.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", "1.0")
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "medium"},
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


# ── fallback to self.reasoning_config when payload missing it ────────────


def test_falls_back_to_reasoning_config_attr_when_payload_lacks_it(monkeypatch, tmp_path):
    """If api_payload has no reasoning field, fall back to self.reasoning_config.

    Some pre-Responses-API paths and synthetic test paths populate the agent
    config but not the per-request payload. We don't want to lose the
    multiplier in that case.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    agent.reasoning_config = {"effort": "high"}  # set on agent, not payload
    payload = {"model": "gpt-5.5", "input": "hi"}
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0 * 5.0


def test_payload_effort_wins_over_agent_attr(monkeypatch, tmp_path):
    """When BOTH payload and self.reasoning_config have effort, payload wins.

    The wire-bound payload is the source of truth: the agent attr may be
    stale or out-of-sync with the actual request being made.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    agent.reasoning_config = {"effort": "high"}  # would give 450
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "low"},  # but payload says low → no mult
    }
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0


# ── context-tier scaling still applies on top of reasoning multiplier ────


def test_long_context_plus_high_effort_compose(monkeypatch, tmp_path):
    """Large prompt (>100k tokens) tier bump AND high-effort multiplier
    both apply. Order: tier bump first (base→240s), then ×5 = 1200s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {
        "model": "gpt-5.5",
        "input": "x" * 500_000,  # ~125k tokens → tier 100k+ → 240s base
        "reasoning": {"effort": "high"},
    }
    # 240.0 * 5.0 = 1200.0
    assert agent._compute_non_stream_stale_timeout(payload) == 1200.0


# ── invalid / malformed payload doesn't crash ────────────────────────────


def test_malformed_reasoning_field_safe(monkeypatch, tmp_path):
    """A non-dict reasoning field should not crash — fall back to base."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_REASONING_MULTIPLIER", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    payload = {"model": "gpt-5.5", "input": "hi", "reasoning": "not-a-dict"}
    assert agent._compute_non_stream_stale_timeout(payload) == 90.0
