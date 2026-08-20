"""MoA privacy redaction filter (config: moa.privacy_filter — display | full).

Reworked from PR #60463 (issue #59959). Secret/credential shapes are handled
by the central redactor (agent.redact.redact_sensitive_text); the MoA filter
adds conservative email + formatted-phone patterns and wires two modes:

  display — redact user-visible surfaces only (reference display events +
            saved MoA traces); the aggregator sees raw advisor text.
  full    — additionally redact the advisor text injected into the
            aggregator prompt (issue #59959's literal ask).
  (off)   — default: everything passes through raw.
"""

from types import SimpleNamespace

from agent.moa_loop import _redact_reference_text


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


# --- pattern behavior -------------------------------------------------------


def test_redacts_email_addresses():
    out = _redact_reference_text("contact jane.doe+dev@example.co.uk for access")
    assert "jane.doe" not in out
    assert "[redacted email]" in out






def test_does_not_mangle_code_review_shaped_text():
    """Advisory text is often code-review shaped. Line numbers, timestamps,
    bare digit runs, git SHAs, IPs, versions, and source-code assignments must
    survive redaction byte-identical — a false positive here corrupts the
    guidance the aggregator acts on."""
    text = (
        "Line 1274: off-by-one at src/moa_loop.py:1274\n"
        "commit 3428e70c599cbfbe240172b4ee1118dbc18a78ca\n"
        "Date: 2026-07-12 12:34:56 +0000\n"
        "epoch 1234567890, id 5551234567, port 127.0.0.1:8080\n"
        "version 1.2.3-4567\n"
        "MAX_TOKENS=4096\n"
        '"apiKey": "test-fixture"\n'
        "timeout = 1234567890\n"
    )
    assert _redact_reference_text(text) == text




def test_non_string_input_passes_through():
    assert _redact_reference_text(None) is None
    assert _redact_reference_text(42) == 42
    assert _redact_reference_text("") == ""


# --- mode wiring ------------------------------------------------------------


SENSITIVE_ADVICE = "email ceo@example.com, phone (555) 867-5309, proceed"


def _privacy_config(home, privacy_filter):
    home.mkdir()
    line = f"  privacy_filter: {privacy_filter}\n" if privacy_filter is not None else ""
    (home / "config.yaml").write_text(
        f"""
moa:
{line}  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _install_fake_llm(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "moa_reference":
            return _response(SENSITIVE_ADVICE)
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    return calls


def _run_facade(monkeypatch, tmp_path, privacy_filter):
    home = tmp_path / ".hermes"
    _privacy_config(home, privacy_filter)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_fake_llm(monkeypatch)

    from agent.moa_loop import MoAChatCompletions

    events = []
    facade = MoAChatCompletions(
        "review", reference_callback=lambda ev, **kw: events.append((ev, kw))
    )
    prepared = facade.create(
        messages=[{"role": "user", "content": "review this"}],
        tools=[],
        _moa_prepare_only=True,
    )
    ref_events = [kw for ev, kw in events if ev == "moa.reference"]
    return prepared, ref_events, facade


def test_privacy_off_by_default_everything_raw(monkeypatch, tmp_path):
    prepared, ref_events, _ = _run_facade(monkeypatch, tmp_path, None)
    assert "ceo@example.com" in prepared["guidance"]
    assert "ceo@example.com" in ref_events[0]["text"]


def test_display_mode_redacts_display_but_not_aggregator(monkeypatch, tmp_path):
    prepared, ref_events, facade = _run_facade(monkeypatch, tmp_path, "display")
    # User-visible reference block: redacted.
    assert "ceo@example.com" not in ref_events[0]["text"]
    assert "[redacted email]" in ref_events[0]["text"]
    assert "(555) 867-5309" not in ref_events[0]["text"]
    # Aggregator input: raw (synthesis quality unaffected).
    assert "ceo@example.com" in prepared["guidance"]
    # Pending trace surface: redacted (traces persist to disk).
    trace_refs = facade._pending_trace["reference_outputs"]
    assert all("ceo@example.com" not in text for _l, text, _a in trace_refs)


def test_full_mode_redacts_aggregator_input_too(monkeypatch, tmp_path):
    prepared, ref_events, _ = _run_facade(monkeypatch, tmp_path, "full")
    assert "ceo@example.com" not in prepared["guidance"]
    assert "[redacted email]" in prepared["guidance"]
    assert "(555) 867-5309" not in prepared["guidance"]
    assert "ceo@example.com" not in ref_events[0]["text"]
    # The redacted guidance is what reaches the aggregator request messages.
    joined_content = "".join(
        str(m.get("content")) for m in prepared["messages"]
    )
    assert "ceo@example.com" not in joined_content




def test_cache_keeps_raw_text_redaction_applied_per_surface(monkeypatch, tmp_path):
    """The reference cache must hold RAW advisor text — redaction happens at
    each consuming surface. Otherwise a mid-session mode change would leak
    (cache pre-redacted with weaker mode) or double-redact."""
    _prepared, _ref_events, facade = _run_facade(monkeypatch, tmp_path, "full")
    assert any("ceo@example.com" in text for _l, text, _a in facade._ref_cache_outputs)


def test_full_mode_covers_one_shot_aggregate_moa_context(monkeypatch, tmp_path):
    """The /moa one-shot synthesis path also honors full mode."""
    home = tmp_path / ".hermes"
    _privacy_config(home, "full")
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = _install_fake_llm(monkeypatch)

    from agent.moa_loop import aggregate_moa_context

    aggregate_moa_context(
        user_prompt="review this",
        api_messages=[{"role": "user", "content": "review this"}],
        reference_models=[{"provider": "openai-codex", "model": "gpt-5.5"}],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    )

    agg_calls = [c for c in calls if c["task"] == "moa_aggregator"]
    assert agg_calls, "aggregator synthesis call expected"
    agg_input = str(agg_calls[0]["messages"])
    assert "ceo@example.com" not in agg_input
    assert "[redacted email]" in agg_input


def test_one_shot_input_fallback_resolves_named_default_with_one_config_read(
    monkeypatch,
):
    """Older direct callers omit the input kwargs; resolve the named default
    preset, and share that config read with the output privacy check."""
    calls = _install_fake_llm(monkeypatch)
    loads = 0

    def load_config():
        nonlocal loads
        loads += 1
        return {
            "moa": {
                "default_preset": "private",
                "presets": {
                    "private": {
                        "reference_input_scope": "current_turn",
                        "reference_input_filter": "redact",
                        "reference_models": [
                            {"provider": "openai-codex", "model": "gpt-5.5"}
                        ],
                        "aggregator": {
                            "provider": "openrouter",
                            "model": "anthropic/claude-opus-4.8",
                        },
                    }
                },
            }
        }

    monkeypatch.setattr("hermes_cli.config.load_config", load_config)
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )
    # Isolate this boundary from the aggregator reasoning helper, which has its
    # own pre-existing config read unrelated to MoA input/privacy resolution.
    monkeypatch.setattr("agent.moa_loop._aggregator_reasoning_config", lambda _slot: None)

    from agent.moa_loop import aggregate_moa_context

    aggregate_moa_context(
        user_prompt="email alice@corp.example",
        api_messages=[
            {"role": "user", "content": "old private history"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "email alice@corp.example"},
        ],
        reference_models=[{"provider": "openai-codex", "model": "gpt-5.5"}],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    )

    ref_call = next(call for call in calls if call["task"] == "moa_reference")
    payload = "".join(str(message.get("content") or "") for message in ref_call["messages"])
    assert loads == 1
    assert "old private history" not in payload
    assert "alice@corp.example" not in payload
    assert "email [redacted email]" in payload
