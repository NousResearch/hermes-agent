from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def test_moa_virtual_provider_aggregator_is_actor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
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
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "moa_reference":
            return _response("reference advice")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )

    result = agent.run_conversation("solve this")

    assert result["final_response"] == "aggregator acted"
    assert [(c["task"], c["provider"], c["model"]) for c in calls] == [
        ("moa_reference", "openai-codex", "gpt-5.5"),
        ("moa_aggregator", "openrouter", "anthropic/claude-opus-4.8"),
    ]
    assert calls[1]["tools"] is not None


def test_reference_messages_strips_system_and_tool_history():
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "system", "content": "huge hermes system prompt"},
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
        {"role": "assistant", "content": "here is my answer"},
    ]

    trimmed = _reference_messages(messages)

    # System prompt, tool-call-only assistant turn, and tool result are gone.
    assert all(m["role"] in ("user", "assistant") for m in trimmed)
    assert all("tool_calls" not in m for m in trimmed)
    assert trimmed == [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "here is my answer"},
    ]


def test_reference_messages_can_include_full_system_context():
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "system", "content": "system instructions for references"},
        {"role": "user", "content": "do the thing"},
    ]

    ref_messages = _reference_messages(
        messages,
        reference_context={"system": "full", "files": {"enabled": False, "names": []}},
    )

    assert ref_messages == [
        {"role": "system", "content": "system instructions for references"},
        {"role": "user", "content": "do the thing"},
    ]


def test_reference_messages_can_include_selected_context_files(monkeypatch, tmp_path):
    from agent.moa_loop import _reference_messages

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "SOUL.md").write_text("Persona rules", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("Project workflow rules", encoding="utf-8")
    (project / "CLAUDE.md").write_text("Not selected", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_messages = _reference_messages(
        [{"role": "user", "content": "review this"}],
        reference_context={
            "system": "none",
            "files": {"enabled": True, "names": ["SOUL.md", "AGENTS.md"]},
        },
        cwd=str(project),
    )

    assert ref_messages[0]["role"] == "system"
    assert "[Mixture of Agents selected reference context files]" in ref_messages[0]["content"]
    assert "## SOUL.md" in ref_messages[0]["content"]
    assert "Persona rules" in ref_messages[0]["content"]
    assert "## AGENTS.md" in ref_messages[0]["content"]
    assert "Project workflow rules" in ref_messages[0]["content"]
    assert "Not selected" not in ref_messages[0]["content"]
    assert ref_messages[-1] == {"role": "user", "content": "review this"}




def test_reference_messages_use_runtime_context_cwd(monkeypatch, tmp_path):
    from agent.moa_loop import _reference_messages

    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("Runtime project workflow rules", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    ref_messages = _reference_messages(
        [{"role": "user", "content": "review this"}],
        reference_context={
            "system": "none",
            "files": {"enabled": True, "names": ["AGENTS.md"]},
        },
    )

    assert ref_messages[0]["role"] == "system"
    assert "Runtime project workflow rules" in ref_messages[0]["content"]


def test_reference_messages_cursorrules_is_exact_file_only(tmp_path):
    from agent.moa_loop import _reference_messages

    project = tmp_path / "project"
    project.mkdir()
    (project / ".cursorrules").write_text("Exact cursor rules", encoding="utf-8")
    extra = project / ".cursor" / "rules"
    extra.mkdir(parents=True)
    (extra / "extra.mdc").write_text("Extra cursor rule should not leak", encoding="utf-8")

    ref_messages = _reference_messages(
        [{"role": "user", "content": "review this"}],
        reference_context={
            "system": "none",
            "files": {"enabled": True, "names": [".cursorrules"]},
        },
        cwd=str(project),
    )

    assert "Exact cursor rules" in ref_messages[0]["content"]
    assert "Extra cursor rule should not leak" not in ref_messages[0]["content"]


def test_moa_facade_references_get_trimmed_messages(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
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
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("ok")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "question"},
            {"role": "tool", "tool_call_id": "x", "content": "leftover"},
        ],
        tools=[{"type": "function"}],
    )

    ref_call = next(c for c in calls if c["task"] == "moa_reference")
    # Reference never sees system prompt or tool-role messages.
    assert all(m["role"] == "user" for m in ref_call["messages"])
    assert ref_call.get("tools") in (None, [])
    # Aggregator still receives the original messages + tool schema.
    agg_call = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg_call["tools"] is not None


def test_moa_disabled_preset_skips_references(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      enabled: false
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("aggregator only")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "question"}], tools=[{"type": "function"}])

    tasks = [c["task"] for c in calls]
    # No reference fan-out — only the aggregator runs.
    assert tasks == ["moa_aggregator"]
    # Aggregator gets the unmodified user message (no MoA guidance appended).
    agg_call = calls[0]
    assert agg_call["messages"][-1]["content"] == "question"


def test_references_run_in_parallel(monkeypatch):
    """References fan out concurrently (delegate-batch semantics), not serially.

    Each reference sleeps; wall-time must approximate the slowest single call,
    not the sum. Order is preserved and a failing reference is isolated.
    """
    import time

    from agent import moa_loop

    # Force _extract_text down its fallback path (no transport normalize).
    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)

    barrier_hits = []

    def slow_call_llm(**kwargs):
        barrier_hits.append(time.monotonic())
        model = kwargs["model"]
        if model == "boom":
            raise RuntimeError("kaboom")
        time.sleep(0.5)
        return _response(f"resp-{kwargs['provider']}")

    monkeypatch.setattr(moa_loop, "call_llm", slow_call_llm)

    refs = [
        {"provider": "p1", "model": "ok"},
        {"provider": "moa", "model": "preset"},  # recursion guard, not dispatched
        {"provider": "p2", "model": "boom"},  # failure isolated
        {"provider": "p3", "model": "ok"},
    ]

    start = time.monotonic()
    out = moa_loop._run_references_parallel(
        refs, [{"role": "user", "content": "hi"}], temperature=0.6, max_tokens=64
    )
    elapsed = time.monotonic() - start

    # Two 0.5s sleeps run concurrently → well under the 1.0s serial floor.
    assert elapsed < 0.9, f"references did not run in parallel (took {elapsed:.2f}s)"
    # Output order matches input order (stable Reference N labelling).
    assert [label for label, _ in out] == ["p1:ok", "moa:preset", "p2:boom", "p3:ok"]
    assert "recursively reference MoA" in out[1][1]
    assert out[2][1].startswith("[failed:")
    assert out[0][1] == "resp-p1"

