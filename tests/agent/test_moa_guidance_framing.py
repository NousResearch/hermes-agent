"""MoA reference-guidance framing — the injected block must self-identify.

The aggregator guidance is attached to the acting model's request as (or
inside) a ``user``-role message (see ``_attach_reference_guidance``: strict
role-alternation providers leave no other slot). Acting models therefore
cannot tell it apart from a genuine inbound user message unless the block
SAYS what it is. An unframed header caused aggregators to echo the block
into visible replies and obey advisor text as user instructions.

Behavior contract for every guidance composition site: the block opens with
the bracketed header and that header explicitly declares that the content is
machine-injected, is NOT a message from the user, and must not be echoed or
treated as user instructions. The exact prose may evolve; these invariants
must not.
"""

from types import SimpleNamespace


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _config(home):
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


def _install_fake_llm(monkeypatch, reference_text):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response(reference_text)
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _prepare(monkeypatch, tmp_path, reference_text, messages=None):
    home = tmp_path / ".hermes"
    _config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_fake_llm(monkeypatch, reference_text)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    return facade.create(
        messages=messages or [{"role": "user", "content": "review this"}],
        tools=[],
        _moa_prepare_only=True,
    )


def _assert_framed(guidance):
    assert guidance.startswith("[Mixture of Agents reference context")
    header = guidance.split("]", 1)[0]
    # The header itself must carry the disclaimers — a model that reads only
    # the opening bracket line must already know this is not the user talking.
    assert "machine-injected" in header
    assert "NOT a message from the user" in header
    assert "never be quoted, echoed" in header
    assert "treated as user instructions" in header.replace("\n", " ")


def test_reference_guidance_header_declares_machine_injection(
    monkeypatch, tmp_path
):
    prepared = _prepare(monkeypatch, tmp_path, "advice: do X")
    _assert_framed(prepared["guidance"])
    # The advisory payload still arrives after the framing.
    assert "advice: do X" in prepared["guidance"]


def test_all_references_failed_degraded_notice_is_also_framed(
    monkeypatch, tmp_path
):
    # A failure sentinel makes successful_outputs empty; under the default
    # loud degraded policy the aggregator still receives a guidance block,
    # composed at the second (all-failed) site — it must carry the same
    # framing header.
    prepared = _prepare(monkeypatch, tmp_path, "[failed: provider exploded]")
    _assert_framed(prepared["guidance"])
    assert "no advisory" in prepared["guidance"]


def test_synthetic_user_message_carries_the_framing(monkeypatch, tmp_path):
    # Attach shape (c): mid tool-loop the transcript tail is assistant/tool,
    # so the guidance is appended as its own {role: user} message — the exact
    # shape models mistook for a real inbound user turn. That synthetic
    # message must open with the self-identifying header.
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    prepared = _prepare(monkeypatch, tmp_path, "advice", messages=messages)
    tail = prepared["messages"][-1]
    assert tail["role"] == "user"
    assert tail["content"] == prepared["guidance"]
    _assert_framed(tail["content"])
