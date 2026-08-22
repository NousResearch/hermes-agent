"""Internal harness forks suppress the MoA advisor fan-out.

Background review / curator forks inherit provider="moa" from the parent
agent but run harness-generated prompts, not the user's task. Advisor
fan-out on those turns burned ~100K+ tokens per housekeeping turn, and
advisors sometimes answered the harness prompt instead of the task
(observed in traces). The fork sets ``_moa_suppress_references`` and the
facade runs the aggregator alone — same path as a disabled preset.
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


def _install_fake_llm(monkeypatch, ref_runs):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def test_suppress_flag_skips_reference_fanout(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    fork_agent = SimpleNamespace(_moa_suppress_references=True)
    facade = MoAChatCompletions("review", agent=fork_agent)
    prepared = facade.create(
        messages=[{"role": "user", "content": "Review the conversation above and update the skill library."}],
        tools=[],
        _moa_prepare_only=True,
    )

    assert ref_runs == [], "advisors must not run for a suppressed fork"
    assert not prepared["guidance"], "no advisor guidance without a fan-out"


def test_unsuppressed_agent_still_fans_out(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review", agent=SimpleNamespace())
    facade.create(
        messages=[{"role": "user", "content": "a real user task"}],
        tools=[],
        _moa_prepare_only=True,
    )

    assert len(ref_runs) == 1


def _module_assigns_suppress_flag(module) -> bool:
    """True when the module contains a real `<x>._moa_suppress_references = True`
    assignment. AST-based: a commented-out line does not count (the previous
    source-string version passed even with the assignment commented out)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "_moa_suppress_references"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                return True
    return False


def test_background_review_fork_sets_suppress_flag():
    """The review fork construction pins the flag."""
    import agent.background_review as background_review

    assert _module_assigns_suppress_flag(background_review)


def test_curator_fork_sets_suppress_flag():
    import agent.curator as curator

    assert _module_assigns_suppress_flag(curator)
