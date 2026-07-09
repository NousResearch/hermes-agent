from types import SimpleNamespace
from unittest.mock import patch



def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")



def test_slot_label_includes_reasoning_effort():
    from agent.moa_loop import _slot_label

    assert _slot_label(
        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "xhigh"}
    ) == "openai-codex:gpt-5.6-sol[reasoning=xhigh]"



def test_slot_reasoning_config_parses_effort_and_none():
    from agent.moa_loop import _slot_reasoning_config

    assert _slot_reasoning_config({"reasoning_effort": "high"}) == {
        "enabled": True,
        "effort": "high",
    }
    assert _slot_reasoning_config({"reasoning_effort": "none"}) == {"enabled": False}
    assert _slot_reasoning_config({}) is None



def test_moa_reference_passes_per_slot_reasoning_config(monkeypatch):
    from agent.moa_loop import _run_reference

    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("advice")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve:
        mock_resolve.return_value = {"provider": "openai-codex", "model": "gpt-5.6-sol"}
        _run_reference(
            {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "low"},
            [{"role": "user", "content": "judge this"}],
        )

    assert captured["reasoning_config"] == {"enabled": True, "effort": "low"}



def test_call_llm_builder_translates_reasoning_config_to_extra_body():
    from agent.auxiliary_client import _build_call_kwargs

    kwargs = _build_call_kwargs(
        "openai-codex",
        "gpt-5.6-sol",
        [{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )
    assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "xhigh"}

    off = _build_call_kwargs(
        "openai-codex",
        "gpt-5.6-sol",
        [{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": False},
    )
    assert off["extra_body"]["reasoning"] == {"enabled": False}
