"""Profile verbosity -> main-agent Responses request (no network)."""
import copy
import pytest
import run_agent
import model_tools
from agent.transports import get_transport


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_profile_config_reaches_responses(monkeypatch, level):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"verbosity": level}})
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr(model_tools, "check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model="gpt-6-astra", provider="custom", api_mode="codex_responses",
        base_url="http://127.0.0.1:8317/v1", api_key="test-only", quiet_mode=True,
        skip_context_files=True, skip_memory=True, reasoning_config={"effort": "high"},
    )
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hello"}], tools_for_api=[])
    kwargs = agent._get_transport().preflight_kwargs(kwargs)
    assert kwargs["text"]["verbosity"] == level
    assert kwargs["reasoning"]["effort"] == "high"


@pytest.mark.parametrize("level", [None, "", "maximum", 5, {"bad": True}])
def test_unset_invalid_omitted(level):
    kw = get_transport("codex_responses").build_kwargs(model="gpt-6-astra", messages=[], verbosity=level)
    assert "text" not in kw


@pytest.mark.parametrize(
    "model,flags",
    [("grok-4", {}), ("gpt-4.1", {}), ("gpt-6-astra", {"is_xai_responses": True}),
     ("gpt-6-astra", {"is_github_responses": True})],
)
def test_other_routes_unchanged(model, flags):
    kw = get_transport("codex_responses").build_kwargs(model=model, messages=[], verbosity="low", **flags)
    assert "text" not in kw


@pytest.mark.parametrize(
    "overrides",
    [{"text": {"format": {"type": "json_object"}}},
     {"text": {"verbosity": "high", "format": {"type": "json_object"}}},
     {"extra_body": {"text": {"verbosity": "medium"}}}],
)
def test_overrides_and_format_preserved(overrides):
    before = copy.deepcopy(overrides)
    kw = get_transport("codex_responses").build_kwargs(
        model="gpt-6-astra", messages=[], verbosity="low", request_overrides=overrides,
    )
    assert overrides == before
    if "extra_body" in overrides:
        assert kw["extra_body"] == overrides["extra_body"]
    else:
        assert kw["text"]["format"] == overrides["text"]["format"]
        assert kw["text"]["verbosity"] == overrides["text"].get("verbosity", "low")


def test_config_key_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert "verbosity" in DEFAULT_CONFIG["agent"]
    assert DEFAULT_CONFIG["agent"]["verbosity"] is None
    kw = get_transport("codex_responses").build_kwargs(model="gpt-5.6-luna", messages=[], verbosity="low")
    assert kw["text"]["verbosity"] == "low"


def test_preflight_preserves_text_verbosity_and_format():
    transport = get_transport("codex_responses")
    kw = transport.build_kwargs(
        model="gpt-6-astra", messages=[], verbosity="low",
        request_overrides={"text": {"format": {"type": "text"}}},
    )
    checked = transport.preflight_kwargs(kw)
    assert checked["text"] == {"format": {"type": "text"}, "verbosity": "low"}
