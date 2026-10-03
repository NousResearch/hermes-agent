"""/status must name the precedence slot that won the model route (#122016).

A stale session override shadowing an explicit config.yaml route is only diagnosable if the
winning source is reported instead of silently switching providers.
"""

from types import SimpleNamespace

from gateway.slash_commands_status import _status_model_route


def _agent(model="gpt-5.6-luna", provider="openai-codex"):
    return SimpleNamespace(model=model, provider=provider, base_url="", api_key="",
                           context_compressor=None)


def _entry():
    return SimpleNamespace(last_prompt_tokens=0)


def test_live_agent_route_names_agent_source():
    result = _status_model_route(_agent(), {}, {}, {}, _entry())
    assert result[5] == "agent"
    assert (result[0], result[1]) == ("gpt-5.6-luna", "openai-codex")


def test_session_override_names_session_override_source():
    result = _status_model_route(
        None, {"model": "deepseek-flash", "provider": "deepseek"}, {}, {}, _entry())
    assert result[5] == "session override"
    assert (result[0], result[1]) == ("deepseek-flash", "deepseek")


def test_persisted_route_names_persisted_route_source():
    result = _status_model_route(
        None, {}, {"model": "deepseek-flash", "billing_provider": "deepseek"}, {}, _entry())
    assert result[5] == "persisted route"


def test_session_row_names_config_source_when_row_partial():
    # Row with only a model: the provider comes from config, so config is the winning source.
    result = _status_model_route(None, {}, {}, {"model": "gpt-5.6-luna"}, _entry())
    assert result[5] == "config"


def test_no_route_anywhere_names_config_source():
    result = _status_model_route(None, {}, {}, {}, _entry())
    assert result[5] == "config"
