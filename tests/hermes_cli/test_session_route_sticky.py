"""Sticky route identity from SessionDB rows (CLI resume / R5)."""

from hermes_cli.session_route import model_config_for_route, route_from_session_row


def test_route_from_session_row_reads_model_and_provider():
    row = {
        "model": "user-preset-alpha",
        "model_config": {
            "provider": "moa",
            "model": "user-preset-alpha",
            "api_mode": "chat_completions",
        },
    }
    assert route_from_session_row(row) == {
        "model": "user-preset-alpha",
        "provider": "moa",
        "api_mode": "chat_completions",
    }


def test_route_from_session_row_parses_json_model_config():
    import json

    row = {
        "model": "opaque-model-a",
        "model_config": json.dumps({"provider": "provider-alpha", "base_url": "https://example.test"}),
    }
    route = route_from_session_row(row)
    assert route["model"] == "opaque-model-a"
    assert route["provider"] == "provider-alpha"
    assert route["base_url"] == "https://example.test"


def test_route_from_session_row_skips_bare_custom_without_base_url():
    row = {
        "model": "opaque-model-a",
        "model_config": {"provider": "custom"},
    }
    route = route_from_session_row(row)
    assert route == {"model": "opaque-model-a"}


def test_model_config_for_route_includes_provider():
    cfg = model_config_for_route(
        model="user-preset-alpha",
        provider="moa",
        extra={"max_iterations": 5},
    )
    assert cfg["model"] == "user-preset-alpha"
    assert cfg["provider"] == "moa"
    assert cfg["max_iterations"] == 5


def test_apply_session_route_from_meta_sets_cli_fields():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class _Stub(CLIAgentSetupMixin):
        def __init__(self):
            self.model = "global-default"
            self.provider = "openrouter"
            self.requested_provider = "openrouter"
            self.base_url = ""
            self.api_mode = ""

    stub = _Stub()
    stub._apply_session_route_from_meta(
        {
            "model": "user-preset-alpha",
            "model_config": {"provider": "moa", "api_mode": "chat_completions"},
        }
    )
    assert stub.model == "user-preset-alpha"
    assert stub.provider == "moa"
    assert stub.requested_provider == "moa"
    assert stub.api_mode == "chat_completions"
