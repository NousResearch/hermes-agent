from agent.cursor.model_discovery import _parse_usable_models_payload
from agent.cursor.proto import agent_pb2


def test_parse_usable_models_payload_deduplicates_and_sorts() -> None:
    response = agent_pb2.GetUsableModelsResponse(
        models=[
            agent_pb2.ModelDetails(model_id="gpt-5.4-medium", display_name="GPT-5.4"),
            agent_pb2.ModelDetails(model_id="composer-2.5", display_name="Composer 2.5"),
            agent_pb2.ModelDetails(model_id="cursor-composer-2.5", display_name="Legacy"),
            agent_pb2.ModelDetails(model_id="default", display_name="Auto"),
        ]
    )
    assert _parse_usable_models_payload(response.SerializeToString()) == [
        "composer-2.5",
        "default",
        "gpt-5.4-medium",
    ]


def test_provider_model_ids_cursor_uses_live_catalog(monkeypatch) -> None:
    from hermes_cli import models as models_module

    class _Profile:
        def fetch_models(self, *, api_key=None, timeout=12.0):
            assert api_key == "cursor-token"
            return ["default", "composer-2.5", "claude-4.6-opus-high"]

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_cursor_runtime_credentials",
        lambda **kwargs: {"api_key": "cursor-token", "base_url": "https://api2.cursor.sh"},
    )
    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: _Profile() if name == "cursor" else None,
    )

    live = models_module.provider_model_ids("cursor", force_refresh=True)
    assert live == ["default", "composer-2.5", "claude-4.6-opus-high"]
