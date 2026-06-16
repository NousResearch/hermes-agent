"""Config ``providers:`` models must augment canonical picker rows."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def test_user_provider_models_merge_into_existing_canonical_row():
    """opencode-go is emitted by HERMES_OVERLAYS before section 3; explicit
    ``providers.opencode-go.models`` must still surface in the picker."""
    live_go_models = [
        "qwen3.7-max",
        "mimo-v2.5",
        "hy3-preview",
    ]
    user_providers = {
        "opencode-go": {
            "base_url": "https://opencode.ai/zen/go/v1",
            "key_env": "OPENCODE_GO_API_KEY",
            "default_model": "minimax-m3",
            "models": {
                "minimax-m3": {"context_length": 131072},
                "minimax-m2.5": {"context_length": 131072},
                "kimi-k2.5": {"context_length": 262144},
            },
        }
    }

    with (
        patch.dict("os.environ", {"OPENCODE_GO_API_KEY": "sk-test"}, clear=False),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=live_go_models,
        ),
    ):
        rows = list_authenticated_providers(
            user_providers=user_providers,
            max_models=50,
        )

    go_row = next(r for r in rows if r["slug"] == "opencode-go")
    assert "minimax-m3" in go_row["models"]
    assert "kimi-k2.5" in go_row["models"]
    assert "qwen3.7-max" in go_row["models"]
    assert go_row["total_models"] >= 5
