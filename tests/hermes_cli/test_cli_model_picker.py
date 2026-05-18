"""Regression tests for the prompt_toolkit /model picker."""

from types import SimpleNamespace
from unittest.mock import patch

from cli import HermesCLI


class _DummyCLI(SimpleNamespace):
    def _invalidate(self, min_interval=0.0):
        self.invalidated = True

    def _close_model_picker(self):  # pragma: no cover - safety if a test misroutes
        self.closed = True


def test_provider_selection_hydrates_truncated_model_list():
    """Entering a provider should fetch the full list when the row was truncated.

    build_models_payload(..., max_models=50) deliberately sends only the first N
    models to keep the provider picker cheap, while preserving the full count in
    total_models.  The second-stage picker must not treat that truncated slice as
    the complete selectable catalog.
    """
    cli = _DummyCLI(
        _model_picker_state={
            "stage": "provider",
            "selected": 0,
            "providers": [
                {
                    "slug": "litellm",
                    "name": "LiteLLM Proxy",
                    "models": ["model-a", "model-b"],
                    "total_models": 4,
                }
            ],
        }
    )

    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["model-a", "model-b", "model-c", "model-d"],
    ) as live_fetch:
        HermesCLI._handle_model_picker_selection(cli)

    live_fetch.assert_called_once_with("litellm")
    state = cli._model_picker_state
    assert state["stage"] == "model"
    assert state["model_list"] == ["model-a", "model-b", "model-c", "model-d"]
    assert state["provider_data"]["slug"] == "litellm"
    assert state["selected"] == 0
    assert cli.invalidated is True


def test_model_picker_choice_wrap_preserves_unselected_indent_for_long_model_id():
    """Long slash-delimited model IDs must not lose the two-space row indent."""
    long_model = "siliconflow-cn/Pro/deepseek-ai/DeepSeek-V3.1-Terminus"

    wrapped = HermesCLI._wrap_model_picker_choice(long_model, "  ", width=40)

    assert wrapped == ["  " + long_model]


def test_model_picker_choice_wrap_preserves_selected_prefix():
    """Selected rows keep the arrow prefix while wrapping the raw model ID."""
    wrapped = HermesCLI._wrap_model_picker_choice("model with spaces here", "❯ ", width=14)

    assert wrapped[0].startswith("❯ ")
    assert all(line.startswith(("❯ ", "  ")) for line in wrapped)


def test_provider_selection_keeps_complete_curated_model_list_without_live_fetch():
    """Complete curated rows should not be overwritten by provider_model_ids()."""
    cli = _DummyCLI(
        _model_picker_state={
            "stage": "provider",
            "selected": 0,
            "providers": [
                {
                    "slug": "nous",
                    "name": "Nous",
                    "models": ["agentic-a", "agentic-b"],
                    "total_models": 2,
                }
            ],
        }
    )

    with patch("hermes_cli.models.provider_model_ids") as live_fetch:
        HermesCLI._handle_model_picker_selection(cli)

    live_fetch.assert_not_called()
    assert cli._model_picker_state["stage"] == "model"
    assert cli._model_picker_state["model_list"] == ["agentic-a", "agentic-b"]
