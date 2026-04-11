"""Tests for canonical model-selection primitives."""

from __future__ import annotations

import hermes_cli.model_selection as ms


class _Pool:
    def __init__(self, has_credentials: bool):
        self._has_credentials = has_credentials

    def has_credentials(self) -> bool:
        return self._has_credentials


def test_build_tree_groups_openrouter_and_oauth(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(True))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_auth_status", lambda provider_id=None: {"logged_in": provider_id == "minimax"})
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: ["gpt-5.4", "gpt-5.4-mini"])
    monkeypatch.setattr(ms, "OPENROUTER_MODELS", [
        ("openai/gpt-5.4", ""),
        ("openai/gpt-5.4-mini", ""),
        ("anthropic/claude-opus-4.6", ""),
    ])

    tree = ms.build_model_selection_tree(current_provider="openrouter", current_model="openai/gpt-5.4")

    assert [source.label for source in tree.sources] == ["OpenRouter", "OAuth", "Other providers"]
    assert tree.sources[0].status_label == "configured"
    assert [provider.label for provider in tree.providers("openrouter")] == ["OpenAI", "Anthropic"]
    assert [provider.label for provider in tree.providers("oauth")] == ["OpenAI", "Nous", "Qwen"]
    assert any(provider.label == "MiniMax" for provider in tree.providers("other"))
    assert [provider.token for provider in tree.providers("openrouter")] == ["openai", "anthropic"]
    assert [provider.token for provider in tree.providers("oauth")] == ["openai", "nous", "qwen"]


def test_controller_navigates_source_provider_model(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(True))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_auth_status", lambda provider_id=None: {"logged_in": provider_id == "minimax"})
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: ["gpt-5.4", "gpt-5.4-mini"])
    monkeypatch.setattr(ms, "OPENROUTER_MODELS", [
        ("openai/gpt-5.4", ""),
        ("anthropic/claude-opus-4.6", ""),
    ])

    controller = ms.ModelSelectionController(
        ms.build_model_selection_tree(
            current_provider="openrouter",
            current_model="openai/gpt-5.4",
        )
    )

    assert controller.current_view().level == "source"
    assert [item.label for item in controller.current_view().items] == ["OpenRouter", "OAuth", "Other providers"]

    request = controller.enter()
    assert request is None
    assert controller.current_view().level == "provider"
    assert controller.current_view().breadcrumb == "OpenRouter"

    controller.move_down()
    request = controller.enter()
    assert request is None
    assert controller.current_view().level == "model"
    assert controller.current_view().breadcrumb == "OpenRouter / Anthropic"

    request = controller.enter()
    assert request == ms.ModelSwitchRequest(
        provider_slug="openrouter",
        model_id="anthropic/claude-opus-4.6",
    )


def test_controller_back_pops_levels(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(True))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_auth_status", lambda provider_id=None: {"logged_in": provider_id == "minimax"})
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: ["gpt-5.4"])
    monkeypatch.setattr(ms, "OPENROUTER_MODELS", [("openai/gpt-5.4", "")])

    controller = ms.ModelSelectionController(ms.build_model_selection_tree())
    controller.enter()
    controller.enter()

    assert controller.current_view().level == "model"
    assert controller.back() is True
    assert controller.current_view().level == "provider"
    assert controller.back() is True
    assert controller.current_view().level == "source"
    assert controller.back() is False


def test_other_source_exposes_minimax_models(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(False))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_auth_status", lambda provider_id=None: {"logged_in": provider_id == "minimax"})
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: [])

    tree = ms.build_model_selection_tree(current_provider="minimax", current_model="MiniMax-M2.7")
    controller = ms.ModelSelectionController(tree)

    controller.set_source("other")
    provider_labels = [item.label for item in controller.current_view().items]
    assert "MiniMax" in provider_labels

    minimax_provider = next(provider for provider in tree.providers("other") if provider.provider_slug == "minimax")
    controller.set_provider(minimax_provider.id)
    assert controller.current_view().breadcrumb == "Other providers / MiniMax"
    assert any(item.label == "MiniMax M2.7" for item in controller.current_view().items)


def test_other_source_marks_current_provider_when_runtime_provider_is_canonical(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(False))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(
        ms,
        "get_auth_status",
        lambda provider_id=None: {"logged_in": provider_id == "copilot"},
    )
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: [])
    monkeypatch.setattr(ms, "OPENROUTER_MODELS", [])
    monkeypatch.setattr(ms, "_PROVIDER_MODELS", {"copilot": ["claude-sonnet-4.6"]})

    tree = ms.build_model_selection_tree(
        current_provider="github-copilot",
        current_model="claude-sonnet-4.6",
    )

    provider = next(
        provider for provider in tree.providers("other") if provider.provider_slug == "copilot"
    )
    assert provider.current is True

    models = tree.models(provider.id)
    assert len(models) == 1
    assert models[0].current is True


def test_user_provider_uses_model_field_when_default_model_missing(monkeypatch):
    monkeypatch.setattr(ms, "load_pool", lambda provider: _Pool(False))
    monkeypatch.setattr(ms, "get_codex_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_nous_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_qwen_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr(ms, "get_auth_status", lambda provider_id=None: {"logged_in": False})
    monkeypatch.setattr(ms, "get_codex_model_ids", lambda access_token=None: [])
    monkeypatch.setattr(ms, "OPENROUTER_MODELS", [])

    tree = ms.build_model_selection_tree(
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        user_providers={
            "lab-provider": {
                "name": "Lab Provider",
                "api": "http://lab.example/v1",
                "model": "lab-model",
            }
        },
    )

    provider = next(
        provider for provider in tree.providers("other") if provider.provider_slug == "lab-provider"
    )
    models = tree.models(provider.id)
    assert [item.model_id for item in models] == ["lab-model"]
