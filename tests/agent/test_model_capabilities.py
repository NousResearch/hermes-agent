from agent.model_capabilities import ModelCapability, resolve_model_capability


def test_resolve_model_capability_reads_provider_model_scores():
    cfg = {
        "providers": {
            "local-llm": {
                "base_url": "http://127.0.0.1:9130/v1",
                "model": "qwen3.6-27b-q4km",
                "context_length": 8192,
                "models": {
                    "qwen3.6-27b-q4km": {
                        "context_length": 8192,
                        "supports_tools": True,
                        "supports_vision": True,
                        "scores": {
                            "general": 0.55,
                            "coding": 0.45,
                            "autonomous_ops": 0.40,
                        },
                        "limits": {"max_safe_tool_iterations": 8},
                        "degraded_label": "local-qwen",
                    }
                },
            }
        }
    }

    cap = resolve_model_capability(
        provider="local-llm",
        model="qwen3.6-27b-q4km",
        config=cfg,
    )

    assert isinstance(cap, ModelCapability)
    assert cap.provider == "local-llm"
    assert cap.model == "qwen3.6-27b-q4km"
    assert cap.context_tokens == 8192
    assert cap.supports_tools is True
    assert cap.supports_vision is True
    assert cap.score("coding") == 0.45
    assert cap.max_safe_tool_iterations == 8
    assert cap.degraded_label == "local-qwen"


def test_resolve_model_capability_uses_conservative_defaults_for_unknown_model():
    cap = resolve_model_capability(provider="unknown-provider", model="mystery", config={})

    assert cap.score("general") == 0.35
    assert cap.score("coding") == 0.25
    assert cap.context_tokens is None
    assert cap.supports_tools is None
    assert cap.degraded_label == "unknown"


def test_resolve_model_capability_treats_gpt55_as_baseline_without_metadata():
    cap = resolve_model_capability(provider="openai-codex", model="gpt-5.5", config={})

    assert cap.score("general") == 0.55
    assert cap.score("coding") == 0.55
    assert cap.score("system_debugging") == 0.55
    assert cap.score("long_context") == 0.55
    assert cap.score("autonomous_ops") == 0.55


def test_resolve_model_capability_normalizes_human_scale_scores():
    cfg = {
        "providers": {
            "openai-codex": {
                "models": {
                    "gpt-5.5": {
                        "scores": {"general": 55, "autonomous_ops": "55"},
                    }
                }
            }
        }
    }

    cap = resolve_model_capability(provider="openai-codex", model="gpt-5.5", config=cfg)

    assert cap.score("general") == 0.55
    assert cap.score("autonomous_ops") == 0.55
