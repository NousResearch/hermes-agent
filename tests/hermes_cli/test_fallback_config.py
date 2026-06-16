from hermes_cli.fallback_config import get_fallback_chain


def test_fallback_order_precedes_older_fallback_keys_and_dedupes():
    cfg = {
        "fallback_order": [
            {"provider": "zai", "model": "glm-5.2", "base_url": "https://api.z.ai/api/coding/paas/v4/"},
            {"provider": "xiaomi", "model": "mimo-v2.5-pro"},
        ],
        "fallback_providers": [
            {"provider": "zai", "model": "glm-5.2", "base_url": "https://api.z.ai/api/coding/paas/v4"},
            {"provider": "minimax", "model": "MiniMax-M2.7"},
        ],
        "fallback_model": {"provider": "omlx-qwen8811", "model": "Qwen3.6-35B"},
    }

    chain = get_fallback_chain(cfg)

    assert [(entry["provider"], entry["model"]) for entry in chain] == [
        ("zai", "glm-5.2"),
        ("xiaomi", "mimo-v2.5-pro"),
        ("minimax", "MiniMax-M2.7"),
        ("omlx-qwen8811", "Qwen3.6-35B"),
    ]
    assert chain[0]["base_url"] == "https://api.z.ai/api/coding/paas/v4"


def test_fallback_order_ignores_malformed_entries():
    cfg = {
        "fallback_order": [
            "bad",
            {"provider": "zai"},
            {"model": "glm-5.2"},
            {"provider": "zai", "model": "glm-5.2"},
        ]
    }

    assert get_fallback_chain(cfg) == [{"provider": "zai", "model": "glm-5.2"}]