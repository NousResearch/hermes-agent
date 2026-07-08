from hermes_cli.router_config import (
    DEFAULT_ROUTER_CLASSIFIER,
    DEFAULT_ROUTER_FALLBACKS,
    DEFAULT_ROUTER_PRESET_NAME,
    DEFAULT_ROUTER_ROUTES,
    exact_router_preset_name,
    list_router_presets,
    normalize_router_config,
    resolve_router_preset,
    set_active_router_preset,
)


def test_normalize_router_config_uses_default_named_preset():
    cfg = normalize_router_config({})

    assert cfg["default_preset"] == DEFAULT_ROUTER_PRESET_NAME
    assert list(cfg["presets"]) == [DEFAULT_ROUTER_PRESET_NAME]
    assert cfg["classifier"] == DEFAULT_ROUTER_CLASSIFIER
    assert cfg["routes"] == DEFAULT_ROUTER_ROUTES
    assert cfg["fallbacks"] == DEFAULT_ROUTER_FALLBACKS


def test_normalize_router_config_preserves_named_presets():
    cfg = normalize_router_config(
        {
            "default_preset": "whatsapp",
            "presets": {
                "whatsapp": {
                    "classifier": {"provider": "openai-codex", "model": "gpt-5.5"},
                    "routes": {
                        "simple": {"provider": "lmstudio", "model": "google/gemma-4-e4b"},
                        "complex": {"provider": "openai-codex", "model": "gpt-5.5"},
                    },
                },
                "cheap": {
                    "classifier": {"provider": "lmstudio", "model": "qwen/qwen3-4b-thinking-2507"},
                },
            },
        }
    )

    assert cfg["default_preset"] == "whatsapp"
    assert set(cfg["presets"]) == {"whatsapp", "cheap"}
    assert cfg["presets"]["cheap"]["classifier"]["provider"] == "lmstudio"
    # Missing routes on "cheap" degrade to defaults — both tiers always exist.
    assert set(cfg["presets"]["cheap"]["routes"]) == {"simple", "complex"}


def test_routes_always_exist_even_when_broken():
    cfg = normalize_router_config({"presets": {"x": {"routes": "garbage"}}})
    preset = cfg["presets"]["x"]
    assert preset["routes"]["simple"]["model"]
    assert preset["routes"]["complex"]["model"]


def test_recursion_guard_rejects_router_and_moa_slots():
    cfg = normalize_router_config(
        {
            "presets": {
                "x": {
                    "classifier": {"provider": "router", "model": "default"},
                    "routes": {
                        "simple": {"provider": "moa", "model": "default"},
                        "complex": {"provider": "openai-codex", "model": "gpt-5.5"},
                    },
                    "fallbacks": [
                        {"provider": "router", "model": "default"},
                        {"provider": "lmstudio", "model": "qwen/qwen3-4b-thinking-2507"},
                    ],
                }
            }
        }
    )
    preset = cfg["presets"]["x"]
    # Virtual-provider slots are dropped: classifier/simple fall back to
    # defaults, the router fallback entry disappears from the chain.
    assert preset["classifier"] == DEFAULT_ROUTER_CLASSIFIER
    assert preset["routes"]["simple"] == DEFAULT_ROUTER_ROUTES["simple"]
    assert preset["routes"]["complex"] == {"provider": "openai-codex", "model": "gpt-5.5"}
    assert preset["fallbacks"] == [
        {"provider": "lmstudio", "model": "qwen/qwen3-4b-thinking-2507"}
    ]


def test_moa_recursion_guard_rejects_router_slots():
    from hermes_cli.moa_config import normalize_moa_config

    cfg = normalize_moa_config(
        {
            "reference_models": [
                {"provider": "router", "model": "default"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ],
        }
    )
    labels = [(s["provider"], s["model"]) for s in cfg["reference_models"]]
    assert ("router", "default") not in labels
    assert ("openai-codex", "gpt-5.5") in labels


def test_explicit_empty_fallbacks_mean_no_fallbacks():
    cfg = normalize_router_config({"presets": {"x": {"fallbacks": []}}})
    assert cfg["presets"]["x"]["fallbacks"] == []


def test_missing_fallbacks_get_defaults():
    cfg = normalize_router_config({"presets": {"x": {}}})
    assert cfg["presets"]["x"]["fallbacks"] == DEFAULT_ROUTER_FALLBACKS


def test_default_route_coercion():
    assert normalize_router_config({"default_route": "complex"})["default_route"] == "complex"
    assert normalize_router_config({"default_route": "banana"})["default_route"] == "simple"
    assert normalize_router_config({})["default_route"] == "simple"


def test_channel_hints_validated():
    cfg = normalize_router_config(
        {"channel_hints": {"whatsapp": "simple", "telegram": "banana", "": "simple"}}
    )
    assert cfg["channel_hints"] == {"whatsapp": "simple"}


def test_exact_router_preset_name_honors_enabled():
    config = {"presets": {"off": {"enabled": False}, "on": {}}}
    assert exact_router_preset_name(config, "off") is None
    assert exact_router_preset_name(config, "on") == "on"
    assert exact_router_preset_name(config, "missing") is None
    assert exact_router_preset_name(config, "") is None


def test_resolve_router_preset_by_name_and_default():
    config = {
        "default_preset": "a",
        "presets": {"a": {}, "b": {"default_route": "complex"}},
    }
    assert resolve_router_preset(config)["default_route"] == "simple"
    assert resolve_router_preset(config, "b")["default_route"] == "complex"
    try:
        resolve_router_preset(config, "missing")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_list_and_set_active_preset():
    config = {"presets": {"a": {}, "b": {}}}
    assert list_router_presets(config) == ["a", "b"]
    cfg = set_active_router_preset(config, "b")
    assert cfg["active_preset"] == "b"
    try:
        set_active_router_preset(config, "missing")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_flat_legacy_shape_becomes_default_preset():
    cfg = normalize_router_config(
        {"classifier": {"provider": "lmstudio", "model": "qwen/qwen3-4b-thinking-2507"}}
    )
    assert cfg["presets"][DEFAULT_ROUTER_PRESET_NAME]["classifier"]["provider"] == "lmstudio"


def test_flattened_view_carries_all_preset_fields():
    """The flattened compatibility view (dashboard/desktop callers) must
    round-trip every normalized preset field — omitting one silently resets
    it when a client reads-then-saves the flattened shape."""
    cfg = normalize_router_config(
        {
            "presets": {
                "default": {
                    "classifier_context_messages": 7,
                    "short_circuit_chars": 40,
                    "classifier_max_tokens": 24,
                }
            }
        }
    )
    preset = cfg["presets"][DEFAULT_ROUTER_PRESET_NAME]
    # Every scalar/collection key of the normalized preset appears flattened.
    for key in preset:
        assert key in cfg, f"flattened view missing preset field {key!r}"
    assert cfg["classifier_context_messages"] == 7
    assert cfg["short_circuit_chars"] == 40
    assert cfg["classifier_max_tokens"] == 24
