from agent.smart_model_routing import apply_source_override, choose_cheap_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


# ----- apply_source_override (model.by_source.<kind>) -----

def test_source_override_returns_unchanged_when_kind_missing():
    m, r = apply_source_override("base-model", {"api_key": "k1"}, {"default": "base-model"}, None)
    assert m == "base-model" and r == {"api_key": "k1"}


def test_source_override_returns_unchanged_when_no_by_source():
    m, r = apply_source_override("base-model", {"api_key": "k1"}, {"default": "base-model"}, "owner")
    assert m == "base-model" and r == {"api_key": "k1"}


def test_source_override_applies_full_bundle():
    cfg = {
        "default": "base-model",
        "by_source": {
            "owner": {
                "model": "strong-model",
                "api_key": "sk-owner",
                "base_url": "https://owner.example/v1",
                "provider": "custom",
            },
        },
    }
    m, r = apply_source_override("base-model", {"api_key": "k1"}, cfg, "owner")
    assert m == "strong-model"
    assert r["api_key"] == "sk-owner"
    assert r["base_url"] == "https://owner.example/v1"
    assert r["provider"] == "custom"


def test_source_override_partial_preserves_base_fields():
    cfg = {"by_source": {"cron": {"model": "cheap-model"}}}
    m, r = apply_source_override(
        "base-model",
        {"api_key": "k1", "base_url": "https://base.example/v1", "provider": "custom"},
        cfg,
        "cron",
    )
    assert m == "cheap-model"
    # Unset fields in entry → inherit base runtime values
    assert r == {"api_key": "k1", "base_url": "https://base.example/v1", "provider": "custom"}


def test_source_override_empty_entry_is_no_op():
    cfg = {"by_source": {"stranger": {}}}
    m, r = apply_source_override("base-model", {"api_key": "k1"}, cfg, "stranger")
    assert m == "base-model" and r == {"api_key": "k1"}


def test_source_override_unknown_kind_is_no_op():
    cfg = {"by_source": {"owner": {"model": "strong-model"}}}
    m, r = apply_source_override("base-model", {"api_key": "k1"}, cfg, "some_unrecognised_kind")
    assert m == "base-model" and r == {"api_key": "k1"}


def test_source_override_handles_null_model_config():
    m, r = apply_source_override("base-model", {"api_key": "k1"}, None, "owner")
    assert m == "base-model" and r == {"api_key": "k1"}


def test_source_override_accepts_args_list():
    cfg = {"by_source": {"cron": {"args": ["--flag", "value"]}}}
    _, r = apply_source_override("base-model", {"args": ["old"]}, cfg, "cron")
    assert r["args"] == ["--flag", "value"]


def test_source_override_ignores_empty_field_values():
    # Empty strings / None / empty list in an entry should NOT overwrite base
    cfg = {"by_source": {"owner": {"api_key": "", "base_url": None, "args": []}}}
    m, r = apply_source_override(
        "base-model",
        {"api_key": "k1", "base_url": "https://base.example/v1"},
        cfg,
        "owner",
    )
    assert m == "base-model"
    assert r["api_key"] == "k1"
    assert r["base_url"] == "https://base.example/v1"
