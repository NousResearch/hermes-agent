from hermes_cli.model_routing import apply_route_to_turn, choose_tier, resolve_model_route


def _config(**overrides):
    cfg = {
        "model_routing": {
            "enabled": True,
            "default_tier": "balanced",
            "tiers": {
                "cheap": {"provider": "openrouter", "model": "cheap-model"},
                "balanced": {"provider": "openrouter", "model": "balanced-model"},
                "best": {"provider": "anthropic", "model": "best-model"},
            },
        }
    }
    cfg["model_routing"].update(overrides)
    return cfg


def _resolver(requested, target_model=None, explicit_base_url=None, **_kwargs):
    return {
        "provider": requested,
        "model": target_model,
        "api_key": f"key-for-{requested}",
        "base_url": explicit_base_url or f"https://{requested}.example/v1",
        "api_mode": "chat_completions",
    }


def _route():
    return {
        "model": "primary-model",
        "runtime": {
            "provider": "openrouter",
            "api_key": "primary-key",
            "base_url": "https://primary.example/v1",
            "api_mode": "chat_completions",
            "max_tokens": 1234,
        },
        "signature": (
            "primary-model",
            "openrouter",
            "https://primary.example/v1",
            "chat_completions",
            None,
            (),
        ),
    }


def test_routing_disabled_preserves_existing_route():
    route = _route()
    out = apply_route_to_turn(
        route=route.copy(),
        user_message="hi",
        config={"model_routing": {"enabled": False}},
        resolver=_resolver,
    )
    assert out["model"] == "primary-model"
    assert out["runtime"]["provider"] == "openrouter"
    assert "routing_decision" not in out


def test_explicit_call_override_wins_over_user_phrase():
    decision = resolve_model_route(
        user_message="use best model for this",
        current_model="primary-model",
        current_runtime=_route()["runtime"],
        config=_config(),
        explicit_override=True,
        resolver=_resolver,
    )
    assert decision is None


def test_use_best_model_routes_to_best_tier():
    out = apply_route_to_turn(
        route=_route(),
        user_message="use best model and think deeply about this design",
        config=_config(),
        resolver=_resolver,
    )
    assert out["model"] == "best-model"
    assert out["runtime"]["provider"] == "anthropic"
    assert out["routing_decision"]["tier"] == "best"
    assert out["routing_decision"]["source"] == "override"


def test_simple_prompt_routes_cheap():
    out = apply_route_to_turn(
        route=_route(),
        user_message="thanks",
        config=_config(),
        resolver=_resolver,
    )
    assert out["model"] == "cheap-model"
    assert out["runtime"]["provider"] == "openrouter"
    assert out["routing_decision"]["tier"] == "cheap"


def test_architecture_debugging_prompt_routes_best():
    tier, reason, confidence, source = choose_tier(
        "Investigate Hermes internals and implement model routing architecture",
        _config(),
    )
    assert tier == "best"
    assert source == "heuristic"
    assert confidence > 0.8
    assert "architecture" in reason


def test_bad_missing_tier_config_falls_back_safely():
    cfg = _config(tiers={"cheap": {"provider": "", "model": ""}})
    out = apply_route_to_turn(
        route=_route(),
        user_message="hi",
        config=cfg,
        resolver=_resolver,
    )
    assert out["model"] == "primary-model"
    assert "routing_decision" not in out


def test_bad_resolver_output_falls_back_safely():
    def boom(**_kwargs):
        raise RuntimeError("provider not configured")

    out = apply_route_to_turn(
        route=_route(),
        user_message="use best model",
        config=_config(),
        resolver=boom,
    )
    assert out["model"] == "primary-model"
    assert "routing_decision" not in out


def test_existing_max_tokens_preserved_for_routed_turn():
    out = apply_route_to_turn(
        route=_route(),
        user_message="hi",
        config=_config(),
        resolver=_resolver,
    )
    assert out["runtime"]["max_tokens"] == 1234


def test_tier_max_tokens_can_override_current_runtime():
    cfg = _config(
        tiers={
            "cheap": {"provider": "openrouter", "model": "cheap-model", "max_tokens": 99},
            "balanced": {"provider": "openrouter", "model": "balanced-model"},
            "best": {"provider": "anthropic", "model": "best-model"},
        }
    )
    out = apply_route_to_turn(
        route=_route(),
        user_message="hi",
        config=cfg,
        resolver=_resolver,
    )
    assert out["runtime"]["max_tokens"] == 99
