"""Declared per-route reasoning vocabularies (``declared_route_efforts``).

The picker offers what the route declares; undeclared routes stay fail-open on the shared
ladder. See the ``reasoning_efforts`` config key and ``ProviderProfile.supported_reasoning_efforts``.
"""

from agent.reasoning_effort import declared_route_efforts

CONFIG = {
    "providers": {
        "relay": {
            "models": {
                "qwen-next": {"reasoning_efforts": ["none", "low", "high", "max"]},
                "no-thinker": {"reasoning_efforts": []},
                "noisy": {"reasoning_efforts": ["LOW ", "high", "not-a-level", "low"]},
                "plain": {"context_length": 128000},
            }
        }
    }
}


def test_config_declaration_is_returned_verbatim_as_a_ladder_subset():
    assert declared_route_efforts("relay", "qwen-next", CONFIG) == ("none", "low", "high", "max")


def test_unknown_levels_are_dropped_and_duplicates_collapse():
    # clamp_effort ignores ladder-unknown levels, so offering them would promise nothing.
    assert declared_route_efforts("relay", "noisy", CONFIG) == ("low", "high")


def test_empty_declaration_is_a_declaration_not_a_gap():
    assert declared_route_efforts("relay", "no-thinker", CONFIG) == ()


def test_model_without_the_key_is_undeclared():
    assert declared_route_efforts("relay", "plain", CONFIG) is None


def test_unknown_provider_or_model_is_undeclared():
    assert declared_route_efforts("relay", "absent", CONFIG) is None
    assert declared_route_efforts("absent", "qwen-next", CONFIG) is None
    assert declared_route_efforts("relay", None, CONFIG) is None


def test_without_a_provider_name_any_entry_declaring_the_model_answers():
    # Transports know the model and the base_url, not the config key that named the route.
    # A declaration is written per model, so the id alone is enough to find it.
    assert declared_route_efforts(None, "qwen-next", CONFIG) == ("none", "low", "high", "max")
    assert declared_route_efforts(None, "absent", CONFIG) is None


def test_malformed_config_never_raises():
    assert declared_route_efforts("relay", "qwen-next", {"providers": "nope"}) is None
    assert declared_route_efforts("relay", "qwen-next", {"providers": {"relay": []}}) is None
    assert declared_route_efforts("relay", "qwen-next", {"providers": {"relay": {"models": {"qwen-next": {"reasoning_efforts": "high"}}}}}) is None


def test_profile_hook_answers_when_config_is_silent(monkeypatch):
    import providers as providers_mod

    class _Profile:
        def supported_reasoning_efforts(self, model):
            return ("none", "low", "high") if model == "declared" else None

    monkeypatch.setattr(providers_mod, "get_provider_profile", lambda name: _Profile())
    assert declared_route_efforts("some-provider", "declared", {}) == ("none", "low", "high")
    assert declared_route_efforts("some-provider", "other", {}) is None


def test_config_wins_over_the_profile_hook(monkeypatch):
    import providers as providers_mod

    class _Profile:
        def supported_reasoning_efforts(self, model):
            return ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    monkeypatch.setattr(providers_mod, "get_provider_profile", lambda name: _Profile())
    assert declared_route_efforts("relay", "qwen-next", CONFIG) == ("none", "low", "high", "max")


def test_a_raising_profile_leaves_the_route_undeclared(monkeypatch):
    import providers as providers_mod

    class _Profile:
        def supported_reasoning_efforts(self, model):
            raise RuntimeError("plugin blew up")

    monkeypatch.setattr(providers_mod, "get_provider_profile", lambda name: _Profile())
    assert declared_route_efforts("some-provider", "any", {}) is None
