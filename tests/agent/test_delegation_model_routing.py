"""Contract tests for the delegation model-profile resolver (agent/delegation_model_routing.py).

Behavior contracts (not snapshots): parsing rejects malformed shapes loudly, selection follows the
documented precedence ladder, unknown profile names fail before child construction with an
actionable message, invalid reasoning_effort warns and is ignored (NS-696 clamp-at-transport
doctrine — never reject), and the resolved route is immutable.
"""

import dataclasses
import logging

import pytest

from agent.delegation_model_routing import (
    ProfileRoute,
    parse_profiles,
    profile_config_errors,
    resolve_profile_route,
    select_profile_name,
)


def _cfg(profiles=None, default_profile="", **extra):
    cfg = {"default_profile": default_profile, "profiles": profiles if profiles is not None else {}}
    cfg.update(extra)
    return cfg


SMALL = {"provider": "anthropic", "model": "claude-haiku-current"}
LARGE = {"provider": "openrouter", "model": "big/model"}


# ---------------------------------------------------------------------------
# parse_profiles
# ---------------------------------------------------------------------------

class TestParseProfiles:
    def test_empty_profiles_parse_to_empty_dict(self):
        assert parse_profiles(_cfg()) == {}

    @pytest.mark.parametrize("cfg", [None, {}, {"profiles": None}, {"profiles": {}}])
    def test_tolerates_missing_or_none_sections(self, cfg):
        assert parse_profiles(cfg) == {}

    def test_parses_minimal_profile(self):
        specs = parse_profiles(_cfg({"small": dict(SMALL)}))
        assert set(specs) == {"small"}
        spec = specs["small"]
        assert spec.provider == "anthropic"
        assert spec.model == "claude-haiku-current"
        assert spec.fallback == ()
        assert spec.max_iterations is None
        assert spec.reasoning_config is None

    def test_unknown_keys_rejected(self):
        cfg = _cfg({"small": {**SMALL, "toolsets": ["web"]}})
        with pytest.raises(ValueError) as exc:
            parse_profiles(cfg)
        assert "toolsets" in str(exc.value)
        assert "small" in str(exc.value)

    def test_missing_model_rejected(self):
        with pytest.raises(ValueError) as exc:
            parse_profiles(_cfg({"small": {"provider": "anthropic"}}))
        assert "model" in str(exc.value)

    def test_non_dict_profile_rejected(self):
        with pytest.raises(ValueError):
            parse_profiles(_cfg({"small": "claude-haiku-current"}))

    @pytest.mark.parametrize("bad_fallback", [
        "openrouter/big",                       # not a list
        [{"provider": "openrouter"}],            # entry missing model
        [{"model": "x"}],                        # entry missing provider
        ["openrouter/big"],                      # entry not a dict
        [{"provider": "p", "model": "m", "extra": 1}],  # unknown entry key
    ])
    def test_malformed_fallback_rejected(self, bad_fallback):
        with pytest.raises(ValueError) as exc:
            parse_profiles(_cfg({"small": {**SMALL, "fallback": bad_fallback}}))
        assert "fallback" in str(exc.value)

    def test_valid_fallback_parsed_as_tuple(self):
        specs = parse_profiles(_cfg({
            "small": {**SMALL, "fallback": [{"provider": "openrouter", "model": "b/c"}]},
        }))
        fb = specs["small"].fallback
        assert isinstance(fb, tuple) and len(fb) == 1
        assert fb[0].provider == "openrouter"
        assert fb[0].model == "b/c"

    def test_valid_reasoning_effort_parsed(self):
        specs = parse_profiles(_cfg({"small": {**SMALL, "reasoning_effort": "high"}}))
        assert specs["small"].reasoning_config == {"enabled": True, "effort": "high"}

    def test_reasoning_effort_none_means_disabled(self):
        specs = parse_profiles(_cfg({"small": {**SMALL, "reasoning_effort": "none"}}))
        assert specs["small"].reasoning_config == {"enabled": False}

    def test_invalid_reasoning_effort_warns_and_is_ignored(self, caplog):
        """NS-696 clamp doctrine: an invalid effort must never reject the profile."""
        with caplog.at_level(logging.WARNING):
            specs = parse_profiles(_cfg({"small": {**SMALL, "reasoning_effort": "turbo"}}))
        assert specs["small"].reasoning_config is None
        assert any("turbo" in rec.getMessage() for rec in caplog.records)

    def test_max_iterations_parsed(self):
        specs = parse_profiles(_cfg({"small": {**SMALL, "max_iterations": 20}}))
        assert specs["small"].max_iterations == 20

    def test_non_int_max_iterations_rejected(self):
        with pytest.raises(ValueError) as exc:
            parse_profiles(_cfg({"small": {**SMALL, "max_iterations": "lots"}}))
        assert "max_iterations" in str(exc.value)


class TestProfileConfigErrors:
    """Multi-issue collector used by `hermes config check` — never raises."""

    def test_clean_config_has_no_errors(self):
        assert profile_config_errors(_cfg({"small": dict(SMALL)})) == []

    def test_collects_multiple_errors(self):
        errors = profile_config_errors(_cfg({
            "a": {"provider": "p"},                 # missing model
            "b": {**SMALL, "bogus": 1},             # unknown key
        }))
        joined = "\n".join(errors)
        assert "a" in joined and "model" in joined
        assert "b" in joined and "bogus" in joined
        assert len(errors) >= 2

    def test_default_profile_must_exist(self):
        errors = profile_config_errors(_cfg({"small": dict(SMALL)}, default_profile="huge"))
        assert any("huge" in e and "small" in e for e in errors)

    def test_empty_default_profile_is_fine(self):
        assert profile_config_errors(_cfg({"small": dict(SMALL)}, default_profile="")) == []

    def test_profiles_must_be_a_mapping(self):
        errors = profile_config_errors({"profiles": ["small"]})
        assert errors and any("profiles" in e for e in errors)


# ---------------------------------------------------------------------------
# select_profile_name — precedence ladder
# ---------------------------------------------------------------------------

FALSY = [False, None, "", 0]


class TestSelectProfileName:
    def test_per_task_wins_over_everything(self):
        cfg = _cfg({"small": dict(SMALL)}, default_profile="large")
        assert select_profile_name("small", "mid", cfg) == "small"

    def test_top_level_wins_over_default(self):
        cfg = _cfg({"small": dict(SMALL)}, default_profile="large")
        assert select_profile_name(None, "small", cfg) == "small"

    def test_default_profile_applies_when_others_silent(self):
        cfg = _cfg({"small": dict(SMALL)}, default_profile="small")
        assert select_profile_name(None, None, cfg) == "small"

    def test_legacy_config_selects_no_profile(self):
        """Level 4: legacy delegation.provider/model config — resolver stays out of the way."""
        cfg = {"provider": "openrouter", "model": "x/y"}
        assert select_profile_name(None, None, cfg) is None

    def test_pure_inherit_selects_no_profile(self):
        """Level 5: nothing configured at all — parent inherit."""
        assert select_profile_name(None, None, {}) is None
        assert select_profile_name(None, None, None) is None

    @pytest.mark.parametrize("falsy", FALSY)
    def test_falsy_task_profile_falls_through(self, falsy):
        cfg = _cfg({"small": dict(SMALL)}, default_profile="small")
        assert select_profile_name(falsy, None, cfg) == "small"

    @pytest.mark.parametrize("falsy", FALSY)
    def test_falsy_top_profile_falls_through(self, falsy):
        cfg = _cfg({"small": dict(SMALL)}, default_profile="small")
        assert select_profile_name(None, falsy, cfg) == "small"

    @pytest.mark.parametrize("falsy", FALSY)
    def test_falsy_default_profile_means_legacy(self, falsy):
        cfg = _cfg({"small": dict(SMALL)}, default_profile=falsy)
        assert select_profile_name(None, None, cfg) is None

    def test_whitespace_names_are_stripped(self):
        cfg = _cfg({"small": dict(SMALL)})
        assert select_profile_name("  small  ", None, cfg) == "small"


# ---------------------------------------------------------------------------
# resolve_profile_route
# ---------------------------------------------------------------------------

class _FakeCaps:
    def __init__(self, supports_tools=True):
        self.supports_tools = supports_tools


@pytest.fixture
def fake_runtime(monkeypatch):
    """Stub the two lazy-imported collaborators at their source modules."""
    calls = {}

    def _resolve(*, requested=None, explicit_api_key=None, explicit_base_url=None, target_model=None):
        calls["requested"] = requested
        calls["target_model"] = target_model
        return {
            "provider": requested or "resolved-default",
            "model": target_model,
            "base_url": "https://api.example.test/v1",
            "api_key": "sk-test",
            "api_mode": "chat_completions",
        }

    import hermes_cli.runtime_provider as rp
    import agent.models_dev as md
    monkeypatch.setattr(rp, "resolve_runtime_provider", _resolve)
    monkeypatch.setattr(md, "get_model_capabilities", lambda *a, **k: _FakeCaps(True))
    return calls


class TestResolveProfileRoute:
    def test_unknown_name_lists_configured_profiles(self):
        cfg = _cfg({"small": dict(SMALL), "large": dict(LARGE)})
        with pytest.raises(ValueError) as exc:
            resolve_profile_route("huge", cfg, parent_agent=None)
        msg = str(exc.value)
        assert "huge" in msg
        assert "small" in msg and "large" in msg

    def test_unknown_name_with_no_profiles_configured(self):
        with pytest.raises(ValueError) as exc:
            resolve_profile_route("small", {}, parent_agent=None)
        assert "small" in str(exc.value)

    def test_route_resolves_through_runtime_provider(self, fake_runtime):
        cfg = _cfg({"small": dict(SMALL)})
        route = resolve_profile_route("small", cfg, parent_agent=None)
        # credential resolution went through resolve_runtime_provider — same path as legacy
        assert fake_runtime["requested"] == "anthropic"
        assert fake_runtime["target_model"] == "claude-haiku-current"
        assert route.requested_profile == "small"
        assert route.provider == "anthropic"
        assert route.model == "claude-haiku-current"
        assert route.base_url == "https://api.example.test/v1"
        assert route.api_key == "sk-test"
        assert route.api_mode == "chat_completions"

    def test_route_carries_profile_extras(self, fake_runtime):
        cfg = _cfg({"small": {**SMALL, "reasoning_effort": "low", "max_iterations": 20,
                              "fallback": [{"provider": "openrouter", "model": "b/c"}]}})
        route = resolve_profile_route("small", cfg, parent_agent=None)
        assert route.reasoning_config == {"enabled": True, "effort": "low"}
        assert route.max_iterations == 20
        assert len(route.fallback) == 1
        assert route.fallback[0].provider == "openrouter"

    def test_route_exposes_supports_tools(self, monkeypatch, fake_runtime):
        import agent.models_dev as md
        monkeypatch.setattr(md, "get_model_capabilities", lambda *a, **k: _FakeCaps(False))
        route = resolve_profile_route("small", _cfg({"small": dict(SMALL)}), parent_agent=None)
        assert route.supports_tools is False

    def test_route_is_frozen(self, fake_runtime):
        route = resolve_profile_route("small", _cfg({"small": dict(SMALL)}), parent_agent=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            route.model = "other"

    def test_route_fallback_is_immutable_sequence(self, fake_runtime):
        route = resolve_profile_route("small", _cfg({"small": dict(SMALL)}), parent_agent=None)
        assert isinstance(route.fallback, tuple)


# ---------------------------------------------------------------------------
# credential-resolution errors surface as ValueError (clean tool_error seam)
# ---------------------------------------------------------------------------

class TestCredentialErrorSurfacing:
    def test_runtime_provider_auth_error_surfaces_as_value_error(self, monkeypatch):
        """AuthError subclasses RuntimeError, which the delegate_task/lifecycle seams do NOT
        catch — the resolver must re-raise it as ValueError naming the profile, mirroring the
        legacy provider branch (tools/delegate_tool_config._runtime_provider_credentials)."""
        import hermes_cli.runtime_provider as rp
        from hermes_cli.auth_constants import AuthError

        def _boom(**_kw):
            raise AuthError("ANTHROPIC_API_KEY is not set", provider="anthropic")

        monkeypatch.setattr(rp, "resolve_runtime_provider", _boom)
        with pytest.raises(ValueError) as exc:
            resolve_profile_route("small", _cfg({"small": dict(SMALL)}), parent_agent=None)
        assert not isinstance(exc.value, RuntimeError)
        msg = str(exc.value)
        assert "small" in msg
        assert "ANTHROPIC_API_KEY is not set" in msg


class TestEmptyProfileNames:
    @pytest.mark.parametrize("bad_name", ["", "   ", "\t"])
    def test_empty_or_whitespace_profile_name_rejected(self, bad_name):
        with pytest.raises(ValueError, match="empty or whitespace"):
            parse_profiles(_cfg({bad_name: dict(SMALL)}))
