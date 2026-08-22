"""Per-model override for list-type ``role: "tool"`` content.

``ProviderProfile.supports_vision_tool_messages`` is keyed on the provider,
which cannot express a relay that fronts several vendors.  The OpenCode Go
catalog serves MiMo alongside Kimi, GLM, MiniMax, DeepSeek and Qwen
(``hermes_cli/models.py``), and only the MiMo route rejects multimodal
tool-result content — so setting the flag provider-wide fixes MiMo by
downgrading every other family on the same relay.  That is the objection
raised on #47026, and #89057 is the same failure reported again (HTTP 500 from
the OpenCode Go proxy, with the rejected image persisting in history so every
later turn in the thread fails too).

These tests pin the scoped alternative: an explicit per-model
``supports_vision_tool_messages`` in config.yaml, resolved before the provider
profile.  Each positive case is paired with a control on a sibling model of the
*same* provider, so a green run cannot mean "everything is downgraded".
"""

from __future__ import annotations

import base64

from agent.image_routing import supports_vision_tool_messages_override


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG).decode()


def _cfg(mimo_value):
    """OpenCode Go-shaped config: MiMo carries the override, siblings do not."""
    mimo = {"supports_vision": True}
    if mimo_value is not None:
        mimo["supports_vision_tool_messages"] = mimo_value
    return {
        "model": {"provider": "opencode-go"},
        "providers": {
            "opencode-go": {
                "models": {
                    "mimo-v2.5": mimo,
                    "kimi-k2": {"supports_vision": True},
                    "glm-4.6": {"supports_vision": True},
                }
            }
        },
    }


class TestScopedOverrideResolution:
    def test_mimo_route_is_downgraded(self):
        assert supports_vision_tool_messages_override(
            "opencode-go", "mimo-v2.5", _cfg(False)
        ) is False

    def test_non_mimo_siblings_are_untouched(self):
        """The control case requested on #47026.

        A provider-wide flag would return False here too, silently costing
        Kimi and GLM their native tool-result images on this relay.
        """
        cfg = _cfg(False)
        assert supports_vision_tool_messages_override(
            "opencode-go", "kimi-k2", cfg
        ) is None
        assert supports_vision_tool_messages_override(
            "opencode-go", "glm-4.6", cfg
        ) is None

    def test_absent_key_returns_none_not_false(self):
        """None means "defer to the provider profile", not "reject"."""
        assert supports_vision_tool_messages_override(
            "opencode-go", "mimo-v2.5", _cfg(None)
        ) is None

    def test_quoted_false_still_means_false(self):
        """``supports_vision`` already guards this common YAML mistake."""
        assert supports_vision_tool_messages_override(
            "opencode-go", "mimo-v2.5", _cfg("false")
        ) is False

    def test_unparseable_value_defers_instead_of_guessing(self):
        assert supports_vision_tool_messages_override(
            "opencode-go", "mimo-v2.5", _cfg("maybe")
        ) is None

    def test_resolves_via_runtime_provider_when_config_provider_differs(self):
        """The runtime provider argument must be honored on its own.

        Regression guard: dropping it from the candidate list still passed the
        config-provider test below, because that one only exercises the
        fallback half of the lookup.
        """
        cfg = _cfg(False)
        cfg["model"]["provider"] = "some-other-provider"
        assert supports_vision_tool_messages_override(
            "opencode-go", "mimo-v2.5", cfg
        ) is False

    def test_resolves_via_config_provider_when_runtime_is_custom(self):
        """Named custom providers are rewritten to ``provider="custom"`` at
        runtime, so the user-declared name survives only under
        ``model.provider``.  Ignoring it would make the override unreachable on
        exactly the setups that need it (same order as ``supports_vision``).
        """
        assert supports_vision_tool_messages_override(
            "custom", "mimo-v2.5", _cfg(False)
        ) is False

    def test_resolves_via_requested_provider_keyword(self):
        cfg = _cfg(False)
        cfg["model"]["provider"] = "some-other-provider"
        assert supports_vision_tool_messages_override(
            "custom", "mimo-v2.5", cfg, requested_provider="opencode-go"
        ) is False

    def test_resolves_through_legacy_custom_providers_list(self):
        """Parity with ``_supports_vision_override``, which reads this shape."""
        cfg = {
            "model": {"provider": "custom"},
            "custom_providers": [
                {
                    "name": "my-relay",
                    "models": {
                        "mimo-v2.5": {"supports_vision_tool_messages": False},
                        "kimi-k2": {"supports_vision": True},
                    },
                }
            ],
        }
        assert supports_vision_tool_messages_override(
            "my-relay", "mimo-v2.5", cfg
        ) is False
        assert supports_vision_tool_messages_override(
            "my-relay", "kimi-k2", cfg
        ) is None

    def test_no_matching_provider_anywhere_returns_none(self):
        cfg = _cfg(False)
        cfg["model"]["provider"] = "unrelated-provider"
        assert supports_vision_tool_messages_override(
            "another-unrelated", "mimo-v2.5", cfg
        ) is None


def _agent(model, monkeypatch, overrides):
    from run_agent import AIAgent
    import agent.image_routing as ir

    agent = object.__new__(AIAgent)
    agent.provider = "opencode-go"
    agent.model = model
    monkeypatch.setattr(
        ir,
        "supports_vision_tool_messages_override",
        lambda provider, model, cfg=None: overrides.get(model),
    )
    return agent


def _multimodal_result():
    return {
        "_multimodal": True,
        "text_summary": "Screenshot (image)",
        "content": [
            {"type": "text", "text": "Screenshot"},
            {"type": "image_url", "image_url": {"url": _DATA_URL}},
        ],
    }


def _has_image(sent):
    return isinstance(sent, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in sent
    )


class TestMethodIntegration:
    """``_provider_supports_vision_tool_messages`` without stubbing the lookup.

    The end-to-end class below monkeypatches the override helper to isolate the
    branching in ``_tool_result_content_for_active_model``.  These tests patch
    only the *config source*, so they also pin that the method passes the real
    provider/model through and that the provider-profile fallback still decides
    when no override exists.
    """

    def _agent_for(self, provider, model):
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        return agent

    def test_per_model_override_beats_provider_profile(self, monkeypatch):
        import agent.image_routing as ir
        monkeypatch.setattr(
            ir, "load_config_readonly", lambda: _cfg(False), raising=False
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: _cfg(False)
        )
        agent = self._agent_for("opencode-go", "mimo-v2.5")
        assert agent._provider_supports_vision_tool_messages() is False

    def test_sibling_model_is_not_downgraded_by_the_override(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: _cfg(False)
        )
        agent = self._agent_for("opencode-go", "kimi-k2")
        assert agent._provider_supports_vision_tool_messages() is True

    def test_provider_profile_still_decides_without_an_override(self, monkeypatch):
        """Negative control for the fallback path.

        With no per-model key, the answer must come from the provider profile,
        not from a hardcoded constant.  ``xiaomi`` ships
        ``supports_vision_tool_messages=False``.
        """
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda: _cfg(None)
        )
        strict = self._agent_for("xiaomi", "mimo-v2.5")
        assert strict._provider_supports_vision_tool_messages() is False
        permissive = self._agent_for("openai", "gpt-5.4")
        assert permissive._provider_supports_vision_tool_messages() is True


class TestEndToEndToolResultShape:
    """What actually reaches the wire — the property #89057 is about."""

    def test_scoped_model_gets_text_summary(self, monkeypatch):
        agent = _agent("mimo-v2.5", monkeypatch, {"mimo-v2.5": False})
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        sent = agent._tool_result_content_for_active_model(
            "vision_analyze", _multimodal_result()
        )
        assert not _has_image(sent)
        # Degradation must land on the text summary, not an empty list or None:
        # a stripped-but-empty result would also satisfy "no image".
        assert isinstance(sent, str)
        assert "Screenshot" in sent

    def test_sibling_model_keeps_native_image(self, monkeypatch):
        """Control: without an override the image must survive untouched."""
        agent = _agent("kimi-k2", monkeypatch, {})
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        sent = agent._tool_result_content_for_active_model(
            "vision_analyze", _multimodal_result()
        )
        assert _has_image(sent)
        # And the parts arrive intact, not a re-wrapped or truncated payload.
        assert sent == _multimodal_result()["content"]
