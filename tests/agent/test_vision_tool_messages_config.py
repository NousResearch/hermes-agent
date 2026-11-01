"""Config-declared ``supports_vision_tool_messages: false`` (per-provider capability).

Some OpenAI-compatible servers ACCEPT a tool-role image part and answer 200, but
silently drop the pixels — the model then sees only the envelope's text placeholder.
Nothing 400s, so no reactive recovery can catch it: the endpoint's owner has to say
so. This config key lets them, per provider / per model:

    model:
      provider: custom:my-endpoint
    providers:
      my-endpoint:
        api: http://localhost:8085/v1
        models:
          my-vision-model:
            supports_vision: true
            supports_vision_tool_messages: false

The native tool-result lane is then skipped and the image is described by the aux
vision LLM instead.
"""

from __future__ import annotations

import pytest

from agent.image_routing import lookup_supports_vision_tool_messages
from tools.vision_tools import _accepts_tool_result_images, _config_rejects_tool_media

MODEL = "my-vision-model"


def _cfg(*, models=None, provider="custom:my-endpoint"):
    """A config declaring the capability on a named custom provider.

    ``model.provider`` carries the ``custom:<name>`` identity, which is how a named
    custom provider is actually recorded in config.yaml (the runtime rewrites it to
    ``provider="custom"`` but keeps the name here).
    """
    return {
        "model": {"provider": provider},
        "providers": {
            "my-endpoint": {
                "api": "http://localhost:8085/v1",
                "models": models if models is not None else {
                    MODEL: {"supports_vision": True, "supports_vision_tool_messages": False},
                },
            },
        },
    }


class TestLookupSupportsVisionToolMessages:
    def test_unset_is_unknown(self):
        assert lookup_supports_vision_tool_messages("custom", MODEL, {}) is None

    def test_named_provider_model_declares_false(self):
        assert lookup_supports_vision_tool_messages("custom", MODEL, _cfg()) is False

    def test_requested_provider_name_also_matches(self):
        """The runtime provider is ``custom``; the bare config name must match too."""
        assert lookup_supports_vision_tool_messages(
            "custom", MODEL, _cfg(), requested_provider="my-endpoint") is False

    def test_top_level_model_key_declares_false(self):
        cfg = {"model": {"provider": "custom:my-endpoint", "supports_vision_tool_messages": False}}
        assert lookup_supports_vision_tool_messages("custom", MODEL, cfg) is False

    def test_true_does_not_veto(self):
        """``true`` means the endpoint DOES carry tool-result images — never a veto."""
        cfg = _cfg(models={MODEL: {"supports_vision_tool_messages": True}})
        assert lookup_supports_vision_tool_messages("custom", MODEL, cfg) is None

    def test_other_provider_unaffected(self):
        """A declaration on one endpoint must not veto a different route."""
        assert lookup_supports_vision_tool_messages(
            "custom", "some-other-model", _cfg()) is None

    def test_profile_veto_still_wins(self):
        """A profile that 400s on list-type tool content is vetoed regardless of config."""
        cfg = _cfg(models={MODEL: {"supports_vision_tool_messages": True}})
        assert lookup_supports_vision_tool_messages("xiaomi", "mimo-v2.5", cfg) is False


class TestConfigRejectsToolMedia:
    def test_declared_false_vetoes(self):
        assert _config_rejects_tool_media("custom", MODEL, _cfg()) is True

    def test_undeclared_is_not_a_veto(self):
        cfg = _cfg(models={MODEL: {"supports_vision": True}})
        assert _config_rejects_tool_media("custom", MODEL, cfg) is False

    def test_missing_config_is_not_a_veto(self):
        assert _config_rejects_tool_media("custom", MODEL, None) is False


class TestAcceptsToolResultImages:
    def test_config_false_vetoes_the_native_lane(self):
        """The whole point: the native lane must be off for such an endpoint."""
        assert _accepts_tool_result_images("custom", MODEL, _cfg()) is False

    def test_without_the_declaration_the_lane_stays_available(self):
        """No declaration → no behavior change for anyone else."""
        cfg = _cfg(models={MODEL: {"supports_vision": True}})
        assert _accepts_tool_result_images("custom", MODEL, cfg) is True


@pytest.mark.parametrize("raw,expected", [
    (True, True),
    (False, False),
    ("true", True),
    ("false", False),
    ("no", False),
    ("off", False),
    (1, True),
    (0, False),
])
def test_yaml_boolean_tokens(raw, expected):
    """Standard YAML booleans are accepted, like the sibling ``supports_vision`` key."""
    cfg = _cfg(models={MODEL: {"supports_vision_tool_messages": raw}})
    verdict = lookup_supports_vision_tool_messages("custom", MODEL, cfg)
    assert (verdict is False) is (expected is False)


def test_garbage_value_is_not_a_veto():
    """A non-boolean value must not silently veto (or silently enable) the lane."""
    cfg = _cfg(models={MODEL: {"supports_vision_tool_messages": "maybe"}})
    assert lookup_supports_vision_tool_messages("custom", MODEL, cfg) is None
