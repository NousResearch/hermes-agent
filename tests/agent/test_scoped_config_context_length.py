"""``model.context_length`` must not leak onto a model it does not describe.

Regression: `model.context_length` written for the configured default (e.g. a
40,960-token local model) was applied verbatim when the active model came from a
CLI `--model` flag (e.g. a 1M-window hosted model). That drove `threshold_tokens`
below the incompressible floor — system prompt + tool schemas — so `should_compress()`
returned True on every turn and the agent compacted forever without progress.
"""
from agent.agent_init import scoped_config_context_length


class TestScopedConfigContextLength:
    def test_applies_when_active_model_matches_config_model(self):
        cfg = {"model": "glm-coding", "context_length": 40960}
        value, mismatch = scoped_config_context_length(cfg, "glm-coding")
        assert value == 40960
        assert mismatch is None

    def test_applies_when_config_only_names_default(self):
        cfg = {"default": "glm-coding", "context_length": 40960}
        value, mismatch = scoped_config_context_length(cfg, "glm-coding")
        assert value == 40960
        assert mismatch is None

    def test_ignored_when_active_model_differs(self):
        """The bug: a CLI --model override inherits the wrong window."""
        cfg = {"model": "glm-coding", "default": "glm-coding", "context_length": 40960}
        value, mismatch = scoped_config_context_length(cfg, "glm-5.2")
        assert value is None, "override for glm-coding must not size glm-5.2"
        assert mismatch == ("glm-coding", "glm-5.2", 40960)

    def test_model_key_wins_over_default_key(self):
        cfg = {"model": "a", "default": "b", "context_length": 1000}
        assert scoped_config_context_length(cfg, "a")[0] == 1000
        assert scoped_config_context_length(cfg, "b")[0] is None

    def test_applies_when_config_names_no_model(self):
        """No model named -> the override is global by intent; keep prior behavior."""
        cfg = {"context_length": 40960}
        value, mismatch = scoped_config_context_length(cfg, "glm-5.2")
        assert value == 40960
        assert mismatch is None

    def test_applies_when_active_model_unknown(self):
        cfg = {"model": "glm-coding", "context_length": 40960}
        value, mismatch = scoped_config_context_length(cfg, "")
        assert value == 40960
        assert mismatch is None

    def test_no_context_length_configured(self):
        assert scoped_config_context_length({"model": "x"}, "y") == (None, None)

    def test_non_dict_config(self):
        assert scoped_config_context_length(None, "y") == (None, None)
        assert scoped_config_context_length("nope", "y") == (None, None)

    def test_invalid_value_is_passed_through_for_caller_validation(self):
        """Type validation stays in the caller; scoping only decides applicability."""
        cfg = {"model": "m", "context_length": "256K"}
        value, mismatch = scoped_config_context_length(cfg, "m")
        assert value == "256K"
        assert mismatch is None
