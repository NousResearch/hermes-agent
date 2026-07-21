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

    def test_default_key_wins_over_legacy_model_key(self):
        """`default` is the model the CLI/gateway actually resolve; the override
        describes it. cli.py promotes legacy `model.model` only when `default`
        is absent, and gateway._resolve_gateway_model reads `default` first —
        a config containing both must not lose the override for its default."""
        cfg = {"model": "a", "default": "b", "context_length": 1000}
        assert scoped_config_context_length(cfg, "b")[0] == 1000
        assert scoped_config_context_length(cfg, "a")[0] is None

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


class TestNormalizedComparison:
    """`agent.model` is normalized before this runs; the config value is not.

    Without normalizing both sides, a valid config that names its model with a
    provider prefix looks like a mismatch and silently loses its override.
    """

    @staticmethod
    def _strip_prefix(name: str) -> str:
        return name.split("/", 1)[1] if "/" in name else name

    def test_prefixed_config_model_matches_normalized_active_model(self):
        cfg = {"model": "zai/glm-4.6", "context_length": 40960}
        value, mismatch = scoped_config_context_length(
            cfg, "glm-4.6", normalize=self._strip_prefix,
        )
        assert value == 40960, "a prefixed config naming the running model must apply"
        assert mismatch is None

    def test_genuine_mismatch_still_detected_after_normalizing(self):
        cfg = {"model": "zai/glm-4.6", "context_length": 40960}
        value, mismatch = scoped_config_context_length(
            cfg, "glm-5.2", normalize=self._strip_prefix,
        )
        assert value is None
        assert mismatch == ("zai/glm-4.6", "glm-5.2", 40960)

    def test_mismatch_reports_the_original_config_string(self):
        """The warning should echo what the user wrote, not the normalized form."""
        cfg = {"model": "anthropic/claude-sonnet-5", "context_length": 1000}
        _, mismatch = scoped_config_context_length(
            cfg, "other", normalize=self._strip_prefix,
        )
        assert mismatch[0] == "anthropic/claude-sonnet-5"

    def test_no_normalizer_compares_raw(self):
        cfg = {"model": "zai/glm-4.6", "context_length": 40960}
        assert scoped_config_context_length(cfg, "glm-4.6")[0] is None

    def test_normalizer_failure_falls_back_to_raw_comparison(self):
        def boom(_name):
            raise RuntimeError("normalizer unavailable")

        cfg = {"model": "m", "context_length": 7}
        assert scoped_config_context_length(cfg, "m", normalize=boom)[0] == 7
