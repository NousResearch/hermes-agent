"""Gateway application precedence for model candidates."""

from gateway.model_resolution import effective_model_candidate


class _ModelValue:
    def __init__(self, model):
        self.model = model


def test_effective_model_candidate_uses_first_non_empty_tier():
    assert effective_model_candidate("", {"model": "session-model"}, "global-model") == "session-model"


def test_effective_model_candidate_supports_gateway_config_shapes():
    assert effective_model_candidate({"model": "dict-model"}) == "dict-model"
    assert effective_model_candidate(_ModelValue("attr-model")) == "attr-model"


def test_effective_model_candidate_falls_through_empty_values():
    assert effective_model_candidate(None, {"model": ""}, _ModelValue(""), "global-model") == "global-model"
    assert effective_model_candidate(None, "", {}) == ""
