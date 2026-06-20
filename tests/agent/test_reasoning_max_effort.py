"""Tests for the ``max`` reasoning-effort level and the nearest-down clamp.

``max`` is the deepest reasoning-effort word a user can request. It is added to
the universal vocabulary (`VALID_REASONING_EFFORTS`), and the request-path clamp
in ``AIAgent._github_models_reasoning_extra_body`` maps it down to the strongest
level a given model actually supports — e.g. ``max`` → ``xhigh`` for GPT-5.x
(whose catalog ceiling is ``xhigh``) rather than collapsing to ``medium``.
"""

import hermes_constants


class TestMaxInVocabulary:
    def test_max_is_valid_effort(self):
        assert "max" in hermes_constants.VALID_REASONING_EFFORTS

    def test_parse_reasoning_effort_accepts_max(self):
        assert hermes_constants.parse_reasoning_effort("max") == {
            "enabled": True,
            "effort": "max",
        }

    def test_parse_reasoning_effort_still_rejects_garbage(self):
        assert hermes_constants.parse_reasoning_effort("ultra") is None

    def test_none_still_disables(self):
        assert hermes_constants.parse_reasoning_effort("none") == {"enabled": False}


class TestNearestDownClamp:
    """Exercises the REAL clamp in
    AIAgent._github_models_reasoning_extra_body by binding the unbound method to
    a lightweight stub (the method only reads self.model + self.reasoning_config
    and calls github_model_reasoning_efforts), so we test shipped code, not a
    re-implementation.
    """

    @staticmethod
    def _clamp(requested_effort, supported_efforts, monkeypatch):
        import types
        import run_agent

        monkeypatch.setattr(
            "hermes_cli.models.github_model_reasoning_efforts",
            lambda model, **kw: list(supported_efforts),
        )
        stub = types.SimpleNamespace(
            model="some-model",
            reasoning_config={"enabled": True, "effort": requested_effort},
        )
        result = run_agent.AIAgent._github_models_reasoning_extra_body(stub)
        assert result is not None
        return result["effort"]

    def test_max_maps_to_xhigh_for_gpt5(self, monkeypatch):
        # GPT-5.x catalog tops out at xhigh; max must land on xhigh, not medium.
        supported = ["minimal", "low", "medium", "high", "xhigh"]
        assert self._clamp("max", supported, monkeypatch) == "xhigh"

    def test_max_honored_when_supported(self, monkeypatch):
        supported = ["low", "medium", "high", "xhigh", "max"]
        assert self._clamp("max", supported, monkeypatch) == "max"

    def test_xhigh_no_longer_force_downgraded_to_high(self, monkeypatch):
        # The old guard collapsed xhigh→high unconditionally; now xhigh is kept
        # when the catalog supports it.
        supported = ["minimal", "low", "medium", "high", "xhigh"]
        assert self._clamp("xhigh", supported, monkeypatch) == "xhigh"

    def test_unsupported_high_picks_strongest_at_or_below(self, monkeypatch):
        # Request high, catalog only has low/medium → medium (nearest down).
        assert self._clamp("high", ["low", "medium"], monkeypatch) == "medium"

    def test_falls_back_to_weakest_when_nothing_at_or_below(self, monkeypatch):
        # Request minimal, catalog only has high → high (weakest available).
        assert self._clamp("minimal", ["high", "xhigh"], monkeypatch) == "high"

    def test_single_supported_level(self, monkeypatch):
        assert self._clamp("max", ["medium"], monkeypatch) == "medium"
