"""The client side of the Nous welcome tier's gateway contract: auxiliary calls on the welcome
host use its one model, the ``x-nous-model-switch`` header moves a session off ``nous/welcome``,
and refusal copy names the way forward (never guest / anonymous / claim)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import anon_auth

WELCOME = "https://welcome-api.nousresearch.com/v1"
PAID = "https://inference-api.nousresearch.com/v1"


# ── Auxiliary client: the welcome host serves exactly one model ──────────────────────────────────

class TestAuxiliaryOnWelcomeHost:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        import agent.auxiliary_client as ac
        with patch.object(ac, "_read_nous_auth", return_value={"auth_method": "anonymous"}), \
             patch.object(ac, "nous_rate_limit_remaining", return_value=None, create=True):
            yield

    def test_text_aux_on_welcome_host_pins_the_welcome_model(self):
        import agent.auxiliary_client as ac
        with patch.object(ac, "_resolve_nous_runtime_api", return_value=("jwt", WELCOME)), \
             patch.object(ac, "_create_openai_client", return_value="client") as create, \
             patch("hermes_cli.models.get_nous_recommended_aux_model") as recommended:
            client, model = ac._try_nous()
        assert (client, model) == ("client", anon_auth.GUEST_MODEL)
        assert create.call_args.kwargs["base_url"] == WELCOME
        recommended.assert_not_called()   # the Portal's pick would be a guaranteed 429 model_not_free
        assert ac.auxiliary_is_nous is True

    def test_vision_aux_on_welcome_host_uses_the_same_model(self):
        import agent.auxiliary_client as ac
        with patch.object(ac, "_resolve_nous_runtime_api", return_value=("jwt", WELCOME)), \
             patch.object(ac, "_create_openai_client", return_value="client"), \
             patch("hermes_cli.models.get_nous_recommended_aux_model") as recommended:
            assert ac._try_nous(vision=True) == ("client", anon_auth.GUEST_MODEL)
        recommended.assert_not_called()

    def test_paid_host_keeps_the_portal_recommendation(self):
        import agent.auxiliary_client as ac
        with patch.object(ac, "_resolve_nous_runtime_api", return_value=("jwt", PAID)), \
             patch.object(ac, "_create_openai_client", return_value="client"), \
             patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="some/free-model"):
            client, model = ac._try_nous()
        assert (client, model) == ("client", "some/free-model")


# ── Model-switch header ───────────────────────────────────────────────────────────────────────────

def _agent(model="nous/welcome", base_url=PAID):
    return SimpleNamespace(model=model, base_url=base_url, provider="nous", statuses=[])


class TestModelSwitchHeader:
    def test_header_is_recorded_not_applied_on_the_streaming_response(self):
        agent = _agent()
        assert anon_auth.note_model_switch(agent, {"X-Nous-Model-Switch": "z-ai/glm-5.3-flash"}) == "z-ai/glm-5.3-flash"
        assert agent.model == "nous/welcome"
        assert agent._nous_pending_model_switch == ("nous/welcome", "z-ai/glm-5.3-flash")

    def test_no_header_records_nothing(self):
        agent = _agent()
        assert anon_auth.note_model_switch(agent, {"content-type": "application/json"}) is None
        assert getattr(agent, "_nous_pending_model_switch", None) is None

    def test_apply_moves_the_session_and_the_config_default(self, monkeypatch):
        agent = _agent()
        agent._buffer_status = agent.statuses.append
        anon_auth.note_model_switch(agent, {"x-nous-model-switch": "z-ai/glm-5.3-flash"})
        writes = []
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"model": {"default": "nous/welcome"}})
        monkeypatch.setattr("hermes_cli.auth._update_config_for_provider",
                            lambda provider, url, default_model=None, **kw: writes.append((provider, url, default_model)))
        assert anon_auth.apply_model_switch(agent) == "z-ai/glm-5.3-flash"
        assert agent.model == "z-ai/glm-5.3-flash"
        assert agent._nous_model_switch == ("nous/welcome", "z-ai/glm-5.3-flash")   # what the gateway cache check reads
        assert writes == [("nous", PAID, "z-ai/glm-5.3-flash")]
        assert agent._nous_pending_model_switch is None
        assert agent.statuses and "z-ai/glm-5.3-flash" in agent.statuses[0]
        assert anon_auth.apply_model_switch(agent) is None   # one application per header

    def test_apply_leaves_a_user_chosen_default_alone(self, monkeypatch):
        agent = _agent()
        anon_auth.note_model_switch(agent, {"x-nous-model-switch": "z-ai/glm-5.3-flash"})
        writes = []
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"model": {"default": "openai/gpt-5"}})
        monkeypatch.setattr("hermes_cli.auth._update_config_for_provider",
                            lambda *a, **kw: writes.append(a))
        assert anon_auth.apply_model_switch(agent) == "z-ai/glm-5.3-flash"
        assert writes == []

    def test_apply_is_a_noop_when_the_session_already_moved(self):
        agent = _agent()
        anon_auth.note_model_switch(agent, {"x-nous-model-switch": "z-ai/glm-5.3-flash"})
        agent.model = "openai/gpt-5"   # a /model or the sign-in sweep won the race
        assert anon_auth.apply_model_switch(agent) is None
        assert agent.model == "openai/gpt-5"

    def test_mixin_capture_records_the_header(self):
        from agent.rate_limit_credits import RateLimitCreditsMixin

        class _Agent(RateLimitCreditsMixin):
            model = "nous/welcome"
            provider = "nous"
        a = _Agent()
        a._capture_nous_model_switch(SimpleNamespace(headers={"x-nous-model-switch": "z-ai/glm-5.3-flash"}))
        assert a._nous_pending_model_switch == ("nous/welcome", "z-ai/glm-5.3-flash")
        a._capture_nous_model_switch(None)   # fail-open


# ── Refusal copy ──────────────────────────────────────────────────────────────────────────────────

class TestRefusalCopy:
    def test_parse_reads_the_gateway_shape(self):
        body = {"status": 429, "message": "m", "reason": "model_not_free", "retry_after": 0,
                "alternates": ["nous/welcome"], "upgrade_url": "https://portal.example/upgrade"}
        assert anon_auth.parse_welcome_refusal(body) == {
            "reason": "model_not_free", "retry_after": 0, "alternates": ["nous/welcome"],
            "upgrade_url": "https://portal.example/upgrade"}
        assert anon_auth.parse_welcome_refusal({"reason": "nope"}) is None
        assert anon_auth.parse_welcome_refusal("not a dict") is None

    def test_copy_names_the_served_model_and_the_sign_in(self):
        refusal = anon_auth.parse_welcome_refusal({"reason": "model_not_free", "alternates": ["nous/welcome"]})
        chat = anon_auth.welcome_refusal_copy(refusal, model="gpt-5", in_chat=True)
        assert chat == "gpt-5 isn't on the Nous free tier; it serves nous/welcome only. Sign in with a Nous account for the full catalog: /login."
        terminal = anon_auth.welcome_refusal_copy(refusal, model="gpt-5", in_chat=False)
        assert "`hermes auth upgrade`" in terminal and "/login" not in terminal

    def test_capacity_copy_carries_the_retry(self):
        refusal = anon_auth.parse_welcome_refusal({"reason": "at_capacity", "retry_after": 30})
        assert "Retrying in 30s." in anon_auth.welcome_refusal_copy(refusal)

    @pytest.mark.parametrize("copy_fn, args", [
        (anon_auth.welcome_refusal_copy, ({"reason": r},)) for r in sorted(anon_auth.WELCOME_REFUSAL_REASONS)
    ] + [(anon_auth.welcome_route_refusal_copy, (k,)) for k in ("anon_on_paid_host", "named_on_welcome_host", "tier_disabled")])
    def test_copy_never_says_guest_anonymous_or_claim(self, copy_fn, args):
        text = copy_fn(*args).lower()
        assert not any(word in text for word in ("guest", "anonymous", "claim"))

    def test_route_refusal_detection(self):
        assert anon_auth.welcome_route_refusal(400, "Anonymous accounts must use https://x for inference.") == "anon_on_paid_host"
        assert anon_auth.welcome_route_refusal(403, "Anonymous accounts are not accepted by this API right now.") == "tier_disabled"
        assert anon_auth.welcome_route_refusal(429, "Anonymous accounts must use") is None
        assert anon_auth.welcome_route_refusal(400, "bad request") is None


class TestTurnRecoveryGuidance:
    def test_guidance_from_classified_context(self):
        from agent.turn_recovery import _welcome_tier_guidance
        classified = SimpleNamespace(error_context={"welcome_refusal": {"reason": "model_not_free", "retry_after": 0, "alternates": []}})
        assert "nous/welcome" in _welcome_tier_guidance(classified, model="gpt-5", in_chat=True)
        classified = SimpleNamespace(error_context={"welcome_route": "tier_disabled"})
        assert "switched off" in _welcome_tier_guidance(classified, model="", in_chat=True)
        assert _welcome_tier_guidance(SimpleNamespace(error_context={}), model="", in_chat=True) == ""
