"""``display.provider_error_detail: false`` keeps the provider's own words out of a failed turn's chat reply.

A non-retryable rejection ends the turn with curated copy plus ``Provider said: <raw provider reply>``.
That tail is useful to the person who runs the bot and noise (or a leak) for the people it answers,
so an operator can opt out per platform: chat then gets the same category reply as a raw envelope,
in ``display.language``. The default is unchanged.
"""

import pytest

import hermes_yaml as yaml
from agent.error_classifier import FailoverReason
from agent.i18n import reset_language_cache, t
from agent.turn_failure_copy import PROVIDER_SAID, nonretryable_copy
from gateway.config import Platform
from gateway.run import _gateway_provider_error_reply, _sanitize_gateway_final_response
from gateway.warning_notifications import provider_error_detail_shown

SUMMARY = ('HTTP 400: Error code: 400 - {"error": {"message": "Unsupported value: \'tool_choice\'", '
           '"type": "invalid_request_error", "request_id": "req_0f3c9a"}}')
FAILED_RESULT = {"failed": True, "completed": False, "failure_reason": "format_error",
                 "failure_retryable": False, "provider": "synthetic-provider", "model": "synthetic-model"}


class _Classified:
    reason = FailoverReason.format_error
    is_auth = False


COPY = nonretryable_copy(_Classified(), provider="openrouter", model="synthetic-model", summary=SUMMARY)


@pytest.fixture
def display(tmp_path, monkeypatch):
    def write(**settings):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": settings}))
        reset_language_cache()
    yield write
    reset_language_cache()


def test_the_default_keeps_the_provider_reply(display):
    display()
    assert PROVIDER_SAID in COPY
    assert _sanitize_gateway_final_response(Platform.WHATSAPP, COPY, agent_result=FAILED_RESULT) == COPY


def test_opting_out_sends_the_category_reply(display):
    display(provider_error_detail=False)
    reply = _sanitize_gateway_final_response(Platform.WHATSAPP, COPY, agent_result=FAILED_RESULT)
    assert reply == _gateway_provider_error_reply(SUMMARY)
    assert "Provider said" not in reply and "req_0f3c9a" not in reply


def test_the_category_reply_follows_display_language(display):
    display(provider_error_detail=False, language="de")
    reply = _sanitize_gateway_final_response(Platform.WHATSAPP, COPY, agent_result=FAILED_RESULT)
    assert reply == t("gateway.errors.provider_kept_failing", lang="de")
    assert reply != t("gateway.errors.provider_kept_failing", lang="en")


def test_a_platform_override_wins(display):
    display(provider_error_detail=False, platforms={"telegram": {"provider_error_detail": True}})
    assert _sanitize_gateway_final_response(Platform.TELEGRAM, COPY, agent_result=FAILED_RESULT) == COPY
    assert PROVIDER_SAID not in _sanitize_gateway_final_response(Platform.WHATSAPP, COPY, agent_result=FAILED_RESULT)


@pytest.mark.parametrize("agent_result", [None, {"failed": False}])
def test_text_from_a_turn_that_did_not_fail_is_left_alone(display, agent_result):
    display(provider_error_detail=False)
    assert _sanitize_gateway_final_response(Platform.WHATSAPP, COPY, agent_result=agent_result) == COPY


def test_raw_text_surfaces_keep_the_copy(display):
    display(provider_error_detail=False)
    assert _sanitize_gateway_final_response("api_server", COPY, agent_result=FAILED_RESULT) == COPY


@pytest.mark.parametrize(("value", "shown"), [
    (False, False), ("false", False), ("off", False), ("no", False),
    (True, True), (None, True), ("maybe", True), ("", True),
])
def test_only_an_explicit_opt_out_withholds_the_provider_reply(value, shown):
    assert provider_error_detail_shown("whatsapp", {"display": {"provider_error_detail": value}}) is shown
