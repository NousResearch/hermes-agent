"""Regression for #130669: failure/cancellation copy follows the rendering profile,
while legacy transcript recognition and English machine diagnostics remain stable.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent import i18n, secret_scope, turn_failure_copy as copy
from agent.error_classifier import FailoverReason
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def language_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    home = tmp_path / "en"
    home.mkdir()
    (home / "config.yaml").write_text("display:\n  language: en\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    i18n.reset_language_cache()
    yield home
    i18n.reset_language_cache()


def test_english_rendering_preserves_defaults_and_formatting(language_home):
    from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX, waiting_for_model_interrupt
    from agent.message_sanitization import close_interrupted_tool_sequence
    from agent.retry_utils import format_reset_window

    fields = dict(label="Acme", model="m", attempts=3, detail="raw {detail}\nprovider line",
                  tokens=3000, window=65536, resume=" (CLI: `hermes --resume abc`)",
                  preview="raw {preview}", limit=9)
    assert copy.failed_turn_notice([]) == copy.FAILED_TURN_NOTICE
    assert copy.failed_turn_notice([{"role": "tool"}]) == copy.PARTIAL_FAILED_TURN_NOTICE
    for code, default in copy._SITE_COPY.items():
        assert copy.site_copy(code, **fields) == default.format_map(copy._Defaults(fields))
    assert copy.site_copy("invalid_response").endswith("\n\nDetails: ")
    with pytest.raises(KeyError) as exc:
        copy.site_copy("unknown")
    assert exc.value.args == ("unknown",)
    for reason in (*copy._EXHAUSTED_LEADS, "unknown"):
        lead = copy._EXHAUSTED_LEADS.get(reason, copy._EXHAUSTED_DEFAULT_LEAD).format(label="Acme", attempts=3)
        for reset in (None, 119, 120, 30995):
            situation = (
                f"its usage limit resets in {format_reset_window(reset)}. "
                "Send /retry after that, or switch models with /model."
                if reset is not None and reset >= 120 else
                f"it looks temporarily unavailable. {copy._NEXT_STEPS_RETRY}"
            )
            assert copy.exhausted_copy(reason, label="Acme", attempts=3, summary=fields["detail"], reset_seconds=reset) == (
                f"{lead} — {situation} To avoid this in future, "
                f"add a backup provider with `hermes fallback add`.\n\nProvider said: {fields['detail']}"
            )
    for reason in (*copy._NONRETRYABLE_COPY, "unknown"):
        classified = SimpleNamespace(reason=FailoverReason(reason), is_auth=False)
        default = copy._NONRETRYABLE_COPY.get(reason, copy._NONRETRYABLE_DEFAULT_COPY)
        for suggestion in (None, "vendor/m"):
            prefix = f" If you typed the name yourself it may be missing its vendor prefix — did you mean '{suggestion}'?" if suggestion else ""
            body = default.format(label=copy.provider_label_for("openrouter"), model="m",
                                  home=copy.display_hermes_home(), prefix_hint=prefix)
            assert copy.nonretryable_copy(classified, provider="openrouter", model="m", summary=fields["detail"],
                                          prefix_suggestion=suggestion) == f"{body}\n\nProvider said: {fields['detail']}"
    for kind, provider in (("oauth", "openai-codex"), ("api_key", "openrouter")):
        classified = SimpleNamespace(reason=FailoverReason.auth, is_auth=True)
        body = copy._AUTH_COPY[kind].format(label=copy.provider_label_for(provider),
                                           relogin=copy.oauth_relogin_command(provider))
        assert copy.nonretryable_copy(classified, provider=provider, model="m", summary=fields["detail"]) == (
            f"{body}\n\nProvider said: {fields['detail']}"
        )
    assert copy.content_policy_copy(label="Acme", summary=fields["detail"]) == (
        f"Acme's safety filter refused this request, so the model didn't answer. "
        f"{copy.CONTENT_POLICY_NEXT_STEPS}\n\nProvider said: {fields['detail']}"
    )
    now = 1_800_000_000
    for seconds, wait in ((1, "1m"), (60, "1m"), (61, "2m"), (3600, "1h 00m"), (3661, "1h 02m")):
        reset_time = time.strftime('%H:%M', time.localtime(now + seconds))
        assert copy.limit_reset_copy(now + seconds, now=now) == f"Limit resets at {reset_time} (in {wait})."
    for seconds in (-1, 0):
        assert copy.limit_reset_copy(now + seconds, now=now) == ""
    assert waiting_for_model_interrupt(1.25) == f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}1.2s elapsed)."
    messages = [{"role": "tool", "content": "done", "tool_call_id": "a"}]
    assert close_interrupted_tool_sequence(messages)
    assert messages[-1]["content"] == "Operation interrupted."
    interrupt_fields = dict(detail=fields["detail"], attempt=2, limit=3, cycle=1, total=5, error_type="TimeoutError")
    interrupt_defaults = {
        "retry": f"Operation interrupted during retry ({fields['detail']}, attempt 2/3).",
        "empty_response": "Operation interrupted: retrying empty response from model (retry 2/3).",
        "provider_recovery": "Operation interrupted: waiting for the provider to recover (cycle 1/5).",
        "api_error": f"Operation interrupted: handling API error (TimeoutError: {fields['detail']}).",
        "api_retry": "Operation interrupted: retrying API call after error (retry 2/3).",
    }
    for key, expected in interrupt_defaults.items():
        assert i18n.t(f"turn_failure.interrupt.{key}", **interrupt_fields) == expected


@pytest.mark.parametrize("lang", ["zh-hant", "es"])
def test_rendering_and_recognition_follow_profile_without_changing_machine_errors(
    language_home, tmp_path, monkeypatch, lang,
):
    from agent.conversation_loop import (
        INTERRUPT_WAITING_FOR_MODEL_PREFIX, is_waiting_for_model_interrupt, waiting_for_model_interrupt,
    )
    from agent.message_sanitization import close_interrupted_tool_sequence
    from agent.turn_api_call import handle_api_interrupt
    from agent.turn_overflow import _Recovery, _recover_context_length
    from agent.turn_retry_state import TurnRetryState
    from agent.turn_recovery import nonretryable_client_error_result
    from agent.retry_utils import format_reset_window
    from gateway.run import _sanitize_gateway_final_response
    from gateway.run_turn import GatewayTurnMixin, is_context_overflow_failure_result
    from tui_gateway import server as tui_server
    from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry
    from run_agent import AIAgent

    other = tmp_path / lang
    other.mkdir()
    (other / "config.yaml").write_text(f"display:\n  language: {lang}\n", encoding="utf-8")
    raw = "HTTP 500: raw {detail} — 原文\nsecond provider line"
    english_interrupt = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}0.5s elapsed)."
    classified_default = SimpleNamespace(reason=FailoverReason.model_not_found, is_auth=False)
    english_nonretryable = copy.nonretryable_copy(
        classified_default, provider="openrouter", model="m", summary=raw, prefix_suggestion="vendor/m",
    )
    english_policy = copy.content_policy_copy(label="Acme", summary=raw)
    english_reset = copy.limit_reset_copy(1_800_003_661, now=1_800_000_000)
    # Real A → B → A config/home/secret-scope resolution under multiplexing. No language-reader mock.
    secret_scope.set_multiplex_active(True)
    secret_token = secret_scope.set_secret_scope({})
    try:
        assert copy.failed_turn_notice([]) == copy.FAILED_TURN_NOTICE
        token = set_hermes_home_override(other)
        try:
            assert i18n.get_language() == lang
            notice = copy.failed_turn_notice([])
            partial_notice = copy.failed_turn_notice([{"role": "tool"}])
            assert notice == i18n.t("turn_failure.failed_turn_notice", lang=lang)
            assert notice != i18n.t("turn_failure.failed_turn_notice", lang="en")
            assert partial_notice == i18n.t("turn_failure.partial_failed_turn_notice", lang=lang)
            assert partial_notice != copy.PARTIAL_FAILED_TURN_NOTICE
            for rows, expected in (([{"role": "user", "content": "q"}], notice),
                                   ([{"role": "user", "content": "q"}, {"role": "tool"}], partial_notice)):
                assert GatewayTurnMixin._hmwa_failed_turn_notice(None, {"messages": rows}) == expected
            for text in (notice, partial_notice, copy.FAILED_TURN_NOTICE, copy.PARTIAL_FAILED_TURN_NOTICE):
                assert copy.untyped_failed_turn_display_kind("assistant", text) == copy.FAILED_TURN_DISPLAY_KIND
                assert copy.untyped_failed_turn_display_kind("user", text) is None
                assert copy.untyped_failed_turn_display_kind("assistant", f"Quote: {text}") is None
            site = copy.site_copy("invalid_response", label="Acme", attempts=3, detail=raw)
            assert site == i18n.t("turn_failure.site.invalid_response", lang=lang, label="Acme", attempts=3,
                                  detail=raw, next_steps=i18n.t("turn_failure.next_steps.retry", lang=lang))
            assert site.endswith(raw)
            assert site != copy.site_copy("invalid_response", lang="en", label="Acme", attempts=3, detail=raw)
            assert "/retry" in site and "/model" in site
            for reset in (None, 30995):
                exhausted = copy.exhausted_copy("rate_limit", label="Acme", attempts=3, summary=raw, reset_seconds=reset)
                assert exhausted.endswith(raw) and "Acme" in exhausted
                localized = dict(lead=i18n.t("turn_failure.exhausted.lead.rate_limit", lang=lang, label="Acme", attempts=3),
                                 situation=i18n.t("turn_failure.exhausted.reset", lang=lang, reset=format_reset_window(reset))
                                 if reset else i18n.t("turn_failure.exhausted.unavailable", lang=lang,
                                                     next_steps=i18n.t("turn_failure.next_steps.retry", lang=lang)), summary=raw)
                english = dict(lead=i18n.t("turn_failure.exhausted.lead.rate_limit", lang="en", label="Acme", attempts=3),
                               situation=i18n.t("turn_failure.exhausted.reset", lang="en", reset=format_reset_window(reset))
                               if reset else i18n.t("turn_failure.exhausted.unavailable", lang="en",
                                                   next_steps=i18n.t("turn_failure.next_steps.retry", lang="en")), summary=raw)
                assert exhausted == i18n.t("turn_failure.exhausted.tail", lang=lang, **localized)
                assert exhausted != i18n.t("turn_failure.exhausted.tail", lang="en", **english)
                assert "/retry" in exhausted and "`hermes fallback add`" in exhausted
            hints = []
            failure_agent = SimpleNamespace(
                log_prefix="", _flush_status_buffer=lambda: None, _emit_diagnostic_status=lambda *a: None,
                _summarize_api_error=str, _persist_session=lambda *a: None,
                _vprint=lambda text, **kw: hints.append(text), _extract_api_error_context=lambda *a: {},
            )
            cases = [(reason, "openrouter", f"turn_failure.nonretryable.{reason}")
                     for reason in copy._NONRETRYABLE_COPY]
            cases += [("unknown", "openrouter", "turn_failure.nonretryable.default"),
                      ("auth", "openrouter", "turn_failure.auth.api_key"),
                      ("auth", "openai-codex", "turn_failure.auth.oauth")]
            for reason, provider, key in cases:
                classified = SimpleNamespace(reason=FailoverReason(reason), is_auth=reason == "auth", retryable=False)
                for suggestion in (None, "vendor/m"):
                    fields = dict(label=copy.provider_label_for(provider), model="m", home=copy.display_hermes_home(),
                                  relogin=copy.oauth_relogin_command(provider))
                    localized = dict(fields, prefix_hint=i18n.t("turn_failure.nonretryable.prefix_hint", lang=lang,
                                                               suggestion=suggestion) if suggestion else "")
                    english = dict(fields, prefix_hint=i18n.t("turn_failure.nonretryable.prefix_hint", lang="en",
                                                             suggestion=suggestion) if suggestion else "")
                    text = copy.nonretryable_copy(classified, provider=provider, model="m", summary=raw,
                                                 prefix_suggestion=suggestion)
                    assert text == i18n.t("turn_failure.nonretryable.tail", lang=lang, summary=raw,
                                          body=i18n.t(key, lang=lang, **localized))
                    assert text != i18n.t("turn_failure.nonretryable.tail", lang="en", summary=raw,
                                          body=i18n.t(key, lang="en", **english))
                    assert text.endswith(raw)
                    for command in ("/model", "/new", "`hermes model`", "`hermes doctor`", "`hermes setup`"):
                        if command in i18n.t(key, lang="en", **english):
                            assert command in text
                    if key.endswith("model_not_found") and suggestion:
                        assert suggestion in text
                    if key.endswith("oauth"):
                        assert f"`{fields['relogin']}`" in text
                result = nonretryable_client_error_result(
                    failure_agent, Exception(raw), classified, status_code=None, api_kwargs=None, api_messages=[],
                    messages=[], conversation_history=None, api_call_count=1, approx_tokens=10,
                    provider=provider, base_url="https://example.invalid", model="m", delivered="partial answer",
                )
                assert result["error"] == raw and result["failure_reason"] == reason
                assert result["partial"] and result["failure_retryable"] is False
                assert result["final_response"] == "partial answer\n\n" + copy.nonretryable_copy(
                    classified, provider=provider, model="m", summary=raw,
                )
                assert not is_context_overflow_failure_result(result, history_len=2)
            text = copy.content_policy_copy(label="Acme", summary=raw)
            assert text == i18n.t("turn_failure.content_policy.body", lang=lang, label="Acme", summary=raw,
                                  next_steps=i18n.t("turn_failure.content_policy.next_steps", lang=lang))
            assert text != i18n.t("turn_failure.content_policy.body", lang="en", label="Acme", summary=raw,
                                  next_steps=i18n.t("turn_failure.content_policy.next_steps", lang="en"))
            assert text.endswith(raw) and "/model" in text
            result = nonretryable_client_error_result(
                failure_agent, Exception(raw),
                SimpleNamespace(reason=FailoverReason.content_policy_blocked, is_auth=False, retryable=False),
                status_code=None, api_kwargs=None, api_messages=[], messages=[], conversation_history=None,
                api_call_count=1, approx_tokens=10, provider="openrouter", base_url="https://example.invalid", model="m",
            )
            assert result["error"] == f"content_policy_blocked: {raw}"
            assert result["final_response"] == "⚠️ " + copy.content_policy_copy(label=copy.provider_label_for("openrouter"), summary=raw)
            assert f"   💡 {i18n.t('turn_failure.content_policy.next_steps', lang=lang)}" in hints
            assert not is_context_overflow_failure_result(result, history_len=2)
            now = 1_800_000_000
            for seconds, wait in ((1, "1m"), (60, "1m"), (61, "2m"), (3600, "1h 00m"), (3661, "1h 02m")):
                fields = dict(time=time.strftime('%H:%M', time.localtime(now + seconds)), wait=wait)
                text = copy.limit_reset_copy(now + seconds, now=now)
                assert text == i18n.t("turn_failure.limit_reset", lang=lang, **fields)
                assert text != i18n.t("turn_failure.limit_reset", lang="en", **fields)
            for seconds in (-1, 0):
                assert copy.limit_reset_copy(now + seconds, now=now) == ""
            interrupt = waiting_for_model_interrupt(1.25)
            assert interrupt == i18n.t("turn_failure.interrupt.waiting_for_model", lang=lang,
                                       prefix=i18n.t("turn_failure.interrupt.waiting_for_model_prefix", lang=lang), elapsed="1.2")
            assert interrupt != i18n.t("turn_failure.interrupt.waiting_for_model", lang="en",
                                       prefix=i18n.t("turn_failure.interrupt.waiting_for_model_prefix", lang="en"), elapsed="1.2")
            for text in (english_interrupt, interrupt):
                assert is_waiting_for_model_interrupt(text)
                assert _sanitize_gateway_final_response("telegram", text) == ""
                assert tui_server._turn_outcome({"final_response": text, "interrupted": True})[:2] == ("", "interrupted")
                assert not is_waiting_for_model_interrupt(f"Quote: {text}")
            messages = [{"role": "tool", "content": "done", "tool_call_id": "a"}]
            assert close_interrupted_tool_sequence(messages)
            assert messages[-1]["content"] == i18n.t("turn_failure.interrupt.operation")
            assert messages[-1]["content"] != i18n.t("turn_failure.interrupt.operation", lang="en")
            child = SimpleNamespace(model="m", session_estimated_cost_usd=0.0, session_cost_status="unknown",
                                    session_prompt_tokens=1, session_completion_tokens=1, _delegate_role="leaf")
            entry = _build_result_entry(
                child, {"final_response": interrupt, "interrupted": True, "completed": False,
                        "messages": [{"role": "assistant", "content": "partial answer"}, *messages]},
                0, 1.0, _SchemaOutcome(None, None, [], 0),
            )
            assert entry["summary"] == "partial answer" and entry["error"] == interrupt
            # Exercise the actual API interrupt renderer after the module was imported in English.
            agent = object.__new__(AIAgent)
            agent._has_pending_redirect = lambda: False
            agent.thinking_callback = None
            agent.log_prefix = ""
            agent._vprint = lambda *a, **k: None
            agent._drop_trailing_empty_response_scaffolding = lambda *a: None
            agent._strip_think_blocks = lambda text: text
            agent._current_streamed_assistant_text = ""
            agent._persist_session = lambda *args: None
            with monkeypatch.context() as clock:
                clock.setattr("agent.turn_api_call.time.time", lambda: 1.25)
                verdict = handle_api_interrupt(
                    agent, _retry=TurnRetryState(), thinking_spinner=None, messages=[], conversation_history=[],
                    api_start_time=0, interrupted=False, final_response=None,
                )
            assert is_waiting_for_model_interrupt(verdict.final_response)
            assert verdict.final_response == interrupt
            # Real overflow recovery, with only the token measurement controlled.
            agent = SimpleNamespace(
                model="m", log_prefix="", _flush_status_buffer=lambda: None,
                _vprint=lambda *a, **k: None, _persist_session=lambda *a: None,
                _buffer_vprint=lambda *a: None,
                max_tokens=None, context_compressor=SimpleNamespace(context_length=65536),
                provider="lmstudio", base_url="http://127.0.0.1:1234/v1", tools=None,
            )
            recovery = _Recovery(
                agent=agent, api_messages=[], system_message=None, effective_task_id="t", api_call_count=2,
                max_compression_attempts=3, messages=[], active_system_prompt=None, conversation_history=None,
                approx_tokens=3000, compression_attempts=3,
            )
            result = recovery.count_attempt().result
            assert result["final_response"] != result["error"]
            assert result["error"] == copy.site_copy("context_overflow", lang="en", model="m")
            assert is_context_overflow_failure_result(result, history_len=2)
            monkeypatch.setattr("agent.model_metadata.estimate_request_tokens_rough", lambda *a, **k: 3000)
            result = _recover_context_length(recovery, TurnRetryState(), "HTTP 500: Context size has been exceeded.").result
            assert result["final_response"] != result["error"]
            assert "3,000" in result["final_response"] and "65,536" in result["final_response"]
            assert result["error"] == copy.site_copy("server_context_rejection", lang="en", model="m", tokens=3000, window=65536)
            assert not is_context_overflow_failure_result(result, history_len=2)
            # Partial user overlays retain the site's missing-field tolerance.
            (other / "locales").mkdir()
            (other / "locales" / f"{lang}.yaml").write_text(
                'turn_failure:\n  site:\n    invalid_response: "{detail}|{missing_field}|{next_steps}"\n', encoding="utf-8",
            )
            i18n.reset_language_cache()
            assert copy.site_copy("invalid_response", detail=raw) == f"{raw}||{i18n.t('turn_failure.next_steps.retry')}"
        finally:
            reset_hermes_home_override(token)
        assert i18n.get_language() == "en"
        assert copy.failed_turn_notice([]) == copy.FAILED_TURN_NOTICE
        assert waiting_for_model_interrupt(0.5) == english_interrupt
        assert copy.nonretryable_copy(
            classified_default, provider="openrouter", model="m", summary=raw, prefix_suggestion="vendor/m",
        ) == english_nonretryable
        assert copy.content_policy_copy(label="Acme", summary=raw) == english_policy
        assert copy.limit_reset_copy(1_800_003_661, now=1_800_000_000) == english_reset
    finally:
        secret_scope.reset_secret_scope(secret_token)
        secret_scope.set_multiplex_active(False)
