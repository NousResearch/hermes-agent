"""Local-model admission failures must remain terminal through retry handling."""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error


class TestLocalAdmissionError:
    def test_admission_block_is_terminal_and_never_falls_back(self):
        from hermes_cli.middleware import LocalModelAdmissionBlockedError

        result = classify_api_error(
            LocalModelAdmissionBlockedError("missing_admission_identity"),
            provider="ollama", model="gemma4:12b-64k",
        )

        assert result.reason == FailoverReason.local_admission_blocked
        assert result.retryable is False
        assert result.should_compress is False
        assert result.should_rotate_credential is False
        assert result.should_fallback is False

    def test_admission_exception_is_terminal_before_provider_hooks_or_fallback(self):
        from hermes_cli.middleware import LocalModelAdmissionBlockedError
        from agent.turn_api_error import settle_unrecovered_error

        class Agent:
            calls = []

            def _summarize_api_error(self, error):
                return str(error)

            def _flush_status_buffer(self):
                self.calls.append("flush")

            def _emit_diagnostic_status(self, text):
                self.calls.append(("diagnostic", text))

            def _has_pending_fallback(self):
                self.calls.append("fallback-check")
                return True

            def __getattr__(self, name):
                if name == "_recover_with_credential_pool":
                    return lambda **kwargs: self.calls.append("credential-pool")
                return lambda *args, **kwargs: self.calls.append(name)

        error = LocalModelAdmissionBlockedError("missing_admission_identity")
        classified = classify_api_error(error, provider="ollama", model="gemma4:12b-64k")
        agent = Agent()
        retry = SimpleNamespace(provider_stale_refresh_attempted=False)

        verdict = settle_unrecovered_error(
            agent, api_error=error, classified=classified, _retry=retry, status_code=None,
            error_msg=str(error), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider="ollama", _base="http://localhost:11434",
            _model="gemma4:12b-64k", messages=[], api_messages=[], api_kwargs={},
            active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=0, max_retries=3, compression_attempts=0, api_call_count=0,
        )

        assert verdict.action == "return"
        assert verdict.result["failure_reason"] == "local_admission_blocked"
        assert verdict.result["failure_retryable"] is False
        assert verdict.result["api_calls"] == 0
        assert agent.calls == [
            "flush",
            ("diagnostic", "❌ Local model admission blocked this request: missing_admission_identity"),
        ]
