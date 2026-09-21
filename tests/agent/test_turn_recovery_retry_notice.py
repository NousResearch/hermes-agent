"""A provider retry backoff names itself on the live status line.

The buffered retry status replays only if every retry fails, so during the
backoff itself the user used to see an anonymous spinner — and right after a
tool that just finished (a connector sign-in landing, say) it read as the
agent going silent. The wait notice is transient (rewritten by the next
frame, cleared on recovery) and rides the frame long provider waits already
use, so it adds none of the transcript chatter the buffer exists to avoid."""

from unittest.mock import MagicMock, patch

import pytest

from agent.turn_recovery import compute_error_backoff


@pytest.mark.real_retry_backoff
def test_retry_backoff_names_the_wait_on_the_live_status_line():
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""

    wait = compute_error_backoff(
        agent, RuntimeError("502"), retry_count=1, max_retries=3,
        is_rate_limited=False, is_zai_coding_overload=False,
        base_url="https://example.test/v1", model="test/model",
    )

    assert wait > 0
    # Still buffered for the exhausted-retries replay …
    agent._buffer_status.assert_called_once()
    # … and named live while the backoff runs.
    agent._emit_wait_notice.assert_called_once()
    text = agent._emit_wait_notice.call_args.args[0]
    assert text.startswith("⏳ waiting on provider")
    assert "attempt 1/3" in text


def _hint_agent(requested: str, resolved: str):
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""
    agent.requested_provider = requested
    agent.provider = resolved
    return agent


def _buffered_lines(agent):
    return [c.args[0] for c in agent._buffer_status.call_args_list]


_TIMEOUT_ERR = Exception("Connection to https://api.example.test timed out after 30000ms")


def test_first_timeout_retry_names_auto_detect_alternatives():
    """#30797: first transport retry under an auto-detected provider lists the other configured
    keys — route named from the agent's own resolved provider, alternatives from the registry."""
    agent = _hint_agent("auto", "gemini")
    with patch("hermes_cli.auth.env_key_provider_candidates", return_value=["gemini", "zai"]):
        compute_error_backoff(
            agent, _TIMEOUT_ERR, retry_count=1, max_retries=3,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url="https://api.example.test/v1", model="test/model",
        )
    hints = [line for line in _buffered_lines(agent) if line.startswith("💡")]
    assert len(hints) == 1
    assert "'gemini' was auto-detected" in hints[0] and "zai" in hints[0]


def test_ambiguity_hint_only_on_first_retry():
    agent = _hint_agent("auto", "gemini")
    with patch("hermes_cli.auth.env_key_provider_candidates", return_value=["gemini", "zai"]):
        compute_error_backoff(
            agent, _TIMEOUT_ERR, retry_count=2, max_retries=3,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url="https://api.example.test/v1", model="test/model",
        )
    assert not [line for line in _buffered_lines(agent) if line.startswith("💡")]


def test_explicit_provider_request_suppresses_ambiguity_hint():
    """The review's first failure mode: an explicitly requested provider must never be told
    "no explicit provider set" (#30797)."""
    agent = _hint_agent("zai", "zai")
    with patch("hermes_cli.auth.env_key_provider_candidates", return_value=["gemini", "zai"]):
        compute_error_backoff(
            agent, _TIMEOUT_ERR, retry_count=1, max_retries=3,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url="https://api.example.test/v1", model="test/model",
        )
    assert not [line for line in _buffered_lines(agent) if line.startswith("💡")]


def test_no_hint_without_alternatives_or_for_non_transport_errors():
    agent = _hint_agent("auto", "gemini")
    with patch("hermes_cli.auth.env_key_provider_candidates", return_value=["gemini"]):
        compute_error_backoff(
            agent, _TIMEOUT_ERR, retry_count=1, max_retries=3,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url="https://api.example.test/v1", model="test/model",
        )
        compute_error_backoff(
            agent, Exception("HTTP 400: invalid_request"), retry_count=1, max_retries=3,
            is_rate_limited=False, is_zai_coding_overload=False,
            base_url="https://api.example.test/v1", model="test/model",
        )
    assert not [line for line in _buffered_lines(agent) if line.startswith("💡")]
