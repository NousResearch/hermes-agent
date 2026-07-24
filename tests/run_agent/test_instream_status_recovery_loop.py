"""Loop-level regression for in-stream status recovery (#66358 review).

An OpenAI-compatible router can deliver an upstream 4xx as an error frame
inside an HTTP-200 SSE stream; the raised exception then has no
``status_code`` attribute. The classifier recovers the status from message
text — these tests pin that the *loop* actually uses the recovered value:
the turn aborts without burning retries, and user-facing status lines say
``HTTP 400``, never ``HTTP None``.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_instream_400_error() -> Exception:
    # No .status_code / .status attribute — only message text.
    return Exception("Error in stream response: Client error: HTTP 400")


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:39080/v1",
            provider="openrouter",
            api_mode="chat_completions",
            model="unsloth/gemma-4-12b-it-UD",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def test_instream_400_fails_fast_with_recovered_status_in_output():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = _make_instream_400_error()

    status_lines = []

    def _record(msg, *a, **k):
        status_lines.append(str(msg))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_vprint", side_effect=_record),
        patch.object(agent, "_buffer_status", side_effect=_record),
        patch.object(agent, "_emit_status", side_effect=_record),
        patch.object(agent, "_buffer_vprint", side_effect=_record),
    ):
        result = agent.run_conversation("hello")

    # Guard against a vacuous pass: the mocked error must be what aborted.
    assert agent.client.chat.completions.create.called
    assert result.get("failed") is True

    # Fail fast: a deterministic in-stream 4xx must not burn retries.
    assert agent.client.chat.completions.create.call_count == 1

    # The recovered status reaches user-facing output — never "HTTP None".
    joined = "\n".join(status_lines)
    assert "HTTP None" not in joined
    assert any("400" in line for line in status_lines)


def test_attribute_status_still_wins_over_message_text():
    """An exception carrying a real status attribute is untouched by the
    backfill — the attribute stays authoritative for the 413 check."""
    agent = _make_agent()
    err = Exception("HTTP 400 mentioned in text but attribute says 403")
    err.status_code = 403
    agent.client.chat.completions.create.side_effect = err

    status_lines = []

    def _record(msg, *a, **k):
        status_lines.append(str(msg))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_vprint", side_effect=_record),
        patch.object(agent, "_buffer_status", side_effect=_record),
        patch.object(agent, "_emit_status", side_effect=_record),
        patch.object(agent, "_buffer_vprint", side_effect=_record),
    ):
        result = agent.run_conversation("hello")

    assert result.get("failed") is True
    joined = "\n".join(status_lines)
    assert "HTTP None" not in joined
    assert "HTTP 403" in joined
