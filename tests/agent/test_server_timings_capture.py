"""Tests for server-reported generation speed capture.

_capture_server_timings reads the llama-server ``timings`` block off a
completed response (SDK ``model_extra`` on the non-streaming path, the
streaming mimic's mirrored attribute otherwise) and records
``predicted_per_second`` on the agent — but only when the resolved provider
profile opts in via ``surfaces_server_timings``. Providers without the
opt-in never record a figure, so their display is unchanged.
"""

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


TIMINGS = {
    "prompt_n": 19,
    "prompt_ms": 41.2,
    "predicted_n": 245,
    "predicted_ms": 3960.1,
    "predicted_per_second": 61.87,
}


def _agent():
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "llamacpp"
    agent.requested_provider = "llamacpp"
    return agent


def _response(model_extra):
    return SimpleNamespace(model_extra=model_extra)


OPTED_IN = SimpleNamespace(surfaces_server_timings=True)
NOT_OPTED_IN = SimpleNamespace()


class TestCaptureServerTimings:
    def test_opted_in_profile_records_predicted_per_second(self):
        agent = _agent()
        with patch("providers.resolve_provider_profile", return_value=OPTED_IN):
            agent._capture_server_timings(_response({"timings": TIMINGS}))
        assert agent.last_server_tps == 61.87

    def test_profile_without_opt_in_records_nothing(self):
        agent = _agent()
        with patch("providers.resolve_provider_profile", return_value=NOT_OPTED_IN):
            agent._capture_server_timings(_response({"timings": TIMINGS}))
        assert agent.last_server_tps is None

    def test_response_without_timings_clears_previous_figure(self):
        agent = _agent()
        agent.last_server_tps = 61.87
        with patch("providers.resolve_provider_profile", return_value=OPTED_IN):
            agent._capture_server_timings(_response(None))
        assert agent.last_server_tps is None

    def test_streaming_mimic_shape_is_read_identically(self):
        agent = _agent()
        mimic = SimpleNamespace(
            id="stream-x",
            model="m",
            choices=[],
            usage=None,
            model_extra={"timings": TIMINGS},
        )
        with patch("providers.resolve_provider_profile", return_value=OPTED_IN):
            agent._capture_server_timings(mimic)
        assert agent.last_server_tps == 61.87

    def test_profile_resolution_failure_records_nothing(self):
        agent = _agent()
        with patch(
            "providers.resolve_provider_profile", side_effect=RuntimeError("boom")
        ):
            agent._capture_server_timings(_response({"timings": TIMINGS}))
        assert agent.last_server_tps is None

    def test_non_numeric_and_non_positive_rates_are_rejected(self):
        agent = _agent()
        for bad in (True, 0, -3.5, "62", None):
            with patch(
                "providers.resolve_provider_profile", return_value=OPTED_IN
            ):
                agent._capture_server_timings(
                    _response({"timings": {"predicted_per_second": bad}})
                )
            assert agent.last_server_tps is None, f"accepted {bad!r}"


class TestChunkServerTimings:
    """The streaming accumulator's per-chunk timings extractor."""

    def test_real_sdk_chunk_with_timings_extra(self):
        from openai.types.chat import ChatCompletionChunk

        from agent.chat_completion_helpers import _chunk_server_timings

        chunk = ChatCompletionChunk(
            id="chatcmpl-1",
            created=1730000000,
            model="m",
            object="chat.completion.chunk",
            choices=[],
            timings={"predicted_per_second": 42.5},
        )
        assert _chunk_server_timings(chunk) == {"predicted_per_second": 42.5}

    def test_chunk_without_timings_returns_none(self):
        from openai.types.chat import ChatCompletionChunk

        from agent.chat_completion_helpers import _chunk_server_timings

        chunk = ChatCompletionChunk(
            id="chatcmpl-1",
            created=1730000000,
            model="m",
            object="chat.completion.chunk",
            choices=[],
        )
        assert _chunk_server_timings(chunk) is None

    def test_foreign_shapes_never_raise(self):
        from agent.chat_completion_helpers import _chunk_server_timings

        assert _chunk_server_timings(object()) is None
        assert _chunk_server_timings(None) is None
        assert _chunk_server_timings(
            SimpleNamespace(model_extra={"timings": "not-a-dict"})
        ) is None
