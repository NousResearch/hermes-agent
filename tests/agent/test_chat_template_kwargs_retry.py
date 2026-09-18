"""Regression tests for the ``chat_template_kwargs`` rejection retry.

``auxiliary.<task>.extra_body`` is forwarded verbatim to
``chat.completions.create``. ``chat_template_kwargs`` (usually
``{"enable_thinking": false}`` for Qwen) is accepted by vLLM and LM Studio
style local servers, but other OpenAI-compatible endpoints reject it with a
hard 400:

    Error code: 400 - {'message': 'chat_template_kwargs: Extra inputs are
    not permitted'}

Production evidence (2026-09-10): the vision auxiliary lane sent
``chat_template_kwargs`` to an endpoint that rejects it and ``vision_analyze``
failed identically 90 times in a single turn.

The field is a rendering hint, never load-bearing for correctness, so the
right reaction is one retry without it — and to remember per base_url that
the server does not support it, so the next call skips it outright.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _is_chat_template_kwargs_rejection,
    _without_chat_template_kwargs,
    _chat_template_kwargs_unsupported,
    _mark_chat_template_kwargs_unsupported,
    _reset_chat_template_kwargs_support,
)


_REJECTION = (
    "Error code: 400 - {'message': 'chat_template_kwargs: Extra inputs are "
    "not permitted', 'type': 'invalid_request_error'}"
)


@pytest.fixture(autouse=True)
def _clean_support_cache():
    _reset_chat_template_kwargs_support()
    yield
    _reset_chat_template_kwargs_support()


class TestDetector:
    @pytest.mark.parametrize("message", [
        _REJECTION,
        "Error code: 400 - chat_template_kwargs: Extra inputs are not permitted",
        "400: Unrecognized request argument supplied: chat_template_kwargs",
        "chat_template_kwargs is not supported by this model",
    ])
    def test_matches_real_provider_phrasings(self, message):
        exc = Exception(message)
        exc.status_code = 400
        assert _is_chat_template_kwargs_rejection(exc) is True

    def test_ignores_unrelated_400s(self):
        exc = Exception("Error code: 400 - {'message': 'context length exceeded'}")
        exc.status_code = 400
        assert _is_chat_template_kwargs_rejection(exc) is False

    def test_ignores_non_400_statuses(self):
        exc = Exception("chat_template_kwargs: Extra inputs are not permitted")
        exc.status_code = 500
        assert _is_chat_template_kwargs_rejection(exc) is False

    def test_does_not_match_response_format_rejection(self):
        """Must not steal the structured-output retry's territory."""
        exc = Exception("output_config: Extra inputs are not permitted")
        exc.status_code = 400
        assert _is_chat_template_kwargs_rejection(exc) is False


class TestStripper:
    def test_removes_the_extra_body_entry(self):
        kwargs = {
            "model": "m",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        out = _without_chat_template_kwargs(kwargs)
        assert out is not None
        assert "extra_body" not in out
        # original untouched
        assert "chat_template_kwargs" in kwargs["extra_body"]

    def test_keeps_sibling_extra_body_fields(self):
        kwargs = {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 40,
            },
        }
        out = _without_chat_template_kwargs(kwargs)
        assert out["extra_body"] == {"top_k": 40}

    def test_returns_none_when_nothing_to_strip(self):
        assert _without_chat_template_kwargs({"model": "m"}) is None
        assert _without_chat_template_kwargs({"extra_body": {"top_k": 40}}) is None


class TestSupportMemo:
    def test_marks_and_reads_back_per_base_url(self):
        assert _chat_template_kwargs_unsupported("https://a.test/v1") is False
        _mark_chat_template_kwargs_unsupported("https://a.test/v1")
        assert _chat_template_kwargs_unsupported("https://a.test/v1") is True
        # Another server is unaffected.
        assert _chat_template_kwargs_unsupported("https://b.test/v1") is False

    def test_blank_base_url_is_never_memoised(self):
        _mark_chat_template_kwargs_unsupported("")
        assert _chat_template_kwargs_unsupported("") is False


def _dummy_response():
    return {"ok": True}


_CTK = {"chat_template_kwargs": {"enable_thinking": False}}

def _patched(client):
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "gpt-5.5", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client",
              return_value=(client, "gpt-5.5")),
        patch("agent.auxiliary_client._validate_llm_response",
              side_effect=lambda resp, _task, **_kw: resp),
    )


class TestSyncRetry:
    def test_retries_once_without_the_field_and_succeeds(self):
        client = MagicMock()
        client.base_url = "https://aux.test/v1"
        client.chat.completions.create.side_effect = [
            RuntimeError(_REJECTION), _dummy_response(),
        ]
        a, b, c = _patched(client)
        with a, b, c:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body=dict(_CTK),
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2
        first = client.chat.completions.create.call_args_list[0].kwargs
        retry = client.chat.completions.create.call_args_list[1].kwargs
        assert "chat_template_kwargs" in (first.get("extra_body") or {})
        assert "chat_template_kwargs" not in (retry.get("extra_body") or {})
        # The rejection is remembered for this server.
        assert _chat_template_kwargs_unsupported("https://aux.test/v1")

    def test_remembered_rejection_skips_the_field_on_the_next_call(self):
        _mark_chat_template_kwargs_unsupported("https://aux.test/v1")
        client = MagicMock()
        client.base_url = "https://aux.test/v1"
        client.chat.completions.create.side_effect = [_dummy_response()]
        a, b, c = _patched(client)
        with a, b, c:
            call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body=dict(_CTK),
            )

        # One call only — no 400 round-trip paid a second time.
        assert client.chat.completions.create.call_count == 1
        sent = client.chat.completions.create.call_args_list[0].kwargs
        assert "chat_template_kwargs" not in (sent.get("extra_body") or {})

    def test_unrelated_400_does_not_strip_the_field(self):
        client = MagicMock()
        client.base_url = "https://aux.test/v1"
        client.chat.completions.create.side_effect = RuntimeError(
            "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'"
        )
        a, b, c = _patched(client)
        with a, b, c, patch(
            "agent.auxiliary_client._try_payment_fallback", return_value=None
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=64,
                    extra_body=dict(_CTK),
                )
        assert not _chat_template_kwargs_unsupported("https://aux.test/v1")


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_retries_once_without_the_field_and_succeeds(self):
        client = MagicMock()
        client.base_url = "https://aux.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=[
            RuntimeError(_REJECTION), _dummy_response(),
        ])
        a, b, c = _patched(client)
        with a, b, c:
            result = await async_call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body=dict(_CTK),
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.await_count == 2
        retry = client.chat.completions.create.call_args_list[1].kwargs
        assert "chat_template_kwargs" not in (retry.get("extra_body") or {})
        assert _chat_template_kwargs_unsupported("https://aux.test/v1")
