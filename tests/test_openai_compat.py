"""OpenAI API compatibility test suite.

Tests whether a custom API endpoint conforms to the OpenAI Chat Completions
interface specification. Use environment variables to configure the target:

    TEST_OPENAI_BASE_URL    - API base URL (default: https://ai-pool.evebattery.com/v1)
    TEST_OPENAI_API_KEY     - API key (required)
    TEST_OPENAI_MODEL       - Model name (default: Qwen3-235B-A22B-w8a8)
    TEST_OPENAI_SSL_VERIFY  - Set to "false" to disable SSL verification (default: true)
    TEST_OPENAI_TIMEOUT     - Request timeout in seconds (default: 60)
    TEST_OPENAI_NO_PROXY    - Set to "true" to bypass system proxy (default: false)

Run:
    $env:TEST_OPENAI_API_KEY="your-key"
    python -m pytest tests/test_openai_compat.py -v -o "addopts="
"""

import json
import os
import time

import pytest
import requests
import urllib3

# Disable SSL warnings when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = os.getenv("TEST_OPENAI_BASE_URL", "https://ai-pool.evebattery.com/v1")
API_KEY = os.getenv("TEST_OPENAI_API_KEY", "")
MODEL = os.getenv("TEST_OPENAI_MODEL", "Qwen3-235B-A22B-w8a8")
SSL_VERIFY = os.getenv("TEST_OPENAI_SSL_VERIFY", "true").lower() != "false"
TIMEOUT = int(os.getenv("TEST_OPENAI_TIMEOUT", "60"))
NO_PROXY = os.getenv("TEST_OPENAI_NO_PROXY", "false").lower() == "true"

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="TEST_OPENAI_API_KEY not set",
)

# Configure proxies
PROXIES = {"http": None, "https": None} if NO_PROXY else None


def _chat_completions(messages, model=None, stream=False, **kwargs):
    """Make a chat completion request and return parsed JSON response."""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model or MODEL,
        "messages": messages,
        **kwargs,
    }
    if stream:
        data["stream"] = True

    resp = requests.post(
        url, json=data, headers=headers, verify=SSL_VERIFY, timeout=TIMEOUT, proxies=PROXIES
    )
    resp.raise_for_status()
    return resp.json()


def _stream_completions(messages, model=None, **kwargs):
    """Make a streaming chat completion request and yield chunks."""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model or MODEL,
        "messages": messages,
        "stream": True,
        **kwargs,
    }

    resp = requests.post(
        url, json=data, headers=headers, verify=SSL_VERIFY, timeout=TIMEOUT, proxies=PROXIES, stream=True
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                yield json.loads(payload)


@pytest.fixture(scope="module")
def chat_response():
    """Make a basic chat completion request and return the response."""
    return _chat_completions(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0,
        max_tokens=64,
    )


# ── Test: Basic Connectivity ──────────────────────────────────────────────


class TestBasicConnectivity:
    """Verify the endpoint is reachable and responds."""

    def test_endpoint_reachable(self):
        """Should get a valid response from the endpoint."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
        assert response is not None
        assert "choices" in response

    def test_models_endpoint(self):
        """If /models is supported, it should return a list."""
        url = f"{BASE_URL}/models"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        try:
            resp = requests.get(url, headers=headers, verify=SSL_VERIFY, timeout=30, proxies=PROXIES)
            if resp.status_code == 404:
                pytest.skip("Endpoint does not support /models")
            resp.raise_for_status()
            data = resp.json()
            assert "data" in data
            assert isinstance(data["data"], list)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("Endpoint does not support /models")
            raise


# ── Test: Chat Completions Response Structure ─────────────────────────────


class TestChatCompletions:
    """Verify the response matches OpenAI Chat Completions spec."""

    def test_has_id_field(self, chat_response):
        """Response must have an id field (string)."""
        assert "id" in chat_response
        assert isinstance(chat_response["id"], str)
        assert len(chat_response["id"]) > 0

    def test_has_object_field(self, chat_response):
        """Response object type must be 'chat.completion'."""
        assert "object" in chat_response
        assert chat_response["object"] == "chat.completion"

    def test_has_created_field(self, chat_response):
        """Response must have a created timestamp (integer)."""
        assert "created" in chat_response
        assert isinstance(chat_response["created"], int)
        assert chat_response["created"] > 0

    def test_has_model_field(self, chat_response):
        """Response must echo back the model name."""
        assert "model" in chat_response
        assert isinstance(chat_response["model"], str)
        assert len(chat_response["model"]) > 0

    def test_has_choices_field(self, chat_response):
        """Response must contain a non-empty choices array."""
        assert "choices" in chat_response
        assert isinstance(chat_response["choices"], list)
        assert len(chat_response["choices"]) >= 1

    def test_has_usage_field(self, chat_response):
        """Response must contain usage statistics."""
        assert "usage" in chat_response
        assert chat_response["usage"] is not None

    def test_usage_has_token_counts(self, chat_response):
        """Usage must include prompt_tokens, completion_tokens, total_tokens."""
        usage = chat_response["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ── Test: Choice Object Structure ─────────────────────────────────────────


class TestChoiceStructure:
    """Verify each choice in the response matches the spec."""

    def test_choice_has_index(self, chat_response):
        """Each choice must have an index."""
        for i, choice in enumerate(chat_response["choices"]):
            assert "index" in choice
            assert choice["index"] == i

    def test_choice_has_message(self, chat_response):
        """Non-streaming response must have message (not delta)."""
        choice = chat_response["choices"][0]
        assert "message" in choice
        assert "delta" not in choice

    def test_message_has_role(self, chat_response):
        """Message must have a role field."""
        message = chat_response["choices"][0]["message"]
        assert "role" in message
        assert message["role"] == "assistant"

    def test_message_has_content(self, chat_response):
        """Message must have a content field (string or null)."""
        message = chat_response["choices"][0]["message"]
        assert "content" in message
        assert message["content"] is None or isinstance(message["content"], str)

    def test_choice_has_finish_reason(self, chat_response):
        """Choice must have a finish_reason."""
        choice = chat_response["choices"][0]
        assert "finish_reason" in choice
        valid_reasons = {"stop", "length", "tool_calls", "content_filter", None}
        assert choice["finish_reason"] in valid_reasons or isinstance(
            choice["finish_reason"], str
        )


# ── Test: Streaming Response ──────────────────────────────────────────────


class TestStreamingResponse:
    """Verify streaming mode returns valid SSE chunks."""

    def test_stream_returns_chunks(self):
        """Streaming should return valid chunks."""
        chunks = list(
            _stream_completions(
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=32,
            )
        )
        assert len(chunks) > 0

    def test_stream_chunk_has_choices(self):
        """Each streamed chunk must have choices."""
        for chunk in _stream_completions(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
        ):
            assert "choices" in chunk
            assert isinstance(chunk["choices"], list)

    def test_stream_chunk_has_delta(self):
        """Streamed chunks must use delta (not message)."""
        for chunk in _stream_completions(
            messages=[{"role": "user", "content": "Say hey"}],
            max_tokens=32,
        ):
            if chunk["choices"]:
                choice = chunk["choices"][0]
                assert "delta" in choice

    def test_stream_first_chunk_has_role(self):
        """First streamed chunk's delta should contain role='assistant'."""
        chunks = list(
            _stream_completions(
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=32,
            )
        )
        if chunks and chunks[0]["choices"]:
            delta = chunks[0]["choices"][0].get("delta", {})
            if "role" in delta and delta["role"]:
                assert delta["role"] == "assistant"

    def test_stream_content_is_string(self):
        """Streamed content deltas must be strings."""
        for chunk in _stream_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=32,
        ):
            if chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    assert isinstance(delta["content"], str)

    def test_stream_finish_reason_at_end(self):
        """Last non-empty chunk should have finish_reason."""
        last_choice = None
        for chunk in _stream_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=32,
        ):
            if chunk["choices"]:
                last_choice = chunk["choices"][0]
        if last_choice:
            assert "finish_reason" in last_choice


# ── Test: Parameter Compatibility ─────────────────────────────────────────


class TestParameterCompatibility:
    """Verify standard OpenAI parameters are accepted."""

    def test_temperature_parameter(self):
        """Should accept temperature in [0, 2]."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            temperature=0.5,
            max_tokens=16,
        )
        assert response["choices"][0]["message"]["content"] is not None

    def test_max_tokens_parameter(self):
        """Should respect max_tokens limit."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "Write a 100 word essay"}],
            max_tokens=10,
        )
        content = response["choices"][0]["message"].get("content") or ""
        assert len(content) < 2000

    def test_top_p_parameter(self):
        """Should accept top_p parameter."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            top_p=0.9,
            max_tokens=16,
        )
        assert response is not None

    def test_stop_parameter(self):
        """Should accept stop sequences."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "Count: 1 2 3 STOP 4 5 6"}],
            stop=["STOP"],
            max_tokens=64,
        )
        assert response["choices"][0]["finish_reason"] in ("stop", "length")

    def test_extra_body_parameter(self):
        """Should accept and pass through extra_body parameters."""
        response = _chat_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=16,
            chat_template_kwargs={"enable_thinking": False},
            truncate_prompt_tokens=128000,
        )
        assert response is not None
        assert response["choices"][0]["message"]["content"] is not None

    def test_multiple_messages(self):
        """Should handle multi-turn conversation."""
        response = _chat_completions(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=32,
        )
        content = response["choices"][0]["message"].get("content") or ""
        assert "Alice" in content or "alice" in content.lower()


# ── Test: Error Handling ──────────────────────────────────────────────────


class TestErrorHandling:
    """Verify error responses follow OpenAI error format."""

    def test_invalid_model_error(self):
        """Invalid model should raise an HTTP error."""
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            _chat_completions(
                model="nonexistent-model-xyz-12345",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=16,
            )
        # Accept 4xx (client error) or 5xx (server error like 503)
        assert exc_info.value.response.status_code >= 400

    def test_empty_messages_error(self):
        """Empty messages should raise an HTTP error."""
        with pytest.raises(requests.exceptions.HTTPError):
            _chat_completions(
                messages=[],
                max_tokens=16,
            )

    def test_invalid_role_error(self):
        """Invalid role may raise an HTTP error or be accepted (server-dependent)."""
        try:
            response = _chat_completions(
                messages=[{"role": "invalid_role", "content": "test"}],
                max_tokens=16,
            )
            # Some servers accept invalid roles gracefully
            assert "choices" in response
        except requests.exceptions.HTTPError:
            # Server rejected the invalid role
            pass


# ── Test: Response Timing ─────────────────────────────────────────────────


class TestResponseTiming:
    """Basic performance checks."""

    def test_response_latency(self):
        """Response should arrive within reasonable time."""
        start = time.time()
        _chat_completions(
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=8,
        )
        elapsed = time.time() - start
        assert elapsed < 30, f"Response took {elapsed:.1f}s (>30s threshold)"


# ── Standalone Runner ─────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-o", "addopts="])
