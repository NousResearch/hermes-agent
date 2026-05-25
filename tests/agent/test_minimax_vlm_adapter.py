"""Tests for MinimaxVlmAuxiliaryClient — the VLM adapter for MiniMax vision.

MiniMax operates two separate APIs:
  /anthropic/v1/messages   — text chat (silently strips image blocks)
  /v1/coding_plan/vlm      — dedicated vision/multimodal endpoint

The VLM adapter translates OpenAI-shape ``chat.completions.create()`` calls
into ``/v1/coding_plan/vlm`` POSTs and returns a chat-completions-shaped
response object. See issue #15715.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.auxiliary_client import (
    MinimaxVlmAuxiliaryClient,
    AsyncMinimaxVlmAuxiliaryClient,
    _MINIMAX_VLM_BASE_URLS,
    _derive_minimax_vlm_endpoint,
    _resolve_minimax_vlm_client,
    resolve_vision_provider_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAABlBMVEUAAAD///+l2Z/dAAAACklEQVR4nGNgAAAAAgABc3UBGAAAAABJRU5ErkJggg=="


def _vision_messages(prompt: str = "What is this?", url: str = _DATA_URL) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Endpoint URL selection
# ---------------------------------------------------------------------------


class TestMinimaxVlmEndpoints:
    """The adapter must POST to the correct regional VLM endpoint."""

    def test_global_provider_uses_minimax_io(self):
        url = _MINIMAX_VLM_BASE_URLS["minimax"]
        assert url == "https://api.minimax.io/v1/coding_plan/vlm"

    def test_china_provider_uses_minimaxi_com(self):
        url = _MINIMAX_VLM_BASE_URLS["minimax-cn"]
        assert url == "https://api.minimaxi.com/v1/coding_plan/vlm"

    def test_global_provider_post_target(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="mm-key", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            args, kwargs = mock_post.call_args
            assert args[0] == "https://api.minimax.io/v1/coding_plan/vlm"

    def test_china_provider_post_target(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="mm-cn-key", provider="minimax-cn")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            args, _ = mock_post.call_args
            assert args[0] == "https://api.minimaxi.com/v1/coding_plan/vlm"


# ---------------------------------------------------------------------------
# Request body shape
# ---------------------------------------------------------------------------


class TestMinimaxVlmRequestBody:
    """The wrapper must extract the prompt + image_url and POST the
    MiniMax-native ``{"prompt": str, "image_url": str}`` body."""

    def test_request_body_has_prompt_and_image_url(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(prompt="Describe the image."),
            )
            _, kwargs = mock_post.call_args
            body = kwargs["json"]
            assert body == {
                "prompt": "Describe the image.",
                "image_url": _DATA_URL,
            }

    def test_authorization_header_is_bearer_api_key(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="my-secret", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            assert headers["Authorization"] == "Bearer my-secret"
            assert headers["Content-Type"] == "application/json"

    def test_string_content_is_supported_when_image_url_kwarg_present(self):
        """Some callers pass plain-string content + a separate image. Make sure
        the adapter still finds the data URL inside any list-of-blocks shape."""
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _DATA_URL}},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            adapter.chat.completions.create(model="MiniMax-M2.7", messages=messages)
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["prompt"] == "Describe."
            assert kwargs["json"]["image_url"] == _DATA_URL


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestMinimaxVlmResponseShape:
    """Adapter must return an object that quacks like a chat.completions
    response — ``response.choices[0].message.content`` must be set, ``usage``
    must not crash callers that read prompt/completion/total tokens."""

    def test_content_is_extracted_from_response(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "content": "A red cat sitting on a chair.",
                    "base_resp": {"status_code": 0, "status_msg": "ok"},
                },
            )
            response = adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            assert response.choices[0].message.content == "A red cat sitting on a chair."
            assert response.choices[0].finish_reason == "stop"
            assert response.choices[0].message.role == "assistant"
            assert response.choices[0].message.tool_calls is None

    def test_usage_attribute_is_present_and_safe_to_read(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            response = adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            assert response.usage is not None
            assert response.usage.prompt_tokens == 0
            assert response.usage.completion_tokens == 0
            assert response.usage.total_tokens == 0

    def test_model_field_is_propagated(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "ok", "base_resp": {"status_code": 0}},
            )
            response = adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=_vision_messages(),
            )
            assert response.model == "MiniMax-M2.7"


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


class TestMinimaxVlmErrors:
    """A non-zero ``base_resp.status_code`` must surface as a useful exception
    (not be silently returned as content). HTTP errors should also raise."""

    def test_nonzero_base_resp_status_raises(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "content": "",
                    "base_resp": {"status_code": 1004, "status_msg": "auth failed"},
                },
            )
            with pytest.raises(RuntimeError) as exc_info:
                adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            assert "1004" in str(exc_info.value)
            assert "auth failed" in str(exc_info.value)

    def test_http_error_status_raises(self):
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=500,
                text="upstream error",
                json=lambda: (_ for _ in ()).throw(ValueError("not json")),
            )
            with pytest.raises(RuntimeError) as exc_info:
                adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            assert "500" in str(exc_info.value)

    def test_missing_image_url_raises(self):
        """A vision call to MiniMax without an image is a programming error —
        the wrapper exists specifically to route image traffic. Surface a
        clear error rather than POSTing a malformed body."""
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        messages = [{"role": "user", "content": "text only"}]
        with pytest.raises(ValueError) as exc_info:
            adapter.chat.completions.create(
                model="MiniMax-M2.7",
                messages=messages,
            )
        assert "image" in str(exc_info.value).lower()

    def test_httpx_timeout_is_wrapped_as_runtime_error(self):
        """``httpx.post`` raises ``httpx.TimeoutException`` on network timeout —
        a raw httpx exception escaping into the vision tool would crash callers
        that only handle ``RuntimeError``. Wrap it with an actionable message."""
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("read timeout")
            with pytest.raises(RuntimeError) as exc_info:
                adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            msg = str(exc_info.value)
            assert "timed out" in msg.lower()
            assert "MiniMax VLM" in msg

    def test_httpx_transport_error_is_wrapped_as_runtime_error(self):
        """Connection errors (DNS, refused, reset) are ``httpx.HTTPError``
        subclasses — must be wrapped so callers get a useful diagnostic."""
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(RuntimeError) as exc_info:
                adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            msg = str(exc_info.value)
            assert "request failed" in msg.lower()
            assert "MiniMax VLM" in msg

    def test_missing_base_resp_logs_warning(self, caplog):
        """If MiniMax (or a proxy) omits ``base_resp`` entirely, the adapter
        proceeds optimistically (status_code is None, treated as success) —
        but logs a warning so operators can spot upstream-shape drift."""
        adapter = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "hello"},  # no base_resp
            )
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                response = adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            assert response.choices[0].message.content == "hello"
            assert any(
                "base_resp missing" in rec.message for rec in caplog.records
            ), f"Expected base_resp warning, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class TestAsyncMinimaxVlmAdapter:
    """The async wrapper delegates to the sync wrapper via to_thread()."""

    def test_async_wrapper_exposes_chat_completions(self):
        sync = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        async_adapter = AsyncMinimaxVlmAuxiliaryClient(sync)
        assert hasattr(async_adapter, "chat")
        assert hasattr(async_adapter.chat, "completions")
        assert hasattr(async_adapter.chat.completions, "create")
        assert async_adapter.api_key == "k"
        assert async_adapter.base_url == _MINIMAX_VLM_BASE_URLS["minimax"]

    def test_async_create_returns_completion_shape(self):
        import asyncio

        sync = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        async_adapter = AsyncMinimaxVlmAuxiliaryClient(sync)
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"content": "async ok", "base_resp": {"status_code": 0}},
            )
            response = asyncio.run(
                async_adapter.chat.completions.create(
                    model="MiniMax-M2.7",
                    messages=_vision_messages(),
                )
            )
            assert response.choices[0].message.content == "async ok"

    def test_async_httpx_timeout_is_wrapped_as_runtime_error(self):
        """Async path runs the sync adapter via ``asyncio.to_thread`` —
        verify the timeout wrapping survives the thread boundary so async
        callers also get a ``RuntimeError`` instead of raw ``httpx.TimeoutException``."""
        import asyncio

        sync = MinimaxVlmAuxiliaryClient(api_key="k", provider="minimax")
        async_adapter = AsyncMinimaxVlmAuxiliaryClient(sync)
        with patch("agent.auxiliary_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("read timeout")
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(
                    async_adapter.chat.completions.create(
                        model="MiniMax-M2.7",
                        messages=_vision_messages(),
                    )
                )
            assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Vision resolver integration
# ---------------------------------------------------------------------------


class TestVisionResolverWrapsMinimax:
    """When the user explicitly requests MiniMax for vision, the resolver must
    return the VLM adapter — not a plain OpenAI client pointing at the
    Anthropic-compat endpoint (which silently drops images)."""

    def test_explicit_minimax_vision_returns_vlm_adapter(self):
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax", "MiniMax-M2.7", None, "mm-key", None),
        ):
            provider, client, _model = resolve_vision_provider_client(
                provider="minimax", model="MiniMax-M2.7"
            )
            assert provider == "minimax"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)

    def test_explicit_minimax_cn_vision_returns_vlm_adapter(self):
        with patch.dict(
            "os.environ",
            {"MINIMAX_CN_API_KEY": "mm-cn-key", "OPENAI_API_KEY": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax-cn", "MiniMax-M2.7", None, "mm-cn-key", None),
        ):
            provider, client, _model = resolve_vision_provider_client(
                provider="minimax-cn", model="MiniMax-M2.7"
            )
            assert provider == "minimax-cn"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)

    def test_async_minimax_vision_returns_async_vlm_adapter(self):
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax", "MiniMax-M2.7", None, "mm-key", None),
        ):
            provider, client, _model = resolve_vision_provider_client(
                provider="minimax", model="MiniMax-M2.7", async_mode=True
            )
            assert provider == "minimax"
            assert isinstance(client, AsyncMinimaxVlmAuxiliaryClient)


# ---------------------------------------------------------------------------
# Base URL override (proxy / corporate-shim support)
# ---------------------------------------------------------------------------


class TestMinimaxVlmBaseUrlOverride:
    """When a user configures a custom MiniMax base_url (env override or
    corporate proxy), vision must route through *the same root* as text —
    otherwise the proxy is bypassed for image traffic. See P2 review on
    the #15715 fix."""

    def test_resolved_base_url_is_used_when_present(self):
        """resolve_api_key_provider_credentials returns both api_key and
        base_url; the resolver must capture the base_url and derive the VLM
        endpoint from it instead of using the hardcoded constant."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "https://internal-proxy.example.com/minimax/anthropic",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.base_url == (
                "https://internal-proxy.example.com/minimax/v1/coding_plan/vlm"
            )

    def test_anthropic_suffix_is_stripped_before_appending_vlm_path(self):
        """When base_url ends with ``/anthropic`` (the default Anthropic-compat
        path), strip it and substitute the VLM path so the endpoint lands as
        a sibling under the same host."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://api.minimax.io/anthropic"
        )
        assert endpoint == "https://api.minimax.io/v1/coding_plan/vlm"

    def test_custom_proxy_path_is_preserved(self):
        """Custom proxies that don't end in ``/anthropic`` must keep their
        full path prefix — the proxy is responsible for routing both endpoints
        under the same shim."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://proxy.example.com/minimax-shim"
        )
        assert endpoint == (
            "https://proxy.example.com/minimax-shim/v1/coding_plan/vlm"
        )

    def test_trailing_slash_is_stripped_before_appending(self):
        """Defensive: a trailing slash on the configured base_url must not
        produce ``//`` in the derived endpoint."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://api.minimax.io/anthropic/"
        )
        assert endpoint == "https://api.minimax.io/v1/coding_plan/vlm"

    def test_falls_back_to_default_when_no_base_url_resolved(self):
        """If resolve_api_key_provider_credentials returns ``base_url=None``
        (rare; user has no env config), use the hardcoded provider default."""
        endpoint = _derive_minimax_vlm_endpoint("minimax", None)
        assert endpoint == _MINIMAX_VLM_BASE_URLS["minimax"]
        assert endpoint == "https://api.minimax.io/v1/coding_plan/vlm"

    def test_falls_back_to_default_when_resolver_returns_empty_base_url(self):
        """Empty-string base_url should also trigger the fallback path."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.base_url == _MINIMAX_VLM_BASE_URLS["minimax"]

    def test_china_provider_with_custom_base_url(self):
        """Region detection still works correctly for the CN provider — the
        VLM endpoint should be derived from the CN-region base_url, not the
        international default."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax-cn",
                "api_key": "mm-cn-key",
                "base_url": "https://cn-proxy.example.com/minimax/anthropic",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            client = _resolve_minimax_vlm_client("minimax-cn")
            assert client is not None
            assert client.base_url == (
                "https://cn-proxy.example.com/minimax/v1/coding_plan/vlm"
            )
            assert client._provider == "minimax-cn"

    def test_explicit_api_key_still_honours_resolved_base_url(self):
        """When an api_key is passed explicitly (skipping the auth lookup for
        the key), the resolver must still try to fetch a configured base_url
        — otherwise an explicit key path would silently bypass the proxy."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "ignored-because-explicit-passed-in",
                "base_url": "https://proxy.example.com/mm/anthropic",
                "source": "env",
            },
        ):
            client = _resolve_minimax_vlm_client(
                "minimax", explicit_api_key="explicit-key"
            )
            assert client is not None
            assert client.api_key == "explicit-key"
            assert client.base_url == (
                "https://proxy.example.com/mm/v1/coding_plan/vlm"
            )


# ---------------------------------------------------------------------------
# Auto-path: model.base_url plumbing for vision auto-detection
# ---------------------------------------------------------------------------


class TestMinimaxVlmAutoPathMainBaseUrl:
    """When ``auxiliary.vision.provider`` is ``auto`` and the user's main
    provider is MiniMax, vision must honour ``model.base_url`` from config —
    same proxy as text traffic (P2 follow-up to #15715).

    The auto path goes through ``_read_main_provider``/``_read_main_model`` /
    ``_read_main_base_url`` to learn the main-model config; the explicit path
    only sees per-task overrides.
    """

    def test_auto_path_honors_main_model_base_url(self):
        """``model.base_url`` from config flows into the VLM endpoint
        derivation, so corporate proxies see vision traffic too."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://proxy.example.com/anthropic",
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert provider == "minimax"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)
            assert client.base_url == (
                "https://proxy.example.com/v1/coding_plan/vlm"
            )

    def test_auto_path_main_base_url_takes_precedence_over_provider_default(self):
        """When ``model.base_url`` is set and the env var is unset, the
        provider default (``https://api.minimax.io/anthropic``) must NOT be
        used — vision should land on the configured proxy."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://proxy.example.com/anthropic",
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert client is not None
            # Provider default would yield this — must NOT be used.
            assert client.base_url != _MINIMAX_VLM_BASE_URLS["minimax"]
            assert "api.minimax.io" not in client.base_url

    def test_auto_path_falls_back_to_env_when_no_main_base_url(self):
        """If ``model.base_url`` is unset but ``MINIMAX_BASE_URL`` is set
        (env override), the env var still wins and is plumbed through the
        credential resolver."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="",
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "https://env-proxy.example.com/anthropic",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert client is not None
            assert client.base_url == (
                "https://env-proxy.example.com/v1/coding_plan/vlm"
            )

    def test_auto_path_falls_back_to_default_when_neither_set(self):
        """Neither ``model.base_url`` nor the env var configured → use the
        hardcoded provider default."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="",
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert client is not None
            assert client.base_url == _MINIMAX_VLM_BASE_URLS["minimax"]

    def test_auto_path_env_var_wins_over_main_base_url(self):
        """When BOTH the env var AND ``model.base_url`` are set, the env var
        wins — same precedence as text traffic (#6039 in
        ``hermes_cli/runtime_provider.py``: pool URL from env beats config-
        configured ``model.base_url``)."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://config-proxy.example.com/anthropic",
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "https://env-proxy.example.com/anthropic",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert client is not None
            # Env var wins.
            assert client.base_url == (
                "https://env-proxy.example.com/v1/coding_plan/vlm"
            )
            assert "config-proxy" not in client.base_url


# ---------------------------------------------------------------------------
# Terminal /v1 stripping (P2 follow-up: documented chat-completions base_url)
# ---------------------------------------------------------------------------


class TestMinimaxVlmV1SuffixStripping:
    """``MINIMAX_BASE_URL`` is documented and supported as a chat-completions-
    style ``…/v1`` URL (see ``hermes_cli/doctor.py:945`` which probes
    ``https://api.minimax.io/v1/models``). For users with that base_url shape,
    the VLM endpoint must not double-stack into ``/v1/v1/coding_plan/vlm``.
    Only the *terminal* ``/v1`` is stripped — internal ``/v1`` segments stay
    intact (the proxy owns its own prefix)."""

    def test_v1_suffix_is_stripped_before_appending_vlm_path(self):
        """``https://api.minimax.chat/v1`` → single ``/v1`` in the result,
        not duplicated."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://api.minimax.chat/v1"
        )
        assert endpoint == "https://api.minimax.chat/v1/coding_plan/vlm"
        # Belt-and-braces: only one occurrence of ``/v1/`` in the path.
        assert endpoint.count("/v1/") == 1

    def test_v1_suffix_with_trailing_slash_is_stripped(self):
        """Trailing slash on a ``/v1`` URL must not block the strip."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://api.minimax.chat/v1/"
        )
        assert endpoint == "https://api.minimax.chat/v1/coding_plan/vlm"
        assert endpoint.count("/v1/") == 1

    def test_v1_in_middle_of_path_is_not_stripped(self):
        """Internal ``/v1`` segments belong to the proxy's own routing prefix
        and must be preserved — only the terminal ``/v1`` (if any) is stripped."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://proxy.com/api/v1/minimax"
        )
        # Terminal ``/minimax`` is preserved, internal ``/api/v1/`` stays put,
        # and the canonical ``/v1/coding_plan/vlm`` is appended as a sibling.
        assert endpoint == "https://proxy.com/api/v1/minimax/v1/coding_plan/vlm"

    def test_anthropic_suffix_still_stripped(self):
        """Regression: the original ``/anthropic`` strip behaviour from #15715
        must continue to work after the ``/v1`` strip is added."""
        endpoint = _derive_minimax_vlm_endpoint(
            "minimax", "https://api.minimax.io/anthropic"
        )
        assert endpoint == "https://api.minimax.io/v1/coding_plan/vlm"


# ---------------------------------------------------------------------------
# Explicit-branch provider gate (P2 follow-up: don't inherit unrelated
# main_base_url when explicit vision provider differs from main provider)
# ---------------------------------------------------------------------------


class TestExplicitMinimaxProviderGate:
    """When ``auxiliary.vision.provider: minimax`` is set explicitly, the
    main-model ``base_url`` must only be plumbed through if the main provider
    IS ALSO MiniMax. Otherwise (e.g. main = openrouter), the explicit branch
    would resolve the VLM endpoint from an unrelated host like
    ``https://openrouter.ai/api/v1/v1/coding_plan/vlm`` — broken.

    Mirrors the provider-match gate in ``hermes_cli/runtime_provider.py`` for
    text traffic where ``model.base_url`` is only honoured when
    ``cfg_provider == provider``.
    """

    def test_explicit_minimax_with_non_minimax_main_provider_ignores_main_base_url(self):
        """Main = openrouter, explicit vision = minimax. The OpenRouter
        ``model.base_url`` MUST NOT leak into the VLM endpoint."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax", "MiniMax-M2.7", None, "mm-key", None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openrouter",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://openrouter.ai/api/v1",
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            provider, client, _model = resolve_vision_provider_client(
                provider="minimax", model="MiniMax-M2.7"
            )
            assert provider == "minimax"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)
            # Default — NOT the OpenRouter URL.
            assert client.base_url == _MINIMAX_VLM_BASE_URLS["minimax"]
            assert "openrouter" not in client.base_url

    def test_explicit_minimax_with_minimax_main_provider_uses_main_base_url(self):
        """Main = minimax with a proxy ``model.base_url``, explicit vision =
        minimax. The proxy must flow through to the VLM endpoint (same proxy
        as text traffic)."""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax", "MiniMax-M2.7", None, "mm-key", None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://my-proxy.com/anthropic",
        ), patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "minimax",
                "api_key": "mm-key",
                "base_url": "",
                "source": "env",
            },
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(False, None),
        ):
            provider, client, _model = resolve_vision_provider_client(
                provider="minimax", model="MiniMax-M2.7"
            )
            assert provider == "minimax"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)
            assert client.base_url == "https://my-proxy.com/v1/coding_plan/vlm"

    def test_auto_minimax_with_minimax_main_provider_uses_main_base_url(self):
        """Auto path with main provider = minimax: by construction the
        provider matches, so ``model.base_url`` is honoured. (Regression
        guard — the auto-path gate is structural, not added in this fix.)"""
        with patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "",
             "MINIMAX_BASE_URL": ""},
            clear=False,
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="minimax",
        ), patch(
            "agent.auxiliary_client._read_main_model",
            return_value="MiniMax-M2.7",
        ), patch(
            "agent.auxiliary_client._read_main_base_url",
            return_value="https://my-proxy.com/anthropic",
        ):
            provider, client, _model = resolve_vision_provider_client()
            assert provider == "minimax"
            assert isinstance(client, MinimaxVlmAuxiliaryClient)
            assert client.base_url == "https://my-proxy.com/v1/coding_plan/vlm"


# ---------------------------------------------------------------------------
# Credential-pool branch: pool entry's base_url must flow into VLM endpoint
# (P2 follow-up to #15715 — round 4)
# ---------------------------------------------------------------------------


def _pool_entry(api_key: str, base_url: str = "") -> SimpleNamespace:
    """Build a fake PooledCredential exposing the two attributes the
    auxiliary client reads via ``_pool_runtime_api_key`` /
    ``_pool_runtime_base_url`` — namely ``runtime_api_key`` and
    ``runtime_base_url`` (see ``agent/credential_pool.py:160-169``)."""
    return SimpleNamespace(
        runtime_api_key=api_key,
        runtime_base_url=base_url or None,
        access_token=api_key,
        base_url=base_url or None,
    )


class TestMinimaxVlmCredentialPoolBaseUrl:
    """When a MiniMax credential-pool entry is selected, its configured
    ``base_url`` (e.g. a proxy the user pointed the credential at) must flow
    into the VLM endpoint — same behaviour as the text-call path which uses
    ``_pool_runtime_base_url(entry, pconfig.inference_base_url)`` (see
    ``agent/auxiliary_client.py:1099`` and ``hermes_cli/runtime_provider.py:196``).

    Without this, vision falls back to the hardcoded provider default while
    text correctly hits the user's proxy — a silent mis-routing identical in
    spirit to the original #15715 bug.
    """

    def test_pool_entry_base_url_overrides_provider_default(self):
        """Pool entry has a custom proxy ``base_url``; no env var and no main
        ``model.base_url``. The VLM endpoint must derive from the pool URL."""
        entry = _pool_entry(
            "pool-key",
            "https://pool-proxy.example.com/minimax/anthropic",
        )
        with patch.dict(
            "os.environ", {"MINIMAX_BASE_URL": ""}, clear=False,
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.api_key == "pool-key"
            assert client.base_url == (
                "https://pool-proxy.example.com/minimax/v1/coding_plan/vlm"
            )
            assert "api.minimax.io" not in client.base_url

    def test_pool_entry_base_url_overridden_by_main_model_when_pool_returns_default(
        self,
    ):
        """If the pool entry's ``base_url`` is the provider default (e.g. the
        credential was created without a custom URL), ``model.base_url`` from
        config.yaml takes over — same precedence as the text-path pool branch
        in ``runtime_provider.py:244-248``."""
        entry = _pool_entry("pool-key", "https://api.minimax.io/anthropic")
        with patch.dict(
            "os.environ", {"MINIMAX_BASE_URL": ""}, clear=False,
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            client = _resolve_minimax_vlm_client(
                "minimax",
                main_model_base_url="https://main-proxy.example.com/anthropic",
            )
            assert client is not None
            # Pool URL is the default → main_model_base_url takes over.
            assert client.base_url == (
                "https://main-proxy.example.com/v1/coding_plan/vlm"
            )

    def test_pool_entry_base_url_used_when_no_main_model_url(self):
        """Pool entry has a custom URL, no main_model_base_url — pool URL
        must be used (not the provider default)."""
        entry = _pool_entry(
            "pool-key", "https://only-pool.example.com/anthropic",
        )
        with patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            # Explicitly pass main_model_base_url=None to mirror the auto path
            # when the user has no model.base_url configured.
            client = _resolve_minimax_vlm_client(
                "minimax", main_model_base_url=None,
            )
            assert client is not None
            assert client.base_url == (
                "https://only-pool.example.com/v1/coding_plan/vlm"
            )

    def test_pool_entry_base_url_with_v1_suffix_strips_correctly(self):
        """Pool entry URL ending in ``/v1`` must combine with the round-3
        terminal ``/v1`` strip so we don't end up with ``/v1/v1/coding_plan/vlm``."""
        entry = _pool_entry(
            "pool-key", "https://pool-proxy.example.com/v1",
        )
        with patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.base_url == (
                "https://pool-proxy.example.com/v1/coding_plan/vlm"
            )
            assert client.base_url.count("/v1/") == 1

    def test_pool_entry_base_url_with_anthropic_suffix_strips_correctly(self):
        """Pool entry URL ending in ``/anthropic`` must have the suffix
        stripped before the VLM path is appended (round-2 strip behaviour)."""
        entry = _pool_entry(
            "pool-key", "https://pool-proxy.example.com/mm/anthropic",
        )
        with patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.base_url == (
                "https://pool-proxy.example.com/mm/v1/coding_plan/vlm"
            )
            assert "/anthropic/" not in client.base_url

    def test_pool_entry_with_no_base_url_falls_back_to_provider_default(self):
        """Defensive: pool entry has no ``base_url`` set. The VLM endpoint
        falls back to the hardcoded provider default — no crash, no empty URL."""
        entry = _pool_entry("pool-key", "")
        with patch.dict(
            "os.environ", {"MINIMAX_BASE_URL": ""}, clear=False,
        ), patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ):
            client = _resolve_minimax_vlm_client("minimax")
            assert client is not None
            assert client.base_url == _MINIMAX_VLM_BASE_URLS["minimax"]
