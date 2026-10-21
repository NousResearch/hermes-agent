"""Resolver-backed regression coverage for same-provider vision retries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _AwaitableResponse:
    """A response-shaped value that works for both sync and async fake SDK calls."""

    choices = [SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
    usage = None

    def __await__(self):
        async def _return_self():
            return self

        return _return_self().__await__()


class _FakeOpenAIClient:
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, api_key: str, base_url: str, wire_calls: list[tuple[str, str]]):
        self.api_key = api_key
        self.base_url = base_url
        self._wire_calls = wire_calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self._wire_calls.append(("chat_completions", self.base_url))
        return _AwaitableResponse()


class _AuthError(Exception):
    status_code = 401


class _PoolRetryClient:
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, api_key: str, base_url: str, calls: list[str], async_mode: bool):
        self.api_key = api_key
        self.base_url = base_url
        self._calls = calls
        self._async_mode = async_mode
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self._calls.append(self.api_key)
        if self.api_key == "pool-first":
            if self._async_mode:
                return _AwaitableError(_AuthError("expired"))
            raise _AuthError("expired")
        return _AwaitableResponse()


class _AwaitableError:
    def __init__(self, error: Exception):
        self._error = error

    def __await__(self):
        async def _raise():
            raise self._error

        return _raise().__await__()


class _AuthRecoveryClient:
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, api_key: str, base_url: str, calls: list[tuple[str, str]],
                 async_mode: bool, fail_first: bool):
        self.api_key = api_key
        self.base_url = base_url
        self._calls = calls
        self._async_mode = async_mode
        self._fail_first = fail_first
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self._calls.append((self.api_key, self.base_url))
        if self._fail_first:
            error = _AuthError("expired")
            if self._async_mode:
                return _AwaitableError(error)
            raise error
        return _AwaitableResponse()


class _CallLevelRetryClient:
    """OpenAI-wire fake that records the complete initial/retry request pair."""

    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, api_key: str, base_url: str, calls: list[dict], async_mode: bool,
                 fail_first: bool):
        self.api_key = api_key
        self.base_url = base_url
        self._calls = calls
        self._async_mode = async_mode
        self._fail_first = fail_first
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._calls.append(dict(kwargs, wire="chat_completions", client_type=type(self).__name__,
                                base_url=self.base_url, api_key=self.api_key))
        if self._fail_first:
            error = _AuthError("expired")
            if self._async_mode:
                return _AwaitableError(error)
            raise error
        return _AwaitableResponse()


def _write_custom_vision_config(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: custom\n"
        "    model: call-level-vision\n"
        "    api_mode: chat_completions\n",
        encoding="utf-8",
    )


def _write_zai_pool(hermes_home):
    import json

    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "zai": [
                        {
                            "id": "zai-first",
                            "label": "first",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "pool-first",
                        },
                        {
                            "id": "zai-second",
                            "label": "second",
                            "auth_type": "api_key",
                            "priority": 1,
                            "source": "manual",
                            "access_token": "pool-second",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_zai_vision_auth_recovery_rotates_pool_key_despite_env_key(
    tmp_path, monkeypatch, async_mode
):
    """An auth retry must re-enter ZAI vision resolution and select the rotated pool key."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_zai_pool(hermes_home)
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: zai\n"
        "    model: glm-5.3-flash\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ZAI_API_KEY", "env-stale")

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    calls: list[str] = []
    constructed: list[str] = []

    def build_openai(*, api_key, base_url, **_kwargs):
        constructed.append(api_key)
        return _PoolRetryClient(api_key, base_url, calls, async_mode)

    with (
        patch.object(aux, "_create_openai_client", side_effect=build_openai),
        patch.object(aux, "_validate_llm_response", side_effect=lambda value, _task: value),
    ):
        if async_mode:
            result = asyncio.run(
                aux._async_call_llm_impl(
                    task="vision", messages=[{"role": "user", "content": "describe"}],
                    max_tokens=128, timeout=30,
                )
            )
        else:
            result = aux._call_llm_impl(
                task="vision", messages=[{"role": "user", "content": "describe"}],
                max_tokens=128, timeout=30,
            )

    assert result is not None
    assert calls == ["pool-first", "pool-second"]
    assert constructed == ["pool-first", "pool-second"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_copilot_vision_retry_rebuild_keeps_vision_header(monkeypatch, async_mode):
    """The real auth-recovery rung must rebuild Copilot with its vision-only header."""
    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_test-token")
    constructed: list[dict] = []
    calls: list[tuple[str, str]] = []

    def build_openai(*, api_key, base_url, **kwargs):
        constructed.append({"api_key": api_key, "base_url": base_url, **kwargs})
        return _AuthRecoveryClient(api_key, base_url, calls, async_mode, fail_first=len(constructed) == 1)

    def refresh_copilot(provider, **_kwargs):
        aux._evict_cached_clients(provider)
        return True

    with (
        patch.object(aux, "_create_openai_client", side_effect=build_openai),
        patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value=("copilot-api", None)),
        patch.object(aux, "_refresh_provider_credentials", side_effect=refresh_copilot),
    ):
        if async_mode:
            result = asyncio.run(
                aux._async_call_llm_impl(
                    task="vision", provider="copilot", model="gpt-5-mini",
                    messages=[{"role": "user", "content": "describe"}],
                    max_tokens=128, timeout=30,
                )
            )
        else:
            result = aux._call_llm_impl(
                task="vision", provider="copilot", model="gpt-5-mini",
                messages=[{"role": "user", "content": "describe"}],
                max_tokens=128, timeout=30,
            )

    assert result is not None
    assert len(constructed) == 2
    assert calls == [("copilot-api", constructed[0]["base_url"]),
                     ("copilot-api", constructed[1]["base_url"])]
    assert all(item["default_headers"]["Copilot-Vision-Request"] == "true" for item in constructed)


@pytest.mark.parametrize("async_mode", [False, True])
def test_auto_custom_runtime_vision_auth_recovery_preserves_route_pairing(
    tmp_path, monkeypatch, async_mode
):
    """Automatic custom vision recovery must retain the live runtime endpoint and key together."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    runtime_base_url = "https://runtime-vision.example/v1"
    runtime_api_key = "runtime-secret"
    constructed: list[tuple[str, str]] = []
    calls: list[tuple[str, str]] = []

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()

    def build_openai(*, api_key, base_url, **_kwargs):
        constructed.append((api_key, base_url))
        return _AuthRecoveryClient(
            api_key, base_url, calls, async_mode, fail_first=len(constructed) == 1
        )

    def refresh_custom(provider, **_kwargs):
        aux._evict_cached_clients(provider)
        return True

    with (
        patch.object(aux, "_create_openai_client", side_effect=build_openai),
        patch.object(aux, "_refresh_provider_credentials", side_effect=refresh_custom),
    ):
        runtime = {
            "provider": "custom", "model": "qwen-vision", "base_url": runtime_base_url,
            "api_key": runtime_api_key, "api_mode": "chat_completions",
        }
        if async_mode:
            result = asyncio.run(
                aux._async_call_llm_impl(
                    task="vision", provider="auto", model="qwen-vision", main_runtime=runtime,
                    messages=[{"role": "user", "content": "describe"}],
                    max_tokens=128, timeout=30,
                )
            )
        else:
            result = aux._call_llm_impl(
                task="vision", provider="auto", model="qwen-vision", main_runtime=runtime,
                messages=[{"role": "user", "content": "describe"}],
                max_tokens=128, timeout=30,
            )

    assert result is not None
    assert constructed == [(runtime_api_key, runtime_base_url)] * 2
    assert calls == constructed


@pytest.mark.parametrize("async_mode", [False, True])
def test_zai_vision_retry_rebuilds_the_selected_chat_transport(tmp_path, monkeypatch, async_mode):
    """A stale task mode must not turn ZAI's OpenAI vision route into Messages on retry."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: zai\n"
        "    model: glm-5.3-flash\n"
        "    api_key: zai-secret\n"
        "    api_mode: anthropic_messages\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    selected_base_url = aux._ZAI_OPENAI_VISION_URLS[0]
    wire_calls: list[tuple[str, str]] = []
    constructed: list[tuple[str, str]] = []
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn", usage=None
    )

    def build_openai(*, api_key, base_url, **_kwargs):
        constructed.append((api_key, base_url))
        return _FakeOpenAIClient(api_key, base_url, wire_calls)

    def build_anthropic(*_args, **_kwargs):
        return SimpleNamespace(base_url=selected_base_url)

    def create_anthropic(_client, _kwargs, **_options):
        wire_calls.append(("anthropic_messages", selected_base_url))
        return response

    with (
        patch.object(aux, "_create_openai_client", side_effect=build_openai),
        patch("agent.anthropic_adapter.build_anthropic_client", side_effect=build_anthropic),
        patch("agent.anthropic_adapter.create_anthropic_message", side_effect=create_anthropic),
        patch.object(aux, "_validate_llm_response", side_effect=lambda value, _task: value),
    ):
        request = aux._prepare_aux_request(
            "vision", provider=None, model=None, base_url=None, api_key=None, main_runtime={},
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, timeout=30.0, extra_body=None, reasoning_config=None,
            extra_headers=None, api_mode=None, route_info=None, async_mode=async_mode,
        )

        assert request.resolved_provider == "zai"
        assert request.resolved_api_mode == "chat_completions"
        assert request.client.base_url == selected_base_url
        aux._evict_cached_clients("zai")

        retry_args = dict(
            task="vision", resolved_provider=request.resolved_provider,
            resolved_api_mode=request.resolved_api_mode, resolved_model=request.resolved_model,
            resolved_base_url=request.resolved_base_url, resolved_api_key=request.resolved_api_key,
            main_runtime={}, final_model=request.final_model,
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, effective_timeout=request.effective_timeout,
            effective_extra_body=request.effective_extra_body, reasoning_config=None,
        )

        if async_mode:
            async def run_attempts():
                await aux._relay_async_completion(
                    request.client, request.kwargs, provider=request.request_provider,
                    api_mode=request.resolved_api_mode,
                )
                return await aux._retry_same_provider_async(**retry_args)

            asyncio.run(run_attempts())
        else:
            aux._relay_sync_completion(
                request.client, request.kwargs, provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )
            aux._retry_same_provider_sync(**retry_args)

    assert constructed == [
        ("zai-secret", selected_base_url),
        ("zai-secret", selected_base_url),
    ]
    assert wire_calls == [
        ("chat_completions", selected_base_url),
        ("chat_completions", selected_base_url),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_call_level_vision_api_mode_reaches_initial_and_retry_resolution(
    tmp_path, monkeypatch, async_mode
):
    """A call-level mode override must beat task mode at both resolver boundaries."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    selected_base_url = "https://vision-gateway.example/v1"
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: custom\n"
        "    model: qwen-vision\n"
        f"    base_url: {selected_base_url}\n"
        "    api_key: custom-secret\n"
        "    api_mode: anthropic_messages\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    wire_calls: list[tuple[str, str]] = []

    def build_openai(*, api_key, base_url, **_kwargs):
        return _FakeOpenAIClient(api_key, base_url, wire_calls)

    with patch.object(aux, "_create_openai_client", side_effect=build_openai):
        request = aux._prepare_aux_request(
            "vision", provider=None, model=None, base_url=None, api_key=None, main_runtime={},
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, timeout=30.0, extra_body=None, reasoning_config=None,
            extra_headers=None, api_mode="chat_completions", route_info=None, async_mode=async_mode,
        )
        assert request.resolved_api_mode == "chat_completions"
        assert request.resolved_base_url_explicit is True
        retry_client, retry_kwargs = aux._prepare_same_provider_retry(
            task="vision", resolved_provider=request.resolved_provider,
            resolved_model=request.resolved_model, resolved_base_url=request.resolved_base_url,
            resolved_base_url_explicit=request.resolved_base_url_explicit,
            resolved_api_key=request.resolved_api_key, resolved_api_mode=request.resolved_api_mode,
            main_runtime={}, final_model=request.final_model,
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, effective_timeout=request.effective_timeout,
            effective_extra_body=request.effective_extra_body, reasoning_config=None,
            async_mode=async_mode,
        )
        if async_mode:
            async def run_attempts():
                await aux._relay_async_completion(
                    request.client, request.kwargs, provider=request.request_provider,
                    api_mode=request.resolved_api_mode,
                )
                await aux._relay_async_completion(
                    retry_client, retry_kwargs, provider=request.request_provider,
                    api_mode=request.resolved_api_mode,
                )

            asyncio.run(run_attempts())
        else:
            aux._relay_sync_completion(
                request.client, request.kwargs, provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )
            aux._relay_sync_completion(
                retry_client, retry_kwargs, provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )

    assert wire_calls == [
        ("chat_completions", selected_base_url),
        ("chat_completions", selected_base_url),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_explicit_custom_vision_messages_mode_matches_the_selected_client(
    tmp_path, monkeypatch, async_mode
):
    """An explicit custom Messages route keeps its task base URL and mode on retry."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    selected_base_url = "https://vision-gateway.example/v1"
    (hermes_home / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: vision-gateway\n"
        f"    base_url: {selected_base_url}\n"
        "    api_key: custom-secret\n"
        "auxiliary:\n"
        "  vision:\n"
        "    provider: vision-gateway\n"
        "    model: qwen-vision\n"
        f"    base_url: {selected_base_url}\n"
        "    api_mode: anthropic_messages\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    wire_calls: list[tuple[str, str]] = []
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn", usage=None
    )

    def build_anthropic(_api_key, base_url):
        assert base_url == selected_base_url
        return SimpleNamespace(base_url=base_url)

    def create_anthropic(_client, _kwargs, **_options):
        wire_calls.append(("anthropic_messages", selected_base_url))
        return response

    with (
        patch("agent.anthropic_adapter.build_anthropic_client", side_effect=build_anthropic),
        patch("agent.anthropic_adapter.create_anthropic_message", side_effect=create_anthropic),
    ):
        request = aux._prepare_aux_request(
            "vision", provider=None, model=None, base_url=None, api_key=None, main_runtime={},
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, timeout=30.0, extra_body=None, reasoning_config=None,
            extra_headers=None, api_mode=None, route_info=None, async_mode=async_mode,
        )
        assert request.resolved_api_mode == "anthropic_messages"
        assert request.client.base_url == selected_base_url
        assert request.resolved_base_url_explicit is True
        retry_client, retry_kwargs = aux._prepare_same_provider_retry(
            task="vision", resolved_provider=request.resolved_provider,
            resolved_model=request.resolved_model, resolved_base_url=request.resolved_base_url,
            resolved_base_url_explicit=request.resolved_base_url_explicit,
            resolved_api_key=request.resolved_api_key, resolved_api_mode=request.resolved_api_mode,
            main_runtime={}, final_model=request.final_model,
            messages=[{"role": "user", "content": "describe this image"}], temperature=None,
            max_tokens=128, tools=None, effective_timeout=request.effective_timeout,
            effective_extra_body=request.effective_extra_body, reasoning_config=None,
            async_mode=async_mode,
        )
        if async_mode:
            async def run_attempts():
                await aux._relay_async_completion(
                    request.client, request.kwargs, provider=request.request_provider,
                    api_mode=request.resolved_api_mode,
                )
                await aux._relay_async_completion(
                    retry_client, retry_kwargs, provider=request.request_provider,
                    api_mode=request.resolved_api_mode,
                )

            asyncio.run(run_attempts())
        else:
            aux._relay_sync_completion(
                request.client, request.kwargs, provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )
            aux._relay_sync_completion(
                retry_client, retry_kwargs, provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )

    assert wire_calls == [
        ("anthropic_messages", selected_base_url),
        ("anthropic_messages", selected_base_url),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_call_level_base_url_without_config_url_survives_auth_retry(
    tmp_path, monkeypatch, async_mode
):
    """The normal recovery ladder must retain a call-level custom endpoint and key."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_custom_vision_config(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    selected_base_url = "https://call-level-vision.example/v1"
    selected_api_key = "call-level-key"
    constructed: list[dict] = []
    wire_calls: list[dict] = []

    def build_openai(*, api_key, base_url, **kwargs):
        constructed.append({"api_key": api_key, "base_url": base_url, **kwargs})
        return _CallLevelRetryClient(
            api_key, base_url, wire_calls, async_mode, fail_first=len(constructed) == 1
        )

    def refresh_custom(provider, **_kwargs):
        assert provider == "custom"
        aux._evict_cached_clients(provider)
        return True

    common = dict(
        task="vision", provider="custom", model="call-level-vision", base_url=selected_base_url,
        api_key=selected_api_key, messages=[{"role": "user", "content": "describe"}],
        max_tokens=64, timeout=30,
    )
    if not async_mode:
        common.update(api_mode="chat_completions", extra_headers={"x-test-route": "keep"})

    with (
        patch.object(aux, "_create_openai_client", side_effect=build_openai),
        patch.object(aux, "_refresh_provider_credentials", side_effect=refresh_custom),
        patch.object(aux, "_validate_llm_response", side_effect=lambda value, _task: value),
    ):
        result = (
            asyncio.run(aux._async_call_llm_impl(**common))
            if async_mode else aux._call_llm_impl(**common)
        )

    assert result is not None
    assert len(constructed) == 2
    assert [(item["api_key"], item["base_url"]) for item in constructed] == [
        (selected_api_key, selected_base_url),
        (selected_api_key, selected_base_url),
    ]
    assert len(wire_calls) == 2
    assert [call["wire"] for call in wire_calls] == ["chat_completions", "chat_completions"]
    assert all(call["client_type"] == "_CallLevelRetryClient" for call in wire_calls)
    assert [(call["api_key"], call["base_url"]) for call in wire_calls] == [
        (selected_api_key, selected_base_url),
        (selected_api_key, selected_base_url),
    ]
    if not async_mode:
        assert [call["extra_headers"] for call in wire_calls] == [
            {"x-test-route": "keep"}, {"x-test-route": "keep"}
        ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_call_level_base_url_retry_negative_control_drops_endpoint(
    tmp_path, monkeypatch, async_mode
):
    """The same real resolver must reject a retry when explicit URL provenance is removed."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_custom_vision_config(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.auxiliary_client as aux

    aux.shutdown_cached_clients()
    selected_base_url = "https://call-level-vision.example/v1"
    constructed: list[tuple[str, str]] = []

    def build_openai(*, api_key, base_url, **_kwargs):
        constructed.append((api_key, base_url))
        return _CallLevelRetryClient(api_key, base_url, [], async_mode, fail_first=False)

    with patch.object(aux, "_create_openai_client", side_effect=build_openai):
        request = aux._prepare_aux_request(
            "vision", provider="custom", model="call-level-vision", base_url=selected_base_url,
            api_key="call-level-key", main_runtime={},
            messages=[{"role": "user", "content": "describe"}], temperature=None,
            max_tokens=64, tools=None, timeout=30, extra_body=None, reasoning_config=None,
            extra_headers=None, api_mode="chat_completions", route_info=None,
            async_mode=async_mode,
        )
        assert request.resolved_base_url_explicit is True
        with pytest.raises(RuntimeError, match="could not be rebuilt"):
            aux._prepare_same_provider_retry(
                task="vision", resolved_provider=request.resolved_provider,
                resolved_model=request.resolved_model, resolved_base_url=request.resolved_base_url,
                resolved_base_url_explicit=False, resolved_api_key=request.resolved_api_key,
                resolved_api_mode=request.resolved_api_mode, main_runtime={},
                final_model=request.final_model, messages=[{"role": "user", "content": "describe"}],
                temperature=None, max_tokens=64, tools=None,
                effective_timeout=request.effective_timeout,
                effective_extra_body=request.effective_extra_body, reasoning_config=None,
                async_mode=async_mode,
            )

    assert constructed == [("call-level-key", selected_base_url)]
