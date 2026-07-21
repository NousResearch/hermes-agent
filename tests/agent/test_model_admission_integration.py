from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import auxiliary_client as aux
from agent import chat_completion_helpers as chat
from hermes_cli.config import DEFAULT_CONFIG


@pytest.fixture(autouse=True)
def _fresh_registry():
    aux._reset_model_admission_registry_for_tests()
    yield
    aux._reset_model_admission_registry_for_tests()


class _RecordingPermit:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.is_finished = False

    def succeed(self) -> None:
        if not self.is_finished:
            self.events.append("succeed")
            self.is_finished = True

    def fail(self, error: BaseException) -> None:
        if not self.is_finished:
            self.events.append(("fail", error))
            self.is_finished = True

    def release(self) -> None:
        if not self.is_finished:
            self.events.append("release")
            self.is_finished = True

    def wrap_stream(self, stream):
        permit = self

        class _Stream:
            def __iter__(self):
                return self

            def __next__(self):
                try:
                    return next(stream)
                except StopIteration:
                    permit.succeed()
                    raise
                except BaseException as error:
                    permit.fail(error)
                    raise

            def close(self):
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                permit.release()

        return _Stream()

    def wrap_async_stream(self, stream):
        permit = self

        class _AsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return await stream.__anext__()
                except StopAsyncIteration:
                    permit.succeed()
                    raise
                except BaseException as error:
                    permit.fail(error)
                    raise

        return _AsyncStream()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _traceback):
        if exc is None:
            self.succeed()
        else:
            self.fail(exc)
        return False


def test_model_admission_is_safe_by_default_and_requires_explicit_enablement():
    config = DEFAULT_CONFIG["model_admission"]

    assert config["enabled"] is False
    assert config["max_in_flight"] > 0
    assert config["per_target"] > 0
    assert config["min_per_target"] <= config["per_target"]


def test_model_admission_config_is_loaded_once_per_process(monkeypatch):
    enabled = dict(DEFAULT_CONFIG["model_admission"])
    enabled.update({"enabled": True, "max_in_flight": 3, "per_target": 2})
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"model_admission": enabled}
    )

    first = aux._get_model_admission_registry()
    assert first.settings.enabled is True
    assert first.settings.max_in_flight == 3
    assert first.settings.per_target == 2

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_admission": {"enabled": False}},
    )
    assert aux._get_model_admission_registry() is first


def test_main_nonstreaming_attempt_reports_success(monkeypatch):
    permit = _RecordingPermit()
    monkeypatch.setattr(chat, "_acquire_model_admission", lambda *_args: permit)
    response = SimpleNamespace(id="ok")
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="m",
    )

    assert (
        chat._dispatch_nonstreaming_api_request(
            agent, {"model": "m", "messages": []}, make_client=lambda *_a, **_k: client
        )
        is response
    )
    assert permit.events == ["succeed"]


def test_main_nonstreaming_429_is_reported_before_retry(monkeypatch):
    permit = _RecordingPermit()
    monkeypatch.setattr(chat, "_acquire_model_admission", lambda *_args: permit)

    class RateLimitError(RuntimeError):
        status_code = 429

    error = RateLimitError("busy")
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: (_ for _ in ()).throw(error)
            )
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="m",
    )

    with pytest.raises(RateLimitError):
        chat._dispatch_nonstreaming_api_request(
            agent,
            {"model": "m", "messages": []},
            make_client=lambda *_a, **_k: client,
        )

    assert permit.events == [("fail", error)]


def test_main_anthropic_permit_covers_manager_exit(monkeypatch):
    events: list[str] = []
    permit = _RecordingPermit()
    monkeypatch.setattr(chat, "_acquire_model_admission", lambda *_args: permit)

    class StreamManager:
        def __enter__(self):
            events.append("stream-enter")
            return iter(["event"])

        def __exit__(self, _exc_type, _exc, _traceback):
            events.append("stream-exit")
            assert permit.events == []

    client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: StreamManager())
    )

    with chat._admitted_anthropic_stream(
        SimpleNamespace(), {"model": "m"}, client
    ) as stream:
        assert list(stream) == ["event"]
        assert permit.events == []
        events.append("body")

    assert events == ["stream-enter", "body", "stream-exit"]
    assert permit.events == ["succeed"]


def test_main_anthropic_close_429_is_reported(monkeypatch):
    permit = _RecordingPermit()
    monkeypatch.setattr(chat, "_acquire_model_admission", lambda *_args: permit)

    class RateLimitError(RuntimeError):
        status_code = 429

    error = RateLimitError("close rate limited")

    class StreamManager:
        def __enter__(self):
            return iter(())

        def __exit__(self, _exc_type, _exc, _traceback):
            raise error

    client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: StreamManager())
    )

    with pytest.raises(RateLimitError):
        with chat._admitted_anthropic_stream(SimpleNamespace(), {"model": "m"}, client):
            pass

    assert permit.events == [("fail", error)]


def test_auxiliary_sync_stream_holds_permit_until_exhaustion(monkeypatch):
    permit = _RecordingPermit()
    acquire = MagicMock(return_value=permit)
    monkeypatch.setattr(aux, "acquire_model_admission", acquire)
    raw_stream = iter(["a", "b"])
    client = SimpleNamespace(
        base_url="https://user:secret@example.test/v1?api_key=hidden",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: raw_stream)
        ),
    )
    wrapped = aux._wrap_auxiliary_client(
        client,
        provider="openrouter",
        model="m",
        base_url=None,
        async_mode=False,
    )

    stream = wrapped.chat.completions.create(model="m", stream=True)
    assert permit.events == []
    assert next(stream) == "a"
    assert permit.events == []
    assert list(stream) == ["b"]
    assert permit.events == ["succeed"]
    acquire.assert_called_once_with(
        "openrouter",
        "https://user:secret@example.test/v1?api_key=hidden",
        "m",
    )


def test_auxiliary_async_attempt_has_its_own_permit(monkeypatch):
    first_permit = _RecordingPermit()
    second_permit = _RecordingPermit()
    permits = [first_permit, second_permit]

    async def acquire(*_args):
        return permits.pop(0)

    monkeypatch.setattr(aux, "acquire_model_admission_async", acquire)

    class RateLimitError(RuntimeError):
        status_code = 429

    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("busy")
        return SimpleNamespace(id="ok")

    client = SimpleNamespace(
        base_url="https://api.example.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    wrapped = aux._wrap_auxiliary_client(
        client,
        provider="custom",
        model="m",
        base_url=None,
        async_mode=True,
    )

    async def exercise():
        with pytest.raises(RateLimitError):
            await wrapped.chat.completions.create(model="m")
        assert first_permit.events
        assert first_permit.events[0][0] == "fail"
        return await wrapped.chat.completions.create(model="m")

    response = asyncio.run(exercise())

    assert response.id == "ok"
    assert second_permit.events == ["succeed"]
    # One failed permit was fully released before the second attempt entered.
    # ``permits`` is empty only if acquisition happened independently twice.
    assert permits == []


def test_virtual_moa_outer_request_does_not_enter_registry(monkeypatch):
    registry = MagicMock()
    monkeypatch.setattr(aux, "_get_model_admission_registry", lambda: registry)

    permit = aux.acquire_model_admission("moa", "in-process://moa", "preset")

    assert permit.is_noop
    registry.acquire.assert_not_called()
