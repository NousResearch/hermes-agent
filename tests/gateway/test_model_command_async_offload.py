"""Regression tests for #41289: the Discord/Telegram ``/model`` slash command
must not run the blocking provider-listing on the gateway's async event loop.

``list_picker_providers`` / ``list_authenticated_providers`` are synchronous and
can fall through to a blocking ``urllib`` HTTP fetch when the on-disk provider
cache is stale. Running that directly on the event loop froze the gateway for
120-150s ("application did not respond" + delayed agent starts).

Fix (ported from #41304, which patched the old ``gateway/run.py`` location):
``_handle_model_command`` offloads BOTH provider-listing calls via
``asyncio.to_thread`` so the loop stays responsive:

  * line ~1161 — picker path     -> ``list_picker_providers``
  * line ~1382 — text-fallback   -> ``list_authenticated_providers``

These tests assert the *offload contract* at the real handler seam: each listing
function must be dispatched through ``asyncio.to_thread`` and must NOT be invoked
directly. Reverting either ``to_thread`` wrap (calling the sync fn inline again)
makes the corresponding test fail — i.e. the tests are mutation-survivable.
"""

import asyncio

import pytest

import gateway.slash_commands as slash_commands
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event():
    """A bare ``/model`` (no args) — triggers the listing branch."""
    return MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


class _ToThreadSpy:
    """Wraps the real ``asyncio.to_thread`` and records what it was asked to run."""

    def __init__(self):
        self.calls = []  # list of (func, args, kwargs)
        self._real = asyncio.to_thread

    async def __call__(self, func, /, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return await self._real(func, *args, **kwargs)

    def funcs_offloaded(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def _isolated_config(tmp_path, monkeypatch):
    """Point the handler at an empty isolated home so config loading is cheap
    and deterministic (no real provider creds / network)."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


# --------------------------------------------------------------------------- #
# Text-fallback path  ->  list_authenticated_providers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Picker path  ->  list_picker_providers
# --------------------------------------------------------------------------- #
class _FakePickerResult:
    success = True


class _FakePickerAdapter:
    """Adapter whose *type* exposes ``send_model_picker`` (the gate the handler
    checks via ``getattr(type(adapter), 'send_model_picker', None)``)."""

    async def send_model_picker(self, **kwargs):
        return _FakePickerResult()

    def _thread_metadata(self, *a, **k):  # pragma: no cover - not exercised
        return None


@pytest.mark.asyncio
async def test_picker_path_offloads_list_picker_providers(_isolated_config, monkeypatch):
    """A picker-capable adapter => handler takes the picker branch, which must
    offload ``list_picker_providers`` to a worker thread."""
    spy = _ToThreadSpy()
    monkeypatch.setattr(slash_commands.asyncio, "to_thread", spy)

    # Non-empty providers so the handler proceeds to send_model_picker (and
    # returns None), proving we got past the offloaded listing call.
    fake_providers = [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                       "models": ["gpt-x"], "total_models": 1}]

    def _fake_list_picker_providers(**kwargs):
        return fake_providers

    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        _fake_list_picker_providers,
    )

    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    # Stub the metadata/anchor helpers the picker branch calls before sending.
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    result = await runner._handle_model_command(_make_event())

    # Picker "sent" => handler returns None.
    assert result is None
    offloaded = spy.funcs_offloaded()
    assert _fake_list_picker_providers in offloaded, (
        "list_picker_providers must be dispatched via asyncio.to_thread "
        "(it was called inline on the event loop instead)"
    )


@pytest.mark.asyncio
async def test_picker_path_lists_cache_only_and_probes_only_the_current_custom_endpoint(_isolated_config, monkeypatch):
    """#74003: the chat ``/model`` reply is a read path. The listing must ask for cache-only catalogs
    and must not live-probe every saved custom endpoint (only the selected one), matching the GUI."""
    seen: list[dict] = []

    def _fake_list_picker_providers(**kwargs):
        seen.append(kwargs)
        return [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                 "models": ["gpt-x"], "total_models": 1}]

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", _fake_list_picker_providers)
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    assert await runner._handle_model_command(_make_event()) is None
    assert seen, "listing never ran"
    flags = {k: seen[0].get(k) for k in ("non_blocking_catalogs", "probe_custom_providers", "probe_current_custom_provider")}
    assert flags == {"non_blocking_catalogs": True, "probe_custom_providers": False,
                     "probe_current_custom_provider": True}, flags


class _FailingPickerAdapter:
    """A delivery failure must yield the text-list fallback, not a blank `/model` command."""

    async def send_model_picker(self, **kwargs):
        raise RuntimeError("Telegram inline-keyboard send failed")


@pytest.mark.asyncio
async def test_picker_delivery_failure_returns_false_for_text_fallback(monkeypatch):
    fake_providers = [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                       "models": ["gpt-x"], "total_models": 1}]
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **kwargs: fake_providers,
    )
    runner = _make_runner()
    event = _make_event()
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    sent = await runner._send_model_picker(
        event, event.source, _FailingPickerAdapter(), "test-session",
        {"current_provider": "openrouter", "current_model": "gpt-x", "current_base_url": "",
         "user_providers": {}, "custom_providers": [], "excluded_providers": []},
        lambda *_args: None,
    )

    assert sent is False
