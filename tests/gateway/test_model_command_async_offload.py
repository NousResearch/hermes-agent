"""Gateway ``/model`` picker listing is a read path (#41289, #74003): it must ask for
cache-only catalogs and live-probe only the currently selected custom endpoint, so a
stale provider cache cannot freeze the gateway on blocking HTTP requests.

``/model --refresh`` is the one exception — the user asked for live data, so the saved
custom endpoints are probed like the TUI refresh path does.
"""

import threading

import pytest

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


def _make_event(text="/model"):
    """A bare ``/model`` (no args) — triggers the listing branch."""
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


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
async def test_picker_path_runs_provider_listing_off_the_event_loop(_isolated_config, monkeypatch):
    """#41289/#41304: ``list_picker_providers`` can fall through to a blocking HTTP fetch, so the
    picker branch must run it on a worker thread — never on the gateway's event-loop thread."""
    listing_threads: list[int] = []

    def _fake_list_picker_providers(**kwargs):
        listing_threads.append(threading.get_ident())
        return [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                 "models": ["gpt-x"], "total_models": 1}]

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", _fake_list_picker_providers)
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    # Picker "sent" => handler returns None, proving it got past the listing call.
    assert await runner._handle_model_command(_make_event()) is None
    assert listing_threads, "listing never ran"
    assert threading.get_ident() not in listing_threads, (
        "list_picker_providers ran inline on the event-loop thread"
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


@pytest.mark.asyncio
async def test_refresh_picker_path_probes_the_saved_custom_endpoints(_isolated_config, monkeypatch):
    """``/model --refresh`` is an explicit request for live data, so it opts into the probes a plain
    open withholds (#74003). Busting the disk cache *without* probing made the flag destructive: an
    unprobed custom row falls back to its declared ``models:`` subset, so the user lost the very
    list they asked to refresh."""
    seen: list[dict] = []
    cleared: list[bool] = []

    def _fake_list_picker_providers(**kwargs):
        seen.append(kwargs)
        return [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                 "models": ["gpt-x"], "total_models": 1}]

    import hermes_cli.models as models_module

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", _fake_list_picker_providers)
    monkeypatch.setattr(models_module, "clear_provider_models_cache",
                        lambda *a, **k: cleared.append(True))
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    assert await runner._handle_model_command(_make_event("/model --refresh")) is None
    assert seen, "listing never ran"
    assert seen[0].get("probe_custom_providers") is True, seen[0]
    assert cleared, "--refresh still busts the disk cache"
