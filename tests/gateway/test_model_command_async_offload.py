"""Gateway ``/model`` picker listing is a read path (#41289, #74003): it must ask for
cache-only catalogs and live-probe only the currently selected custom endpoint, so a
stale provider cache cannot freeze the gateway on blocking HTTP fetches.
"""


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


def _make_event():
    """A bare ``/model`` (no args) — triggers the listing branch."""
    return MessageEvent(
        text="/model",
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
async def test_picker_path_requests_explicit_providers_only(_isolated_config, monkeypatch):
    """Chat pickers share the Desktop rule (#56974): ambient credentials (gh CLI -> Copilot) must not
    be offered as connected providers, so the gateway asks for the explicit-only listing."""
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
    assert seen and seen[0].get("explicit_only") is True, seen


@pytest.mark.asyncio
async def test_text_listing_hides_ambient_providers(_isolated_config, monkeypatch):
    """The text fallback (no picker support) applies the same explicit-provider filter."""
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "is_current": True, "is_user_defined": False,
         "models": ["gpt-x"], "total_models": 1, "source": "built-in"},
        {"slug": "copilot", "name": "GitHub Copilot", "is_current": False, "is_user_defined": False,
         "models": ["gpt-5.4"], "total_models": 1, "source": "built-in"},
    ]
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [dict(r) for r in rows])
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda slug: False)
    monkeypatch.setattr("hermes_cli.inventory._external_process_signed_in", lambda slug: False)

    reply = await _make_runner()._handle_model_command(_make_event())
    assert "OpenRouter" in reply
    assert "Copilot" not in reply
