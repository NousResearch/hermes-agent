"""Gateway ``/model`` picker listing is a read path (#41289, #74003): it must ask for
cache-only catalogs and live-probe only the currently selected custom endpoint, so a
stale provider cache cannot freeze the gateway on blocking HTTP fetches.
"""

import asyncio
import json
from pathlib import Path
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

    # _real_discovery_home sets Path.home() to tmp_path; auth's pytest guard rejects ~/.hermes.
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
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
        self.providers = kwargs["providers"]
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
    and must not live-probe every saved custom endpoint (only the selected one), matching the GUI.
    It also requests Desktop's explicit-provider filter (#56974)."""
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
    flags = {k: seen[0].get(k) for k in (
        "non_blocking_catalogs", "probe_custom_providers", "probe_current_custom_provider", "explicit_only")}
    assert flags == {"non_blocking_catalogs": True, "probe_custom_providers": False,
                     "probe_current_custom_provider": True, "explicit_only": True}, flags


@pytest.fixture
def _real_discovery_home(_isolated_config, tmp_path, monkeypatch):
    """Real discovery and auth in a private home; disable only remote/background catalog work."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Catalog fingerprints also use os.path.expanduser, including for other CLIs' files.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(_isolated_config))
    for var in ("GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT_ID", "VERTEX_CREDENTIALS_PATH",
                "AWS_PROFILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_WEB_IDENTITY_TOKEN_FILE",
                "LM_BASE_URL", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    (_isolated_config / "config.yaml").write_text(
        "model:\n  provider: deepseek\n  default: deepseek-chat\n", encoding="utf-8")
    (_isolated_config / ".env").write_text("DEEPSEEK_API_KEY=test-deepseek-key\n", encoding="utf-8")
    (_isolated_config / "auth.json").write_text(
        json.dumps({"version": 1, "active_provider": "deepseek", "providers": {}}), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.model_catalog._fetch_manifest", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.models._spawn_swr_refresh", lambda *_a, **_kw: None)
    return _isolated_config


async def _gateway_picker_and_text(monkeypatch):
    """Collect the actual rows delivered to a picker, then the text fallback, through /model."""
    runner = _make_runner()
    adapter = _FakePickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)
    assert await runner._handle_model_command(_make_event()) is None
    rows = {row["slug"]: row for row in adapter.providers}
    assert rows["deepseek"]["models"]
    runner.adapters = {}
    reply = await runner._handle_model_command(_make_event())
    assert "`--provider deepseek`" in reply
    return rows, reply


@pytest.mark.asyncio
async def test_vertex_ambient_credentials_filtered_by_both_gateway_listings(_real_discovery_home, tmp_path, monkeypatch):
    """A real ambient Vertex row has models; only explicit Vertex configuration should retain it."""
    from hermes_cli.model_switch import list_authenticated_providers

    credentials = tmp_path / "google-credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    discovered = await asyncio.to_thread(
        list_authenticated_providers, current_provider="deepseek", non_blocking_catalogs=True)
    vertex = next(row for row in discovered if row["slug"] == "vertex")
    assert vertex["models"]
    assert not vertex["is_current"]

    rows, reply = await _gateway_picker_and_text(monkeypatch)
    assert "vertex" not in rows
    assert "`--provider vertex`" not in reply

    from hermes_cli.config import set_config_value
    await asyncio.to_thread(set_config_value, "vertex.project_id", "test-vertex-project")
    rows, reply = await _gateway_picker_and_text(monkeypatch)
    assert rows["vertex"]["models"]
    assert "`--provider vertex`" in reply


@pytest.mark.asyncio
async def test_qwen_cli_empty_row_is_removed_from_text_listing(_real_discovery_home, tmp_path, monkeypatch):
    """A bare Qwen CLI login adds a text heading, but the unfiltered picker already drops its empty row."""
    from gateway.slash_commands_model import _model_provider_listing_lines
    from hermes_cli.model_switch import list_authenticated_providers
    from hermes_cli.model_switch_providers import list_picker_providers

    qwen_home = tmp_path / ".qwen"
    qwen_home.mkdir()
    (qwen_home / "oauth_creds.json").write_text(
        json.dumps({"access_token": "test-qwen-token"}), encoding="utf-8")
    discovered = await asyncio.to_thread(
        list_authenticated_providers, current_provider="deepseek", non_blocking_catalogs=True)
    qwen = next(row for row in discovered if row["slug"] == "qwen-oauth")
    assert qwen["models"] == []
    assert "`--provider qwen-oauth`" in "\n".join(_model_provider_listing_lines(discovered))
    unfiltered_picker = await asyncio.to_thread(
        list_picker_providers, current_provider="deepseek", non_blocking_catalogs=True)
    assert "qwen-oauth" not in {row["slug"] for row in unfiltered_picker}

    rows, reply = await _gateway_picker_and_text(monkeypatch)
    assert "qwen-oauth" not in rows
    assert "`--provider qwen-oauth`" not in reply


_UNCONFIGURED_PROVIDER_ROWS = [
    {"slug": "openrouter", "name": "OpenRouter", "is_current": True, "is_user_defined": False,
     "models": ["gpt-x"], "total_models": 1, "source": "built-in"},
    {"slug": "deepseek", "name": "DeepSeek", "is_current": False, "is_user_defined": False,
     "models": ["deepseek-test-model"], "total_models": 1, "source": "built-in"},
]


@pytest.mark.asyncio
async def test_text_listing_filters_off_the_event_loop(_isolated_config, monkeypatch):
    """The filter reads config/.env/pool synchronously, so it runs in the listing's worker thread (#41289)."""
    import hermes_cli.model_switch_providers as msp

    loop_thread = threading.get_ident()
    filter_threads: list[int] = []
    real_filter = msp.filter_explicit_picker_rows

    def _spy(providers, current_provider=""):
        filter_threads.append(threading.get_ident())
        return real_filter(providers, current_provider=current_provider)

    monkeypatch.setattr(msp, "filter_explicit_picker_rows", _spy)
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers",
                        lambda **kw: [dict(r) for r in _UNCONFIGURED_PROVIDER_ROWS])
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda slug: False)
    monkeypatch.setattr("hermes_cli.inventory._external_process_signed_in", lambda slug: False)

    reply = await _make_runner()._handle_model_command(_make_event())
    assert "`--provider openrouter`" in reply
    assert "DeepSeek" not in reply
    assert filter_threads and loop_thread not in filter_threads, "filter ran on the event loop"


@pytest.fixture
def _multiplexed_profile(_isolated_config, tmp_path, monkeypatch):
    """A multiplexed gateway routing the chat to a profile with no provider keys in its ``.env``:
    an unscoped secret read raises here, so the listing must bind the routed profile's scope."""
    from agent import secret_scope

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text((_isolated_config / "config.yaml").read_text(encoding="utf-8"),
                                              encoding="utf-8")
    secret_scope.set_multiplex_active(True)
    yield profile_home
    secret_scope.set_multiplex_active(False)


def _make_multiplex_runner(monkeypatch, profile_home):
    import types

    runner = _make_runner()
    monkeypatch.setattr(runner, "config", types.SimpleNamespace(multiplex_profiles=True), raising=False)
    monkeypatch.setattr(runner, "_resolve_profile_home_for_source", lambda source: profile_home, raising=False)
    return runner


@pytest.mark.asyncio
async def test_multiplexed_text_listing_filters_in_routed_profile_scope(_multiplexed_profile, monkeypatch):
    """Unscoped, the real explicit-config check raises ``UnscopedSecretError``; the filter must neither
    fail open (unconfigured DeepSeek back in the list) nor lose the listing — it runs in the profile's scope."""
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers",
                        lambda **kw: [dict(r) for r in _UNCONFIGURED_PROVIDER_ROWS])
    monkeypatch.setattr("hermes_cli.inventory._external_process_signed_in", lambda slug: False)

    reply = await _make_multiplex_runner(monkeypatch, _multiplexed_profile)._handle_model_command(_make_event())
    assert "`--provider openrouter`" in reply
    assert "DeepSeek" not in reply


@pytest.mark.asyncio
async def test_multiplexed_picker_lists_in_routed_profile_scope(_multiplexed_profile, monkeypatch):
    """The picker listing (and its explicit-only filter) runs in the same profile scope as the
    selection callback, so it judges the routed profile's config and ``.env``."""
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home_override

    seen: list[tuple] = []

    def _fake_list_picker_providers(**kwargs):
        seen.append((current_secret_scope() is not None, get_hermes_home_override()))
        return [{"slug": "openrouter", "name": "OpenRouter", "is_current": True,
                 "models": ["gpt-x"], "total_models": 1}]

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", _fake_list_picker_providers)
    runner = _make_multiplex_runner(monkeypatch, _multiplexed_profile)
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)

    assert await runner._handle_model_command(_make_event()) is None
    assert seen == [(True, str(_multiplexed_profile))], seen
