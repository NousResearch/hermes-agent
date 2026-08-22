"""Regression: /model must read and persist against the ROUTED profile's
config.yaml when gateway.multiplex_profiles is on, not the default profile.

Issue #69178, site 2: ``_handle_model_command`` resolved
``_command_profile_home`` (via ``_resolve_profile_home_for_source``) purely
to compute a ``config_path`` variable used later for an ``os.path.exists``
check, but the actual reads (``_load_gateway_config()``) and the final
``save_config()`` write-through were never wrapped in
``_profile_runtime_scope(_command_profile_home)``. Both silently operated
on the *default* profile's HERMES_HOME, so a user who ran ``/model`` from a
channel routed to a secondary profile saw the default profile's current
model and, worse, ``--global`` persisted the switch into the default
profile's config instead of the routed one.

The fix scopes both the initial config read and the ``_finish_switch``
persist block in ``with _profile_runtime_scope(_command_profile_home):``,
mirroring the existing pattern used by the interactive picker callback.
"""

import threading
import types

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _MultiplexConfig:
    multiplex_profiles = True


def _make_runner(profile_home):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner.config = _MultiplexConfig()
    runner._resolve_profile_home_for_source = lambda source: profile_home
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="12345", chat_type="group"),
    )


def _fake_switch_result(model="gpt-5.5", provider="openrouter"):
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        provider_changed=True,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        is_global=True,
    )


def _write_config(home, model_name, provider="openrouter"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": model_name, "provider": provider}, "providers": {}}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_model_global_persists_to_routed_profile_not_default(tmp_path, monkeypatch):
    """/model --global from a secondary-profile-routed source must write
    that profile's config.yaml, leaving the default profile's config
    untouched."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )

    runner = _make_runner(profile_home)
    result = await runner._handle_model_command(_make_event("/model gpt-5.5 --global"))

    assert result is not None
    assert "gpt-5.5" in result

    written_profile = yaml.safe_load((profile_home / "config.yaml").read_text(encoding="utf-8"))
    written_default = yaml.safe_load((default_home / "config.yaml").read_text(encoding="utf-8"))

    assert written_profile["model"]["default"] == "gpt-5.5", (
        "the routed profile's config.yaml should have been rewritten with the new model"
    )
    assert written_default["model"]["default"] == "default-model", (
        "the DEFAULT profile's config.yaml must be untouched by a secondary-profile /model --global"
    )


@pytest.mark.asyncio
async def test_model_reads_current_model_from_routed_profile(tmp_path, monkeypatch):
    """The pre-switch ``current_model`` shown/used by /model must come from
    the routed profile's config, not the default profile's."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured = {}

    def _fake_switch_model(**kwargs):
        captured["current_model"] = kwargs.get("current_model")
        return _fake_switch_result()

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)

    runner = _make_runner(profile_home)
    await runner._handle_model_command(_make_event("/model gpt-5.5 --global"))

    assert captured["current_model"] == "profile-model", (
        "current_model passed to switch_model() should reflect the ROUTED "
        "profile's config, not the default profile's"
    )


@pytest.mark.asyncio
async def test_model_switch_resolves_credentials_under_routed_profile_scope(
    tmp_path, monkeypatch
):
    """Regression (Sol xhigh follow-up review of #69178): the ``_switch_model``
    call — and any credential resolution it performs internally via
    ``resolve_runtime_provider`` / ``get_secret`` — must run under the ROUTED
    profile's ``_profile_runtime_scope``, not unscoped.

    ``get_secret()`` (agent/secret_scope.py) reads a context-local secret
    scope installed by ``_profile_runtime_scope`` and falls back to ambient
    ``os.environ`` only when no scope is installed. This test does NOT mock
    ``switch_model`` into a no-op (Sol's point: that hides the defect) — the
    fake instead performs a real credential read via ``get_secret`` from
    *inside* the offloaded thread, exactly like ``resolve_runtime_provider``
    does for ``${ENV}``/``key_env`` credential branches, and asserts it sees
    the routed profile's secret rather than the ambient/default one.

    Before the fix: ``_switch_model`` ran via ``asyncio.to_thread`` outside
    ``_model_cmd_scope_factory()``, so no secret scope was installed in the
    thread's copied context and ``get_secret`` fell through to
    ``os.environ`` — i.e. whatever secret happened to be ambient (here,
    simulating the "wrong profile's" value already sitting in the process
    environment).

    After the fix: the call runs inside ``with _model_cmd_scope_factory():``,
    which is a real ``_profile_runtime_scope(profile_home)`` — a
    contextvars-based scope that ``asyncio.to_thread`` propagates into the
    worker thread (`contextvars.copy_context()` + ``ctx.run`` — verified via
    ``asyncio.to_thread``'s source). ``get_secret`` then resolves from the
    profile's own ``.env``, ignoring the ambient value.
    """
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    # Profile-scoped secret: only visible when _profile_runtime_scope for
    # `profile_home` is active (installed via the profile's own .env).
    (profile_home / ".env").write_text(
        "HERMES_TEST_SECRET=profile-secret\n", encoding="utf-8"
    )

    # Ambient/ "wrong profile" value already sitting in os.environ — this is
    # what a fail-open, unscoped read would leak.
    monkeypatch.setenv("HERMES_TEST_SECRET", "leaked-ambient-secret")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured = {}

    def _fake_switch_model(**kwargs):
        from agent.secret_scope import get_secret

        captured["resolved_secret"] = get_secret("HERMES_TEST_SECRET")
        return _fake_switch_result()

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)

    runner = _make_runner(profile_home)
    result = await runner._handle_model_command(_make_event("/model gpt-5.5 --global"))

    assert result is not None
    assert captured["resolved_secret"] == "profile-secret", (
        "credential resolution inside the offloaded _switch_model call must "
        "see the ROUTED profile's secret scope, not the ambient/default "
        f"value (got {captured.get('resolved_secret')!r})"
    )


@pytest.mark.asyncio
async def test_model_refresh_clears_only_routed_profile_cache(tmp_path, monkeypatch):
    """/model --refresh from a routed source must clear the ROUTED profile's
    provider-models cache. The cache path resolves through get_hermes_home(),
    so an unscoped clear deletes the default profile's cache and leaves the
    routed profile's stale file in place (#69242 sweeper review)."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")
    default_cache = default_home / "provider_models_cache.json"
    profile_cache = profile_home / "provider_models_cache.json"
    default_cache.write_text("{}", encoding="utf-8")
    profile_cache.write_text("{}", encoding="utf-8")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )

    runner = _make_runner(profile_home)
    result = await runner._handle_model_command(
        _make_event("/model gpt-5.5 --global --refresh")
    )
    assert result is not None

    assert not profile_cache.exists(), (
        "the routed profile's cache must be cleared by --refresh"
    )
    assert default_cache.exists(), (
        "the DEFAULT profile's cache must survive a routed /model --refresh"
    )


@pytest.mark.asyncio
async def test_model_refresh_lists_providers_under_routed_scope(tmp_path, monkeypatch):
    """A bare routed ``/model --refresh`` (no target) falls through to the
    provider listing. That listing reads and repopulates the catalog cache
    through get_hermes_home(), so it must observe the ROUTED profile's home,
    not the default one (#69242 sweeper review, second pass)."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    observed = []

    def _observing_list(**_kw):
        from hermes_constants import get_hermes_home

        observed.append(get_hermes_home())
        return []

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers", _observing_list
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", _observing_list
    )

    runner = _make_runner(profile_home)
    await runner._handle_model_command(_make_event("/model --refresh"))

    assert observed, "provider listing was never reached"
    assert all(h == profile_home for h in observed), (
        f"provider listing observed {observed} instead of the routed "
        f"profile home {profile_home}"
    )


# ---------------------------------------------------------------------------
# Rebase-composition coverage: the routed-profile scope (#69242) must survive
# main's event-loop offloads (asyncio.to_thread / *_async wrappers). Each test
# asserts BOTH halves at once: the profile-dependent read happens (a) off the
# event-loop thread and (b) under the ROUTED profile's scope, relying on
# asyncio.to_thread's contextvars propagation into the worker thread.
# ---------------------------------------------------------------------------


class _CapturingPickerAdapter:
    """Picker-capable adapter (method on the *class*, per the handler's
    ``getattr(type(adapter), "send_model_picker", None)`` gate) that stashes
    the ``on_model_selected`` closure so the test can fire a tap."""

    def __init__(self):
        self.captured_callback = None

    async def send_model_picker(self, *, on_model_selected, **kwargs):
        self.captured_callback = on_model_selected
        return types.SimpleNamespace(success=True)


def _stub_context_length(monkeypatch, value=272000):
    """Pin the sync display-context resolver so no provider probe runs."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: value,
    )


@pytest.mark.asyncio
async def test_typed_enrich_offload_sees_routed_profile_in_worker_thread(
    tmp_path, monkeypatch
):
    """T1 (typed path): ``enrich_model_switch_warnings_for_gateway`` is
    offloaded via ``await asyncio.to_thread(...)`` (main's #41289-family
    offload) *inside* ``with _model_cmd_scope_factory():`` (#69242). The
    worker thread must therefore still observe the ROUTED profile's
    HERMES_HOME — to_thread copies the calling context, so the ContextVar
    scope travels with the call. A composition that hoists the await out of
    the ``with`` keeps the gateway responsive but silently enriches against
    the DEFAULT profile's config."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    _stub_context_length(monkeypatch)

    loop_thread = threading.get_ident()
    observed = {}

    def _observing_enrich(*_a, **_kw):
        from hermes_constants import get_hermes_home

        observed["home"] = get_hermes_home()
        observed["thread"] = threading.get_ident()

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway",
        _observing_enrich,
    )

    runner = _make_runner(profile_home)
    result = await runner._handle_model_command(_make_event("/model gpt-5.5 --global"))

    assert result is not None
    assert observed, "enrich_model_switch_warnings_for_gateway was never invoked"
    assert observed["thread"] != loop_thread, (
        "enrich must run offloaded in a worker thread, not on the event loop"
    )
    assert observed["home"] == profile_home, (
        f"enrich observed HERMES_HOME {observed['home']} in the worker thread "
        f"instead of the routed profile home {profile_home}"
    )


@pytest.mark.asyncio
async def test_typed_context_length_async_sees_routed_profile_in_worker_thread(
    tmp_path, monkeypatch
):
    """T2 (typed path): the confirmation message resolves the display context
    length via ``await resolve_display_context_length_async(...)`` — a
    ``to_thread`` wrapper around the sync probe ladder — from inside
    ``_finish_switch``'s ``with _model_cmd_scope_factory():``. The sync
    resolver in the worker thread must see the ROUTED profile's HERMES_HOME
    (its provider probes read profile-relative config/caches/credentials)."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    # Keep the observation single-source: the real enrich also calls the sync
    # resolver internally, which would add a second (also-scoped) sample.
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway",
        lambda *_a, **_kw: None,
    )

    loop_thread = threading.get_ident()
    observed = {}

    def _observing_resolve(*_a, **_kw):
        from hermes_constants import get_hermes_home

        observed["home"] = get_hermes_home()
        observed["thread"] = threading.get_ident()
        return 272000

    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        _observing_resolve,
    )

    runner = _make_runner(profile_home)
    result = await runner._handle_model_command(_make_event("/model gpt-5.5 --global"))

    assert result is not None
    assert "gpt-5.5" in result
    assert observed, "resolve_display_context_length was never reached"
    assert observed["thread"] != loop_thread, (
        "the context-length resolver must run in the async wrapper's worker "
        "thread, not on the event loop"
    )
    assert observed["home"] == profile_home, (
        f"the context-length resolver observed HERMES_HOME {observed['home']} "
        f"instead of the routed profile home {profile_home}"
    )


@pytest.mark.asyncio
async def test_picker_enrich_offload_sees_routed_profile_in_worker_thread(
    tmp_path, monkeypatch
):
    """T3 (picker path): a tapped model runs ``_on_model_selected_scoped``
    under ``with _profile_runtime_scope(_picker_profile_home):`` (the
    ``_on_model_selected`` wrapper), and the enrich call inside it is
    offloaded via ``await asyncio.to_thread(...)``. The worker thread must
    observe the ROUTED profile's HERMES_HOME through the propagated scope."""
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "work"
    _write_config(default_home, "default-model")
    _write_config(profile_home, "profile-model")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kw: [
            {"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5.5"]}
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    _stub_context_length(monkeypatch)

    loop_thread = threading.get_ident()
    observed = {}

    def _observing_enrich(*_a, **_kw):
        from hermes_constants import get_hermes_home

        observed["home"] = get_hermes_home()
        observed["thread"] = threading.get_ident()

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway",
        _observing_enrich,
    )

    adapter = _CapturingPickerAdapter()
    runner = _make_runner(profile_home)
    runner.adapters = {Platform.DISCORD: adapter}

    sent = await runner._handle_model_command(_make_event("/model --global"))
    assert sent is None, "bare /model should send the picker and return None"
    assert adapter.captured_callback is not None, "picker callback was not wired"

    confirmation = await adapter.captured_callback("12345", "gpt-5.5", "openrouter")

    assert "gpt-5.5" in confirmation
    assert observed, "picker enrich was never invoked"
    assert observed["thread"] != loop_thread, (
        "picker enrich must run offloaded in a worker thread, not on the loop"
    )
    assert observed["home"] == profile_home, (
        f"picker enrich observed HERMES_HOME {observed['home']} in the worker "
        f"thread instead of the routed profile home {profile_home}"
    )
