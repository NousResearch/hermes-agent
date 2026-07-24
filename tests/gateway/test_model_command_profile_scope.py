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
