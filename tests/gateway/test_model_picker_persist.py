"""Regression tests for gateway inline-keyboard model-picker persistence.

#49066 made the typed ``/model <name>`` command persist the selected model to
``config.yaml`` by default. But the inline-keyboard picker callback
(``_on_model_selected`` in ``gateway/slash_commands.py``) was left session-only:
it hard-coded ``is_global=False`` and never wrote ``config.yaml``, so *tapping* a
model in the Telegram/Discord picker silently reverted on the next launch while
*typing* the same model persisted — a contradiction the same PR introduced.

After the fix (#49176), the picker callback honors the resolved
``persist_global`` and runs the same read-modify-write block the text path
uses, so a tapped model behaves exactly like a typed one.  Since the
session-scope-by-default change, both default to session-only and persist
only with ``--global`` (or ``model.persist_switch_by_default: true``).

These tests drive the real ``_handle_model_command`` with a fake picker-capable
adapter that captures the ``on_model_selected`` callback, then invoke that
callback and assert ``config.yaml`` is (or isn't) updated — exercising the exact
closure the PR changed, against a real temp ``HERMES_HOME``.
"""

import types

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakePickerAdapter:
    """Minimal adapter that looks picker-capable and captures the callback.

    ``_handle_model_command`` gates the picker path on
    ``getattr(type(adapter), "send_model_picker", None) is not None``, so the
    method must exist on the class, not just the instance.
    """

    def __init__(self):
        self.captured_callback = None

    async def send_model_picker(self, *, on_model_selected, **kwargs):
        # Stash the closure the handler built so the test can fire a "tap".
        self.captured_callback = on_model_selected
        return types.SimpleNamespace(success=True)


def _make_runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    """A successful ModelSwitchResult that bypasses real provider resolution."""
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openrouter",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        is_global=True,
    )


def _stub_picker_dependencies(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kw: [{"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5.5"]}],
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: 272000,
    )


def _setup_isolated_home(tmp_path, monkeypatch, model_yaml_value):
    """Write a config.yaml with the given ``model:`` value and stub heavy bits."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": model_yaml_value, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    _stub_picker_dependencies(monkeypatch)
    # save_config writes to ``get_hermes_home() / config.yaml`` — point it here.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


def _make_named_runner(monkeypatch, default_adapter, named_adapter, named_home):
    runner = _make_runner(default_adapter)
    monkeypatch.setattr(
        runner, "config", types.SimpleNamespace(multiplex_profiles=True), raising=False
    )
    monkeypatch.setattr(
        runner,
        "_profile_adapters",
        {"named": {Platform.TELEGRAM: named_adapter}},
        raising=False,
    )
    monkeypatch.setattr(
        runner, "_resolve_profile_home_for_source", lambda source: named_home
    )
    return runner


def _named_event(args):
    return MessageEvent(
        text=f"/model {args}".rstrip(),
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="named-chat",
            chat_type="dm",
            profile="named",
        ),
    )


async def _drive_picker(runner, event):
    """Run the handler (which sends the picker) then fire the captured tap."""
    sent = await runner._handle_model_command(event)
    # Bare /model returns None (picker sent); the adapter captured the callback.
    assert sent is None
    adapter = runner.adapters[Platform.TELEGRAM]
    assert adapter.captured_callback is not None, "picker callback was not wired"
    # Simulate the user tapping "gpt-5.5" under the openrouter provider.
    return await adapter.captured_callback("12345", "gpt-5.5", "openrouter")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed_model",
    [
        # Already-nested dict (common case).
        {
            "default": "old-model",
            "provider": "custom",
            "base_url": "https://api.custom.example/v1",
            "context_length": 1_048_576,
            "api_key": "sk-stale",
            "api_mode": "anthropic_messages",
        },
        # Flat-string model: must be coerced to a nested dict on a tap (same
        # scalar-``model:`` guard the text path has) instead of raising
        # ``TypeError`` on assignment.
        "deepseek-v4-flash",
    ],
    ids=["nested-dict", "flat-string"],
)
async def test_picker_tap_global_flag_persists(tmp_path, monkeypatch, seed_model):
    """Tapping a model in a ``/model --global`` picker persists to config.yaml,
    matching the typed ``/model --global`` path. The written ``model:`` must
    always end up a nested dict regardless of the seed shape."""
    adapter = _FakePickerAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, seed_model)

    confirmation = await _drive_picker(_make_runner(adapter), _make_event("/model --global"))

    assert confirmation is not None
    assert "gpt-5.5" in confirmation
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict), (
        "model: should be coerced to a dict, got %r" % (written["model"],)
    )
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"
    assert "base_url" not in written["model"]
    assert "api_key" not in written["model"]
    assert "api_mode" not in written["model"]
    assert "context_length" not in written["model"]


@pytest.mark.asyncio
async def test_picker_tap_is_session_scoped_by_default(tmp_path, monkeypatch):
    """Tapping a model in a bare ``/model`` picker applies an in-memory session
    override and does NOT touch config.yaml — switches are session-scoped
    unless the user opts in with ``--global`` (or sets
    ``model.persist_switch_by_default: true``)."""
    adapter = _FakePickerAdapter()
    cfg_path = _setup_isolated_home(
        tmp_path, monkeypatch, {"default": "old-model", "provider": "openrouter"}
    )
    runner = _make_runner(adapter)

    confirmation = await _drive_picker(runner, _make_event("/model"))

    assert confirmation is not None
    assert "gpt-5.5" in confirmation
    # The session override IS applied in-memory (the switch worked).
    assert runner._session_model_overrides, "session override should be set"
    assert any(
        ov.get("model") == "gpt-5.5"
        for ov in runner._session_model_overrides.values()
    )
    # But config.yaml is untouched — session-scoped by default.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "old-model"
    assert written["model"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_picker_tap_session_flag_does_not_persist(tmp_path, monkeypatch):
    """``/model --session`` then a picker tap stays in-memory only — config
    untouched, but the in-memory session override must still be applied (the
    switch worked, it just wasn't persisted)."""
    adapter = _FakePickerAdapter()
    cfg_path = _setup_isolated_home(
        tmp_path, monkeypatch, {"default": "old-model", "provider": "openai-codex"}
    )
    runner = _make_runner(adapter)

    confirmation = await _drive_picker(runner, _make_event("/model --session"))

    assert confirmation is not None
    assert "gpt-5.5" in confirmation
    # The session override IS applied in-memory (proves the path didn't no-op).
    assert runner._session_model_overrides, "session override should be set"
    assert any(
        ov.get("model") == "gpt-5.5"
        for ov in runner._session_model_overrides.values()
    )
    # But config.yaml is untouched — the override is in-memory only.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "old-model"
    assert written["model"]["provider"] == "openai-codex"


@pytest.mark.asyncio
async def test_multiplex_picker_keeps_profile_adapter_and_callback_scope(
    tmp_path, monkeypatch
):
    """A named profile must present and execute its picker under one identity."""
    from agent.secret_scope import get_secret, set_multiplex_active

    default_adapter = _FakePickerAdapter()
    named_adapter = _FakePickerAdapter()
    named_home = tmp_path / "profiles" / "named"
    named_home.mkdir(parents=True)
    (named_home / ".env").write_text("PROFILE_MODEL_KEY=named-secret\n", encoding="utf-8")
    runner = _make_named_runner(monkeypatch, default_adapter, named_adapter, named_home)
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )
    resolved = []

    def _profile_switch(**kwargs):
        resolved.append(get_secret("PROFILE_MODEL_KEY"))
        return _fake_switch_result()

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _profile_switch)
    event = _named_event("--session")

    set_multiplex_active(True)
    try:
        sent = await runner._handle_model_command(event)

        assert sent is None
        assert default_adapter.captured_callback is None
        assert named_adapter.captured_callback is not None
        assert resolved == []

        confirmation = await named_adapter.captured_callback(
            "named-chat", "gpt-5.5", "openrouter"
        )
    finally:
        set_multiplex_active(False)

    assert "gpt-5.5" in confirmation
    assert resolved == ["named-secret"]


@pytest.mark.asyncio
async def test_multiplex_picker_global_persists_only_named_profile(
    tmp_path, monkeypatch
):
    """A named picker must not seed its global write from the default profile."""
    import gateway.run as gateway_run
    from agent.secret_scope import set_multiplex_active

    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "named"
    default_home.mkdir(parents=True)
    named_home.mkdir(parents=True)
    default_cfg = {
        "marker": "default",
        "model": {"default": "default-old", "provider": "openai-codex"},
    }
    named_cfg = {
        "marker": "named",
        "model": {"default": "named-old", "provider": "openai-codex"},
    }
    (default_home / "config.yaml").write_text(
        yaml.safe_dump(default_cfg, sort_keys=False), encoding="utf-8"
    )
    (named_home / "config.yaml").write_text(
        yaml.safe_dump(named_cfg, sort_keys=False), encoding="utf-8"
    )

    default_adapter = _FakePickerAdapter()
    named_adapter = _FakePickerAdapter()
    runner = _make_named_runner(monkeypatch, default_adapter, named_adapter, named_home)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    _stub_picker_dependencies(monkeypatch)
    event = _named_event("--global")

    set_multiplex_active(True)
    try:
        with gateway_run._profile_runtime_scope(named_home):
            sent = await runner._handle_model_command(event)
        assert sent is None
        assert named_adapter.captured_callback is not None
        confirmation = await named_adapter.captured_callback(
            "named-chat", "gpt-5.5", "openrouter"
        )
    finally:
        set_multiplex_active(False)

    assert "gpt-5.5" in confirmation
    assert yaml.safe_load((default_home / "config.yaml").read_text()) == default_cfg
    written = yaml.safe_load((named_home / "config.yaml").read_text())
    assert written["marker"] == "named"
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"
