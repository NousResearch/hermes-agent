"""Regression tests for gateway inline-keyboard model-picker persistence.

#49066 made the typed ``/model <name>`` command persist the selected model to
``config.yaml`` by default. But the inline-keyboard picker callback
(``_on_model_selected`` in ``gateway/slash_commands.py``) was left session-only:
it hard-coded ``is_global=False`` and never wrote ``config.yaml``, so *tapping* a
model in the Telegram/Discord picker silently reverted on the next launch while
*typing* the same model persisted — a contradiction the same PR introduced.

After the fix (#49176), the picker callback honors the resolved
``persist_global`` (defaults to ``True``, still respects ``--session``) and runs
the same read-modify-write block the text path uses, so a tapped model survives
across sessions like a typed one.

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
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    # The picker-setup path calls list_picker_providers, which otherwise hits
    # the network (OpenRouter model catalog). Stub it to a minimal list — these
    # tests capture and fire the on_model_selected callback and don't assert on
    # picker contents. The handler imports it as a local alias at call time, so
    # patching the source-module attribute takes effect.
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kw: [{"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5.5"]}],
    )
    # switch_model is imported as a local alias inside the handler
    # (`from hermes_cli.model_switch import switch_model as _switch_model`),
    # so patching the source-module attribute takes effect at call time.
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    # The confirmation builder resolves context length for display, which
    # otherwise makes real outbound HTTP calls (Ollama /api/show + the
    # OpenRouter models catalog). Stub it — these tests don't assert on the
    # displayed context, and the closure imports it lazily from this module.
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *a, **k: 272000,
    )
    # save_config writes to ``get_hermes_home() / config.yaml`` — point it here.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


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
        {"default": "old-model", "provider": "openai-codex"},
        # Flat-string model: must be coerced to a nested dict on a tap (same
        # scalar-``model:`` guard the text path has) instead of raising
        # ``TypeError`` on assignment.
        "deepseek-v4-flash",
    ],
    ids=["nested-dict", "flat-string"],
)
async def test_picker_tap_persists_by_default(tmp_path, monkeypatch, seed_model):
    """Tapping a model in the picker (bare /model) persists to config.yaml,
    matching the typed ``/model`` default — this is the #49176 fix. The written
    ``model:`` must always end up a nested dict regardless of the seed shape."""
    adapter = _FakePickerAdapter()
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, seed_model)

    confirmation = await _drive_picker(_make_runner(adapter), _make_event("/model"))

    assert confirmation is not None
    assert "gpt-5.5" in confirmation
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict), (
        "model: should be coerced to a dict, got %r" % (written["model"],)
    )
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"
    assert written["model"]["base_url"] == "https://openrouter.ai/api/v1"


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
