"""Gateway ``/model`` must honour an operator-configured ``model.allowlist``.

When ``model.allowlist`` is set in config.yaml, ``/model`` may only switch to a
model in that set; a disallowed target (typed name, alias, or ``--provider``
auto-detect, checked on the RESOLVED model) is refused and nothing is applied.
An empty / unset allowlist means no restriction (the default), preserving the
prior behaviour. The picker + text list are also filtered to the allowlist.
"""

import types

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slash_commands import (
    _filter_providers_to_allowlist,
    _model_blocked_by_allowlist,
    _normalize_model_allowlist,
)


# ---------------------------------------------------------------------------
# Pure-helper unit tests (no event loop, no config)
# ---------------------------------------------------------------------------


def test_normalize_allowlist_handles_missing_and_malformed():
    assert _normalize_model_allowlist(None) == frozenset()
    assert _normalize_model_allowlist("not-a-list") == frozenset()
    assert _normalize_model_allowlist([]) == frozenset()
    # lowercases, trims, drops blanks + non-strings
    assert _normalize_model_allowlist(["A/B", " c/d ", "", 5, None]) == frozenset(
        {"a/b", "c/d"}
    )


def test_blocked_by_allowlist_semantics():
    allow = frozenset({"a/b", "c/d"})
    # an empty allowlist never blocks (unrestricted default)
    assert _model_blocked_by_allowlist("anything", frozenset()) is False
    # membership is case- and whitespace-insensitive
    assert _model_blocked_by_allowlist("A/B", allow) is False
    assert _model_blocked_by_allowlist("  c/d ", allow) is False
    # anything outside the set is blocked, including empty
    assert _model_blocked_by_allowlist("e/f", allow) is True
    assert _model_blocked_by_allowlist("", allow) is True


def test_filter_providers_drops_disallowed_models_and_empty_providers():
    allow = frozenset({"a/b"})
    providers = [
        {"name": "P1", "slug": "p1", "models": ["a/b", "x/y"], "total_models": 2},
        {"name": "P2", "slug": "p2", "models": ["x/y"], "total_models": 1},
    ]
    out = _filter_providers_to_allowlist(providers, allow)
    assert len(out) == 1
    assert out[0]["slug"] == "p1"
    assert out[0]["models"] == ["a/b"]
    assert out[0]["total_models"] == 1
    # no-op (same object) when the allowlist is empty
    assert _filter_providers_to_allowlist(providers, frozenset()) is providers


# ---------------------------------------------------------------------------
# Handler gate tests (mirror tests/gateway/test_model_command_expensive_confirm)
# ---------------------------------------------------------------------------


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._evict_cached_agent = lambda session_key: None
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _setup_home(tmp_path, monkeypatch, *, allowlist):
    """Isolated HERMES_HOME whose config carries (or omits) model.allowlist.

    ``switch_model`` is stubbed to ECHO the requested model as ``new_model`` so
    the allowlist gate (which checks the resolved model) is driven by the input.
    The expensive-model warning is stubbed off so an allowed switch applies
    immediately rather than routing through the confirm gate.
    """
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    model_cfg = {"default": "old-model", "provider": "openrouter"}
    if allowlist is not None:
        model_cfg["allowlist"] = allowlist
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": model_cfg, "providers": {}}), encoding="utf-8"
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **kw: None,
    )

    def _fake_switch(**kw):
        return ModelSwitchResult(
            success=True,
            new_model=kw["raw_input"],
            target_provider="openrouter",
            provider_changed=False,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)


@pytest.mark.asyncio
async def test_disallowed_model_is_refused(tmp_path, monkeypatch):
    """A typed model outside the allowlist is refused; nothing is applied."""
    _setup_home(tmp_path, monkeypatch, allowlist=["allowed/one", "allowed/two"])
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model evil/model"))

    assert result is not None
    assert "not in this bot's allowed model list" in result
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_allowed_model_switches(tmp_path, monkeypatch):
    """A model in the allowlist switches normally."""
    _setup_home(tmp_path, monkeypatch, allowlist=["allowed/one", "allowed/two"])
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model allowed/two"))

    assert result is not None
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "allowed/two"


@pytest.mark.asyncio
async def test_allowlist_membership_is_case_insensitive(tmp_path, monkeypatch):
    """Allowlist matching ignores case, so a differently-cased id still passes."""
    _setup_home(tmp_path, monkeypatch, allowlist=["Allowed/One"])
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model allowed/one"))

    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "allowed/one"


@pytest.mark.asyncio
async def test_no_allowlist_allows_any_model(tmp_path, monkeypatch):
    """With no allowlist configured, any model switches (back-compat)."""
    _setup_home(tmp_path, monkeypatch, allowlist=None)
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model any/model"))

    assert result is not None
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "any/model"


@pytest.mark.asyncio
async def test_gate_checks_resolved_model_not_typed_string(tmp_path, monkeypatch):
    """The gate must check the RESOLVED model, not the typed string.

    Mutation guard: the typed token "allowed/one" IS in the allowlist, but the
    resolver maps it to "resolved/blocked" (simulating an alias / --provider
    auto-detect). A gate that checked the typed string would wrongly allow it; a
    correct gate checks result.new_model and refuses. This pins the choice and
    fails if someone "simplifies" the check to use model_input.
    """
    _setup_home(tmp_path, monkeypatch, allowlist=["allowed/one"])
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: __import__("hermes_cli.model_switch", fromlist=["ModelSwitchResult"]).ModelSwitchResult(
            success=True,
            new_model="resolved/blocked",  # resolves to something NOT allowlisted
            target_provider="openrouter",
            provider_changed=False,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        ),
    )
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model allowed/one"))

    assert result is not None
    assert "not in this bot's allowed model list" in result
    assert "resolved/blocked" in result  # message names the resolved id
    assert runner._session_model_overrides == {}


class _FakePickerAdapter:
    """Picker-capable adapter that captures the on_model_selected closure.

    _handle_model_command gates the picker path on
    ``getattr(type(adapter), "send_model_picker", None) is not None``, so the
    method must exist on the class. Mirrors tests/gateway/test_model_picker_persist.
    """

    def __init__(self):
        self.captured_callback = None

    async def send_model_picker(self, *, on_model_selected, **kwargs):
        self.captured_callback = on_model_selected
        return types.SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_picker_callback_refuses_disallowed_model(tmp_path, monkeypatch):
    """The interactive-picker callback enforces the allowlist too (defense in
    depth — the picker is already filtered, but a tap on a disallowed id, e.g.
    from a stale keyboard, must still be refused with no side effect)."""
    _setup_home(tmp_path, monkeypatch, allowlist=["allowed/one"])
    # The picker setup path lists providers; return one allowed model so the
    # picker is shown and the callback is captured.
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **kw: [{"slug": "openrouter", "name": "OpenRouter", "models": ["allowed/one"]}],
    )
    adapter = _FakePickerAdapter()
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: adapter}

    # Bare /model sends the picker (returns None) and wires the callback.
    sent = await runner._handle_model_command(_make_event("/model"))
    assert sent is None
    assert adapter.captured_callback is not None, "picker callback was not wired"

    # Fire a tap on a disallowed model id — must be refused, nothing applied.
    reply = await adapter.captured_callback("12345", "evil/model", "openrouter")
    assert "not in this bot's allowed model list" in reply
    assert runner._session_model_overrides == {}
