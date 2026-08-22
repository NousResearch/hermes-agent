"""Tests for the deliberate /model + /reasoning switch announce.

A slash command's confirmation reaches only the person who ran it (ephemeral on
native Discord slash), so before this feature everyone else in a shared channel
saw the assistant's model or reasoning effort change with no explanation. The
gateway now posts one channel-visible ``🔀`` line whenever a deliberate switch
changes the EFFECTIVE state, gated by ``model.announce_switch`` (default on) and
silent on a genuine no-op.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text, platform=Platform.DISCORD, user_id="u1", chat_id="c1"):
    source = SessionSource(
        platform=platform, user_id=user_id, chat_id=chat_id, user_name="tester"
    )
    return MessageEvent(text=text, source=source)


def _yaml_load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _make_runner(monkeypatch, tmp_path, config_yaml="agent:\n  reasoning_effort: xhigh\n"):
    """Bare GatewayRunner + a stub adapter that records ``send`` calls."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        gateway_run, "_load_gateway_config",
        lambda: _yaml_load(hermes_home / "config.yaml"),
    )
    monkeypatch.setattr(
        gateway_run, "_load_gateway_runtime_config",
        lambda: _yaml_load(hermes_home / "config.yaml"),
    )

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_reasoning_overrides = {}
    runner._session_model_overrides = {}
    runner._show_reasoning = False
    runner._reasoning_config = None
    runner._evict_cached_agent = MagicMock()
    runner._save_gateway_config_key = MagicMock(return_value=True)
    runner._normalize_source_for_session_key = lambda source: source
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._try_send_choice_picker = AsyncMock(return_value=False)

    adapter = MagicMock()
    adapter.send = AsyncMock()
    # No ``send_choice_picker`` on the type → the text path is taken.
    del adapter.send_choice_picker
    runner.adapters = {Platform.DISCORD: adapter}
    runner._adapter_for_source = lambda source: adapter
    return runner, adapter


def _sent_texts(adapter):
    return [call.args[1] for call in adapter.send.call_args_list]


class TestReasoningSwitchAnnounce:
    @pytest.mark.asyncio
    async def test_reasoning_switch_announces(self, tmp_path, monkeypatch):
        """A session pinned to high, switched to xhigh, announces high → xhigh."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        event = _make_event("/reasoning xhigh")
        session_key = runner._session_key_for_source(event.source)
        runner._session_reasoning_overrides[session_key] = {
            "enabled": True, "effort": "high",
        }

        await runner._handle_reasoning_command(event)

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: high → xhigh" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_old_side_is_the_config_default_when_unset(self, tmp_path, monkeypatch):
        """With no session override the baseline is the resolved config default.

        Not the empty string — otherwise the "old" side of the line would be
        blank for the most common case.
        """
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._handle_reasoning_command(_make_event("/reasoning high"))

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: xhigh → high" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_silent_on_noop_against_config_default(self, tmp_path, monkeypatch):
        """/reasoning xhigh on a defaulted-to-xhigh session changes nothing.

        This is why the announce baseline resolves the config fallback: a naive
        "" baseline would compare "" != "xhigh" and announce a phantom switch.
        """
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._handle_reasoning_command(_make_event("/reasoning xhigh"))

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_silent_on_noop_against_explicit_override(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        event = _make_event("/reasoning high")
        session_key = runner._session_key_for_source(event.source)
        runner._session_reasoning_overrides[session_key] = {
            "enabled": True, "effort": "high",
        }

        await runner._handle_reasoning_command(event)

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_reset_announces_back_to_config_default(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        event = _make_event("/reasoning reset")
        session_key = runner._session_key_for_source(event.source)
        runner._session_reasoning_overrides[session_key] = {
            "enabled": True, "effort": "low",
        }

        await runner._handle_reasoning_command(event)

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: low → xhigh" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_unconfigured_effort_still_announces(self, tmp_path, monkeypatch):
        """With no ``agent.reasoning_effort`` at all, a switch still announces.

        The resolver returns None for an unconfigured effort. If the baseline
        passed that through as "", the announce's own empty-string guard would
        swallow a genuine switch and the channel would see nothing — the exact
        silence this feature exists to remove. The baseline resolves the
        provider default (medium) instead.
        """
        runner, adapter = _make_runner(monkeypatch, tmp_path, "agent: {}\n")

        await runner._handle_reasoning_command(_make_event("/reasoning high"))

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: medium → high" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_display_toggle_does_not_announce(self, tmp_path, monkeypatch):
        """show/hide changes rendering, not the effort — nothing to announce."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._handle_reasoning_command(_make_event("/reasoning show"))

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_unknown_level_does_not_announce(self, tmp_path, monkeypatch):
        """A rejected argument applied nothing, so it must announce nothing."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._handle_reasoning_command(_make_event("/reasoning bogus"))

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_disabled_reasoning_renders_as_none(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._handle_reasoning_command(_make_event("/reasoning none"))

        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: xhigh → none" in text for text in texts), texts

    @pytest.mark.asyncio
    async def test_baseline_tracks_the_sessions_effective_model(self, tmp_path, monkeypatch):
        """A per-model override for the session's /model-switched model wins.

        The announce baseline must resolve against the model the session
        actually runs, not config ``model.default`` — otherwise the "old" side
        reports an effort that session never had.
        """
        runner, adapter = _make_runner(
            monkeypatch, tmp_path,
            "model:\n"
            "  default: base-model\n"
            "agent:\n"
            "  reasoning_effort: medium\n"
            "  reasoning_overrides:\n"
            "    switched-model: low\n",
        )
        event = _make_event("/reasoning high")
        session_key = runner._session_key_for_source(event.source)
        runner._session_model_overrides[session_key] = {"model": "switched-model"}

        await runner._handle_reasoning_command(event)

        texts = _sent_texts(adapter)
        # low (the switched model's override), NOT medium (the global default).
        assert any("🔀 Reasoning: low → high" in text for text in texts), texts


class TestModelSwitchAnnounce:
    @pytest.mark.asyncio
    async def test_model_switch_announces_full_route(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        source = _make_event("/model x").source

        await runner._announce_switch(
            source, "model", "anthropic/opus", "openai/gpt-5.4",
        )

        texts = _sent_texts(adapter)
        assert any("🔀 Model: anthropic/opus → openai/gpt-5.4" in t for t in texts), texts

    @pytest.mark.asyncio
    async def test_same_route_is_silent(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        source = _make_event("/model x").source

        await runner._announce_switch(
            source, "model", "anthropic/opus", "anthropic/opus",
        )

        assert adapter.send.call_count == 0

    @pytest.mark.asyncio
    async def test_same_model_different_provider_announces(self, tmp_path, monkeypatch):
        """A same-slug re-route is a real change the channel should see."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        source = _make_event("/model x").source

        await runner._announce_switch(
            source, "model", "provider-a/same-model", "provider-b/same-model",
        )

        assert adapter.send.call_count == 1, _sent_texts(adapter)


class TestAnnounceGate:
    @pytest.mark.asyncio
    async def test_gate_off_suppresses(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(
            monkeypatch, tmp_path,
            "agent:\n  reasoning_effort: xhigh\nmodel:\n  announce_switch: false\n",
        )

        await runner._handle_reasoning_command(_make_event("/reasoning high"))

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_gate_on_by_default(self, tmp_path, monkeypatch):
        """An untouched config announces — the feature is opt-OUT."""
        runner, adapter = _make_runner(monkeypatch, tmp_path, "agent: {}\n")
        source = _make_event("/model x").source

        await runner._announce_switch(source, "model", "a/b", "c/d")

        assert adapter.send.call_count == 1, _sent_texts(adapter)

    @pytest.mark.parametrize("raw", [False, "false", "no", "off", 0, "OFF"])
    def test_falsey_spellings_disable(self, raw):
        assert gateway_run.GatewayRunner._switch_announce_enabled(
            {"model": {"announce_switch": raw}}
        ) is False

    @pytest.mark.parametrize(
        "cfg",
        [None, {}, {"model": None}, {"model": {}}, {"model": "a-bare-string"}],
    )
    def test_gate_fails_open(self, cfg):
        """Silence must never be the failure mode of an unreadable config.

        A bare-string ``model:`` (a real shape users write) makes the nested
        ``.get`` raise; the gate has to swallow that and stay ON.
        """
        assert gateway_run.GatewayRunner._switch_announce_enabled(cfg) is True


class TestAnnounceIsBestEffort:
    @pytest.mark.asyncio
    async def test_send_failure_does_not_break_the_handler(self, tmp_path, monkeypatch):
        """The user must still get their confirmation when the announce fails."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        adapter.send = AsyncMock(side_effect=RuntimeError("adapter exploded"))

        result = await runner._handle_reasoning_command(_make_event("/reasoning high"))

        assert isinstance(result, str) and result

    @pytest.mark.asyncio
    async def test_no_adapter_is_not_an_error(self, tmp_path, monkeypatch):
        runner, _adapter = _make_runner(monkeypatch, tmp_path)
        runner._adapter_for_source = lambda source: None

        await runner._announce_switch(_make_event("/x").source, "model", "a/b", "c/d")

    @pytest.mark.asyncio
    async def test_unknown_kind_sends_nothing(self, tmp_path, monkeypatch):
        """The catalog key is looked up from a fixed enum.

        A typo'd kind must not post a raw dotted i18n key to the channel.
        """
        runner, adapter = _make_runner(monkeypatch, tmp_path)

        await runner._announce_switch(_make_event("/x").source, "Model", "a/b", "c/d")

        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_metadata_failure_still_sends(self, tmp_path, monkeypatch):
        """Thread metadata is a nicety; losing it must not lose the announce."""
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        runner._thread_metadata_for_source = MagicMock(
            side_effect=RuntimeError("no thread")
        )

        await runner._announce_switch(_make_event("/x").source, "model", "a/b", "c/d")

        assert adapter.send.call_count == 1
        assert adapter.send.call_args.kwargs["metadata"] is None


class TestAnnounceLocaleCatalog:
    def test_every_locale_carries_both_keys_with_placeholders(self):
        """The announce is the one user-facing string this feature ships.

        ``test_i18n`` already pins catalog parity; this pins that the keys the
        code actually looks up exist and keep their substitutions.
        """
        import pathlib

        locales_dir = pathlib.Path(gateway_run.__file__).resolve().parents[1] / "locales"
        catalogs = sorted(locales_dir.glob("*.yaml"))
        assert catalogs, "no locale catalogs found"

        for path in catalogs:
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            section = (data.get("gateway") or {}).get("switch_announce") or {}
            for kind, dotted in gateway_run._SWITCH_ANNOUNCE_KEYS.items():
                assert dotted == f"gateway.switch_announce.{kind}"
                value = section.get(kind)
                assert value, f"{path.name}: missing gateway.switch_announce.{kind}"
                assert "{old}" in value and "{new}" in value, (
                    f"{path.name}: gateway.switch_announce.{kind} lost a placeholder"
                )
