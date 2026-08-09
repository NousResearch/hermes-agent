"""Regression tests for stale ``model.base_url`` surviving a ``/model --global``
switch (2026-07-08: switching to ``nous`` left ``base_url`` pointed at the
previous provider's ``opencode.ai/zen/go/v1`` endpoint, silently mismatched).

Root cause: ``HermesCLI._apply_model_switch_result`` (interactive picker path)
and ``HermesCLI._handle_model_switch`` (typed ``/model <name> --global`` path)
both persisted ``model.default`` and ``model.provider`` on every switch but
never touched ``model.base_url`` at all, so whatever value was last written by
a different code path (a manual incident fix, the dashboard, or a Discord
``/model`` switch) survived forever. ``tui_gateway/server.py:_persist_model_switch``
already fixed the same bug for the TUI gateway path (#48305); these two CLI
call sites get the same fix.

The clear-on-switch half is gated on ``result.provider_changed`` (not written
unconditionally) to avoid a second regression: picking a different model on
the SAME custom/named endpoint must not null out a working base_url the user
configured elsewhere, which is exactly what already happened once in
``hermes_cli/web_server.py:_apply_main_model_assignment`` before it was fixed
there too.

The same two call sites also never touched ``model.api_key`` / ``model.api_mode``,
which can go stale the same way (a key set for a bare "custom" endpoint
surviving a switch to a provider that resolves credentials from the pool
instead). ``gateway/slash_commands.py`` already guards this via
``clear_model_endpoint_credentials``; these two call sites now clear the same
fields under the same condition (target provider isn't literally "custom"),
but never WRITE a resolved credential back to disk, since ``result.api_key``
can be a live pool-rotated key that must not be pinned into config.yaml.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli.model_switch import ModelSwitchResult


def _bound(fn, instance):
    return fn.__get__(instance, type(instance))


class _StubCLI:
    """Minimum attrs ``_apply_model_switch_result`` reads/writes on ``self``."""
    agent = None
    model = ""
    provider = ""
    requested_provider = ""
    api_key = ""
    _explicit_api_key = ""
    base_url = ""
    _explicit_base_url = ""
    api_mode = ""
    _pending_model_switch_note = ""


def _run_apply(monkeypatch, result, persist_global: bool):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    save_mock = MagicMock()
    monkeypatch.setattr(cli_mod, "save_config_value", save_mock)
    cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, persist_global)
    return save_mock


class TestApplyModelSwitchResultPersistsBaseUrl:
    """Picker path (``_apply_model_switch_result``)."""

    def test_clears_base_url_when_switching_to_a_provider_that_needs_none(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="",
            base_url="",  # switch_model() correctly resolved no override for nous
            api_mode="",
            provider_label="Nous",
            is_global=True,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=True)

        save_mock.assert_any_call("model.default", "tencent/hy3:free")
        save_mock.assert_any_call("model.provider", "nous")
        # The stale base_url from whatever provider was active before must be
        # explicitly cleared, not silently left untouched.
        save_mock.assert_any_call("model.base_url", None)

    def test_writes_base_url_when_new_provider_needs_one(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="deepseek-v4-flash",
            target_provider="opencode-go",
            provider_changed=True,
            api_key="",
            base_url="https://opencode.ai/zen/go/v1",
            api_mode="",
            provider_label="OpenCode Go",
            is_global=True,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=True)

        save_mock.assert_any_call("model.base_url", "https://opencode.ai/zen/go/v1")

    def test_same_provider_reassignment_does_not_clear_base_url(self, monkeypatch):
        """Picking a different model on the SAME custom/named provider, where
        resolution happens to come back with an empty base_url, must NOT wipe
        the endpoint that's still configured elsewhere. Only a genuine
        provider change may clear it."""
        result = ModelSwitchResult(
            success=True,
            new_model="some-other-model",
            target_provider="custom:mylocal",
            provider_changed=False,
            api_key="",
            base_url="",
            api_mode="",
            provider_label="My Local",
            is_global=True,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=True)

        calls = [c.args for c in save_mock.call_args_list]
        assert ("model.base_url", None) not in calls
        assert not any(c[0] == "model.base_url" for c in calls), (
            f"base_url must be left untouched on a same-provider switch, got: {calls}"
        )

    def test_session_only_switch_does_not_touch_config(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            base_url="",
            provider_label="Nous",
            is_global=False,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=False)

        save_mock.assert_not_called()

    def test_real_config_round_trip_clears_stale_base_url(self, tmp_path, monkeypatch):
        """Same incident as the _handle_model_switch round-trip test below,
        but through the interactive picker path."""
        import yaml
        import cli as cli_mod

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "model:\n"
            "  default: deepseek-v4-flash\n"
            "  provider: opencode-go\n"
            "  base_url: https://opencode.ai/zen/go/v1\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            base_url="",
            provider_label="Nous",
            is_global=True,
        )
        cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, True)

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["default"] == "tencent/hy3:free"
        assert saved["model"]["provider"] == "nous"
        assert not saved["model"].get("base_url"), (
            f"stale base_url survived the switch: {saved['model'].get('base_url')!r}"
        )


def _run_handle_model_switch(monkeypatch, cmd, result, *, current_provider="opencode-go",
                              current_base_url="https://opencode.ai/zen/go/v1"):
    import cli as cli_mod

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    save_mock = MagicMock()
    monkeypatch.setattr(cli_mod, "save_config_value", save_mock)

    self_ = SimpleNamespace(
        agent=None,
        model="deepseek-v4-flash",
        provider=current_provider,
        requested_provider=current_provider,
        api_key="",
        _explicit_api_key="",
        base_url=current_base_url,
        _explicit_base_url=current_base_url,
        api_mode="",
        conversation_history=[],
        _confirm_expensive_model_switch=lambda _result: True,
    )
    _bound(cli_mod.HermesCLI._handle_model_switch, self_)(cmd)
    return save_mock


class TestHandleModelSwitchPersistsBaseUrl:
    """Typed ``/model <name> --provider <provider> --global`` path, the one
    Sahil actually types."""

    def test_clears_base_url_when_switching_to_native_provider(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="",
            base_url="",
            provider_label="Nous",
            is_global=True,
        )
        save_mock = _run_handle_model_switch(
            monkeypatch, "/model tencent/hy3:free --provider nous --global", result,
        )

        save_mock.assert_any_call("model.default", "tencent/hy3:free")
        save_mock.assert_any_call("model.provider", "nous")
        save_mock.assert_any_call("model.base_url", None)

    def test_writes_base_url_when_switching_to_endpoint_provider(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="minimax-m3",
            target_provider="opencode-go",
            provider_changed=False,
            api_key="",
            base_url="https://opencode.ai/zen/go/v1",
            provider_label="OpenCode Go",
            is_global=True,
        )
        save_mock = _run_handle_model_switch(
            monkeypatch, "/model minimax-m3 --provider opencode-go --global", result,
        )

        save_mock.assert_any_call("model.base_url", "https://opencode.ai/zen/go/v1")

    def test_same_provider_reassignment_does_not_clear_base_url(self, monkeypatch):
        """Regression guard: picking another model on the SAME custom endpoint
        (no provider change) must never null the working base_url, even if
        resolution happens to come back empty for that call."""
        result = ModelSwitchResult(
            success=True,
            new_model="some-other-model",
            target_provider="custom:mylocal",
            provider_changed=False,
            api_key="",
            base_url="",
            provider_label="My Local",
            is_global=True,
        )
        save_mock = _run_handle_model_switch(
            monkeypatch, "/model some-other-model --global", result,
            current_provider="custom:mylocal",
            current_base_url="http://localhost:1234/v1",
        )

        calls = [c.args for c in save_mock.call_args_list]
        assert not any(c[0] == "model.base_url" for c in calls), (
            f"base_url must be left untouched on a same-provider switch, got: {calls}"
        )


class TestHandleModelSwitchRealConfigRoundTrip:
    """End-to-end against a real temp config.yaml, not just asserting the
    save_config_value call was made, but that it actually clears the stale
    field on disk. Mirrors tests/test_tui_gateway_server.py's
    test_persist_model_switch_clears_stale_base_url (#48305) for the sibling
    fix in tui_gateway/server.py._persist_model_switch."""

    def test_reproduces_and_fixes_the_2026_07_08_nous_incident(self, tmp_path, monkeypatch):
        import yaml
        import cli as cli_mod

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "model:\n"
            "  default: deepseek-v4-flash\n"
            "  provider: opencode-go\n"
            "  base_url: https://opencode.ai/zen/go/v1\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="",
            base_url="",  # nous is a native provider, correctly resolves to none
            provider_label="Nous",
            is_global=True,
        )
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
        monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

        self_ = SimpleNamespace(
            agent=None,
            model="deepseek-v4-flash",
            provider="opencode-go",
            requested_provider="opencode-go",
            api_key="",
            _explicit_api_key="",
            base_url="https://opencode.ai/zen/go/v1",
            _explicit_base_url="https://opencode.ai/zen/go/v1",
            api_mode="",
            conversation_history=[],
            _confirm_expensive_model_switch=lambda _result: True,
        )
        _bound(cli_mod.HermesCLI._handle_model_switch, self_)(
            "/model tencent/hy3:free --provider nous --global"
        )

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["default"] == "tencent/hy3:free"
        assert saved["model"]["provider"] == "nous"
        # This is the exact bug: base_url from opencode-go must not survive
        # the switch to nous. Before the fix this assertion failed, the
        # stale opencode.ai/zen/go/v1 URL was still there.
        assert not saved["model"].get("base_url"), (
            f"stale base_url survived the switch: {saved['model'].get('base_url')!r}"
        )

    def test_same_provider_switch_preserves_working_base_url_on_disk(self, tmp_path, monkeypatch):
        """The regression the gating fix (High #1 in review) exists to
        prevent: re-picking a model on the same custom endpoint must not
        wipe a working base_url from config.yaml."""
        import yaml
        import cli as cli_mod

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "model:\n"
            "  default: old-model\n"
            "  provider: custom:mylocal\n"
            "  base_url: http://localhost:1234/v1\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = ModelSwitchResult(
            success=True,
            new_model="new-model",
            target_provider="custom:mylocal",
            provider_changed=False,
            base_url="",  # resolution came back empty for this same-provider call
            provider_label="My Local",
            is_global=True,
        )
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
        monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

        self_ = SimpleNamespace(
            agent=None,
            model="old-model",
            provider="custom:mylocal",
            requested_provider="custom:mylocal",
            api_key="",
            _explicit_api_key="",
            base_url="http://localhost:1234/v1",
            _explicit_base_url="http://localhost:1234/v1",
            api_mode="",
            conversation_history=[],
            _confirm_expensive_model_switch=lambda _result: True,
        )
        _bound(cli_mod.HermesCLI._handle_model_switch, self_)("/model new-model --global")

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["default"] == "new-model"
        assert saved["model"]["base_url"] == "http://localhost:1234/v1", (
            f"working base_url was wiped on a same-provider switch: {saved['model'].get('base_url')!r}"
        )


class TestModelSwitchClearsStaleCredentials:
    """model.api_key / model.api_mode must be cleared when switching away
    from a bare "custom" endpoint, but a resolved credential must never be
    written back (that would pin a pool-rotated key into config.yaml)."""

    def test_apply_path_clears_credentials_when_leaving_custom(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="sk-live-pool-rotated-key",  # switch_model() resolved a real key
            base_url="",
            api_mode="chat_completions",
            provider_label="Nous",
            is_global=True,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=True)

        save_mock.assert_any_call("model.api_key", None)
        save_mock.assert_any_call("model.api_mode", None)
        # The resolved credential must never be written to disk, only cleared.
        calls = [c.args for c in save_mock.call_args_list]
        assert ("model.api_key", "sk-live-pool-rotated-key") not in calls

    def test_apply_path_preserves_credentials_on_bare_custom(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="local-model",
            target_provider="custom",
            provider_changed=False,
            api_key="",
            base_url="http://localhost:1234/v1",
            api_mode="",
            provider_label="Custom endpoint",
            is_global=True,
        )
        save_mock = _run_apply(monkeypatch, result, persist_global=True)

        calls = [c.args for c in save_mock.call_args_list]
        assert not any(c[0] == "model.api_key" for c in calls), (
            f"api_key must survive a switch that stays on bare custom: {calls}"
        )
        assert not any(c[0] == "model.api_mode" for c in calls)

    def test_handle_path_clears_credentials_when_leaving_custom(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="sk-live-pool-rotated-key",
            base_url="",
            api_mode="chat_completions",
            provider_label="Nous",
            is_global=True,
        )
        save_mock = _run_handle_model_switch(
            monkeypatch, "/model tencent/hy3:free --provider nous --global", result,
            current_provider="custom", current_base_url="http://localhost:1234/v1",
        )

        save_mock.assert_any_call("model.api_key", None)
        save_mock.assert_any_call("model.api_mode", None)
        calls = [c.args for c in save_mock.call_args_list]
        assert ("model.api_key", "sk-live-pool-rotated-key") not in calls

    def test_handle_path_preserves_credentials_on_bare_custom(self, monkeypatch):
        result = ModelSwitchResult(
            success=True,
            new_model="another-local-model",
            target_provider="custom",
            provider_changed=False,
            api_key="",
            base_url="http://localhost:1234/v1",
            provider_label="Custom endpoint",
            is_global=True,
        )
        save_mock = _run_handle_model_switch(
            monkeypatch, "/model another-local-model --provider custom --global", result,
            current_provider="custom", current_base_url="http://localhost:1234/v1",
        )

        calls = [c.args for c in save_mock.call_args_list]
        assert not any(c[0] == "model.api_key" for c in calls)
        assert not any(c[0] == "model.api_mode" for c in calls)

    def test_real_config_round_trip_clears_stale_credentials(self, tmp_path, monkeypatch):
        """A model.api_key/api_mode left over from a bare-custom setup must
        not survive a switch to a pool/env-backed provider."""
        import yaml
        import cli as cli_mod

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "model:\n"
            "  default: local-model\n"
            "  provider: custom\n"
            "  base_url: http://localhost:1234/v1\n"
            "  api_key: sk-stale-local-key\n"
            "  api_mode: chat_completions\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)

        result = ModelSwitchResult(
            success=True,
            new_model="tencent/hy3:free",
            target_provider="nous",
            provider_changed=True,
            api_key="sk-live-pool-rotated-key",
            base_url="",
            provider_label="Nous",
            is_global=True,
        )
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)

        self_ = SimpleNamespace(
            agent=None,
            model="local-model",
            provider="custom",
            requested_provider="custom",
            api_key="",
            _explicit_api_key="",
            base_url="http://localhost:1234/v1",
            _explicit_base_url="http://localhost:1234/v1",
            api_mode="chat_completions",
            conversation_history=[],
            _confirm_expensive_model_switch=lambda _result: True,
        )
        _bound(cli_mod.HermesCLI._handle_model_switch, self_)(
            "/model tencent/hy3:free --provider nous --global"
        )

        saved = yaml.safe_load(cfg_path.read_text())
        assert saved["model"]["provider"] == "nous"
        assert not saved["model"].get("api_key"), (
            f"stale custom api_key survived the switch: {saved['model'].get('api_key')!r}"
        )
        assert not saved["model"].get("api_mode")
        # And the live pool-rotated key from switch_model()'s result must
        # never have been written to disk in the first place.
        assert saved["model"].get("api_key") != "sk-live-pool-rotated-key"
