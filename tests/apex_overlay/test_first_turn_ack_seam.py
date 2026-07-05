"""Seam-test + behavior test for apex_overlay.first_turn_ack (hc-401 SEAM C).

Pins the two upstream symbols the seam monkey-patches so an upstream rename turns
a silently-disarmed guard into a loud CI failure:

* ``gateway.display_config.resolve_display_setting`` (signature) — the DEFAULT +
  bool-coercion half hangs off it.
* ``gateway.run.GatewayRunner._handle_message_with_agent`` — the SEND half wraps
  it to fire the ack before the first model call. Its (event, source, _quick_key,
  run_generation) parameter order is what the wrapper forwards.

Behavior proven:
* first_turn_ack resolves True for a no-edit CN-IM platform (dingtalk) and False
  for feishu, with no config;
* a config override wins over the tier default (both directions);
* every OTHER display setting delegates to the original resolve unchanged;
* the wrapped turn method fires the ack (once, debounced) then calls through.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter).
"""

from __future__ import annotations

import asyncio
import inspect

from apex_overlay import first_turn_ack


# ---------------------------------------------------------------------------
# Seam assertions — pin the patched symbols
# ---------------------------------------------------------------------------

def test_seam_target_resolve_display_setting_signature():
    from gateway import display_config

    fn = getattr(display_config, first_turn_ack._TARGET_RESOLVE_FN, None)
    assert fn is not None, (
        "gateway.display_config.resolve_display_setting is gone — the "
        "first_turn_ack default can no longer be supplied. Update "
        "apex_overlay.first_turn_ack._TARGET_RESOLVE_FN."
    )
    params = list(inspect.signature(fn).parameters)
    assert params[:3] == ["user_config", "platform_key", "setting"], (
        f"resolve_display_setting param order changed to {params!r}; the overlay "
        f"wrapper forwards (user_config, platform_key, setting, fallback)."
    )


def test_seam_target_turn_method_exists_with_expected_params():
    from gateway.run import GatewayRunner

    method = getattr(GatewayRunner, first_turn_ack._TARGET_TURN_METHOD, None)
    assert method is not None, (
        "GatewayRunner._handle_message_with_agent is gone — the first_turn_ack "
        "SEND half can no longer fire before the first model call. Update "
        "apex_overlay.first_turn_ack._TARGET_TURN_METHOD."
    )
    assert inspect.iscoroutinefunction(method), (
        "_handle_message_with_agent must be async — the wrapper awaits it."
    )
    params = list(inspect.signature(method).parameters)
    # self, event, source, _quick_key, run_generation
    assert params[:5] == ["self", "event", "source", "_quick_key", "run_generation"], (
        f"_handle_message_with_agent param order changed to {params!r}; the "
        f"overlay wrapper forwards (self, event, source, _quick_key, "
        f"run_generation). The ack uses _quick_key as the cooldown key."
    )


def test_apply_installs_both_halves_and_is_idempotent():
    from gateway import display_config
    from gateway.run import GatewayRunner

    first_turn_ack._APPLIED = False
    assert first_turn_ack.apply() is True
    assert getattr(display_config.resolve_display_setting, first_turn_ack._MARK_RESOLVE, False)
    assert getattr(
        GatewayRunner._handle_message_with_agent, first_turn_ack._MARK_TURN, False
    )

    # Idempotent: no double-wrap.
    resolve_ref = display_config.resolve_display_setting
    turn_ref = GatewayRunner._handle_message_with_agent
    assert first_turn_ack.apply() is True
    assert display_config.resolve_display_setting is resolve_ref
    assert GatewayRunner._handle_message_with_agent is turn_ref


# ---------------------------------------------------------------------------
# Behavior — the resolve default + coercion
# ---------------------------------------------------------------------------

def test_default_true_for_cn_no_edit_platforms():
    for plat in ("weixin", "wecom", "wecom_callback", "dingtalk", "qqbot"):
        assert first_turn_ack.resolve_display_setting({}, plat, "first_turn_ack", False) is True, (
            f"{plat} should default first_turn_ack ON"
        )


def test_default_false_for_feishu_and_international():
    for plat in ("feishu", "telegram", "slack", "discord", "cli"):
        assert first_turn_ack.resolve_display_setting({}, plat, "first_turn_ack", False) is False, (
            f"{plat} should default first_turn_ack OFF"
        )


def test_config_override_wins_both_directions():
    # override OFF on a default-ON platform
    cfg_off = {"display": {"platforms": {"dingtalk": {"first_turn_ack": False}}}}
    assert first_turn_ack.resolve_display_setting(cfg_off, "dingtalk", "first_turn_ack", None) is False

    # override ON on a default-OFF platform
    cfg_on = {"display": {"platforms": {"feishu": {"first_turn_ack": True}}}}
    assert first_turn_ack.resolve_display_setting(cfg_on, "feishu", "first_turn_ack", None) is True

    # YAML-style string values coerce to bool
    cfg_str = {"display": {"platforms": {"telegram": {"first_turn_ack": "yes"}}}}
    assert first_turn_ack.resolve_display_setting(cfg_str, "telegram", "first_turn_ack", None) is True


def test_other_settings_delegate_to_upstream():
    """A non-first_turn_ack setting must pass straight through to upstream resolve.

    We assert the overlay resolve returns exactly what the ORIGINAL upstream
    resolve returns for the same inputs (tool_progress here).
    """
    from gateway import display_config

    # Reach the true upstream even if apply() already wrapped it.
    orig = display_config.resolve_display_setting
    real = getattr(orig, "__wrapped__", orig)

    for plat in ("telegram", "feishu", "dingtalk", "cli"):
        expected = real({}, plat, "tool_progress", "sentinel")
        got = first_turn_ack.resolve_display_setting({}, plat, "tool_progress", "sentinel")
        assert got == expected, (
            f"tool_progress for {plat}: overlay resolve {got!r} != upstream {expected!r}"
        )


def test_wrapped_resolve_delegates_other_settings():
    """The installed wrapper (not just the module fn) delegates other settings."""
    from gateway import display_config

    first_turn_ack._APPLIED = False
    first_turn_ack.apply()

    wrapped = display_config.resolve_display_setting
    real = getattr(wrapped, "__wrapped__")
    # first_turn_ack: wrapper supplies the default
    assert wrapped({}, "dingtalk", "first_turn_ack", False) is True
    # other setting: wrapper == upstream
    assert wrapped({}, "telegram", "show_reasoning", None) == real({}, "telegram", "show_reasoning", None)


# ---------------------------------------------------------------------------
# Behavior — the SEND half fires the ack, debounced, then calls through
# ---------------------------------------------------------------------------

class _FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text):
        self.sent.append((chat_id, text))


class _FakeSource:
    def __init__(self, platform_value, chat_id="chat-1"):
        self.platform = _FakePlatform(platform_value)
        self.chat_id = chat_id


class _FakePlatform:
    def __init__(self, value):
        self.value = value


class _FakeRunner:
    def __init__(self, adapter, platform_value):
        self._plat = _FakePlatform(platform_value)
        self.adapters = {self._plat: adapter}

    # adapters is keyed by the platform object; our source carries a DIFFERENT
    # _FakePlatform instance, so emulate .get by value match.
    def _adapter_for(self, platform):
        return None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_send_ack_fires_for_cn_platform(monkeypatch):
    """_send_first_turn_ack sends one ack for a default-ON platform."""
    adapter = _FakeAdapter()
    plat = _FakePlatform("dingtalk")

    class Runner:
        adapters = {plat: adapter}

    src = type("S", (), {"platform": plat, "chat_id": "c1"})()

    # config load returns {} → tier default applies (dingtalk ON)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    async def _go():
        await first_turn_ack._send_first_turn_ack(Runner, src, "sesskey-1")
        # the send is a fire-and-forget task; let it run
        await asyncio.sleep(0)

    asyncio.new_event_loop().run_until_complete(_go())
    assert adapter.sent, "expected a first-turn ack to be sent for dingtalk"
    assert adapter.sent[0][0] == "c1"
    assert "收到" in adapter.sent[0][1] or "⏳" in adapter.sent[0][1]


def test_send_ack_skipped_for_feishu(monkeypatch):
    adapter = _FakeAdapter()
    plat = _FakePlatform("feishu")

    class Runner:
        adapters = {plat: adapter}

    src = type("S", (), {"platform": plat, "chat_id": "c1"})()
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    async def _go():
        await first_turn_ack._send_first_turn_ack(Runner, src, "sesskey-2")
        await asyncio.sleep(0)

    asyncio.new_event_loop().run_until_complete(_go())
    assert adapter.sent == [], "feishu must not receive a first-turn ack"


def test_send_ack_debounced_within_cooldown(monkeypatch):
    adapter = _FakeAdapter()
    plat = _FakePlatform("wecom")

    class Runner:
        adapters = {plat: adapter}

    src = type("S", (), {"platform": plat, "chat_id": "c1"})()
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    async def _go():
        await first_turn_ack._send_first_turn_ack(Runner, src, "same-key")
        await asyncio.sleep(0)
        await first_turn_ack._send_first_turn_ack(Runner, src, "same-key")
        await asyncio.sleep(0)

    asyncio.new_event_loop().run_until_complete(_go())
    assert len(adapter.sent) == 1, "second ack within cooldown must be suppressed"


def test_send_ack_respects_env_kill_switch(monkeypatch):
    adapter = _FakeAdapter()
    plat = _FakePlatform("dingtalk")

    class Runner:
        adapters = {plat: adapter}

    src = type("S", (), {"platform": plat, "chat_id": "c1"})()
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setenv("HERMES_GATEWAY_FIRST_TURN_ACK_ENABLED", "false")

    async def _go():
        await first_turn_ack._send_first_turn_ack(Runner, src, "k")
        await asyncio.sleep(0)

    asyncio.new_event_loop().run_until_complete(_go())
    assert adapter.sent == [], "kill switch off → no ack"


def test_turn_wrapper_fires_ack_then_delegates(monkeypatch):
    """The wrapped _handle_message_with_agent fires the ack, then calls through."""
    fired = {"ack": 0, "orig": 0}

    async def _orig(self, event, source, _quick_key, run_generation):
        fired["orig"] += 1
        return "orig-result"

    async def _fake_ack(runner, source, session_key):
        fired["ack"] += 1

    monkeypatch.setattr(first_turn_ack, "_send_first_turn_ack", _fake_ack)
    wrapped = first_turn_ack._wrap_turn_method(_orig)

    async def _go():
        return await wrapped("self", "event", "source", "qk", 7)

    result = asyncio.new_event_loop().run_until_complete(_go())
    assert result == "orig-result"
    assert fired["ack"] == 1, "ack must fire on turn start"
    assert fired["orig"] == 1, "original method must still run"


def test_turn_wrapper_never_lets_ack_break_the_turn(monkeypatch):
    """If the ack raises, the original turn method still runs."""
    fired = {"orig": 0}

    async def _orig(self, event, source, _quick_key, run_generation):
        fired["orig"] += 1
        return "ok"

    async def _boom(*a, **k):
        raise RuntimeError("ack exploded")

    monkeypatch.setattr(first_turn_ack, "_send_first_turn_ack", _boom)
    wrapped = first_turn_ack._wrap_turn_method(_orig)

    async def _go():
        return await wrapped("self", "e", "s", "qk", 1)

    result = asyncio.new_event_loop().run_until_complete(_go())
    assert result == "ok"
    assert fired["orig"] == 1


def test_plugin_register_applies_seam():
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_first_turn_ack_plugin_under_test")
    assert "first_turn_ack" in called, (
        "plugin.register() must call first_turn_ack.apply()"
    )
