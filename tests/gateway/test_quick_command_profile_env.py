"""A routed profile's quick command is its own, and its ``type: exec`` snippet runs in its own env.

A multiplexed gateway handles a routed profile's message inside ``_async_profile_runtime_scope``
(``_make_profile_message_handler``), but the child env was built from the launch profile's
``os.environ``, so the snippet saw the launch profile's ``.env`` values and none of its own
(#131549). The lookup read ``self.config`` alone (the launch profile's), so a routed profile's own
``quick_commands`` answered "Unknown command" (#132517).
"""

import pytest

from agent.secret_scope import set_multiplex_active

_PROBE = "echo who={} A=${{A_MARKER:-unset}} B=${{B_MARKER:-unset}} TENV=${{TERMINAL_ENV:-unset}}"


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_routed_profile_quick_command_is_its_own_and_sees_its_own_env(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _async_profile_runtime_scope
    from gateway.session import SessionSource

    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\nTERMINAL_ENV=docker\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("A_MARKER", "a")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.delenv("B_MARKER", raising=False)

    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner.config = GatewayConfig(multiplex_profiles=True, quick_commands={
        "probe": {"type": "exec", "command": _PROBE.format("a")}})
    runner._profile_configs = {"b": GatewayConfig(quick_commands={
        "deep": {"type": "alias", "target": "/probe"},
        "probe": {"type": "exec", "command": _PROBE.format("b")}})}

    async def run(text, profile):
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1",
                               chat_type="dm", profile=profile)
        event = MessageEvent(text=text, source=source, message_id="m1")
        _, _, command, _ = await runner._hm_resolve_command(event, source, "k")
        return (await runner._hm_dispatch_quick_and_plugin_commands(event, source, command))[1]

    set_multiplex_active(True)
    try:
        async with _async_profile_runtime_scope(b):
            routed = await run("/deep", "b")
        async with _async_profile_runtime_scope(a):
            launch = await run("/probe", None)
    finally:
        set_multiplex_active(False)
    assert routed == "who=b A=unset B=b TENV=unset"
    assert launch == "who=a A=a B=unset TENV=docker"
