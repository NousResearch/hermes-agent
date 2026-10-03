"""A ``type: exec`` quick command for a routed profile runs in that profile's env, not the launch one's.

A multiplexed gateway handles a routed profile's message inside ``_async_profile_runtime_scope``
(``_make_profile_message_handler``), but the child env was built from the launch profile's
``os.environ``, so the snippet saw the launch profile's ``.env`` values and none of its own.
"""

import pytest

from agent.secret_scope import set_multiplex_active

_PROBE = "echo A=${A_MARKER:-unset} B=${B_MARKER:-unset} TENV=${TERMINAL_ENV:-unset}"


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_routed_profile_quick_exec_sees_its_own_env(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner, _async_profile_runtime_scope

    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\nTERMINAL_ENV=docker\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("A_MARKER", "a")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.delenv("B_MARKER", raising=False)
    set_multiplex_active(True)
    try:
        async with _async_profile_runtime_scope(b):
            routed = await GatewayRunner._hm_run_exec_quick_command(None, "probe", _PROBE)
        async with _async_profile_runtime_scope(a):
            launch = await GatewayRunner._hm_run_exec_quick_command(None, "probe", _PROBE)
    finally:
        set_multiplex_active(False)
    assert routed == "A=unset B=b TENV=unset"
    assert launch == "A=a B=unset TENV=docker"
