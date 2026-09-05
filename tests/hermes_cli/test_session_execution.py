"""Public registration contract, using real imports and isolated HERMES_HOME."""
import importlib

import pytest


def test_immutable_context_aliases_revocation_and_conflicting_owners(tmp_path, monkeypatch):
    api = importlib.import_module("hermes_cli.session_execution")
    values = {"DISPLAY": ":42"}
    context = api.SessionExecutionContext(env_set=values, env_unset={"HOST_SOCKET"})
    api.register_session_execution_context("a", context, task_ids=("task-a",))
    try:
        values["DISPLAY"] = ":0"
        assert api.get_session_execution_context(task_id="task-a") is context
        assert api.get_session_execution_context(session_id="other") is None
        assert context.apply_env({"HOST_SOCKET": "host", "DISPLAY": ":0"}) == {"DISPLAY": ":42"}
        with pytest.raises(TypeError):
            context.env_set["DISPLAY"] = ":0"
        lease = api.resolve_session_execution_context(task_id="task-a")
        api.register_session_execution_context("b", api.SessionExecutionContext())
        with pytest.raises(api.SessionExecutionError, match="conflicting"):
            api.get_session_execution_context(session_id="b", task_id="task-a")
        with pytest.raises(api.SessionExecutionError, match="owned"):
            api.register_session_execution_context("b", context, task_ids=("task-a",))
        with monkeypatch.context() as scoped:
            scoped.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
            assert api.get_session_execution_context(session_id="a") is None
        assert api.remove_session_execution_context("a")
        with pytest.raises(api.SessionExecutionError, match="revoked"):
            lease.apply_env({})
        assert not api.remove_session_execution_context("a")
    finally:
        api.remove_session_execution_context("a")
        api.remove_session_execution_context("b")


@pytest.mark.linux_only
def test_desktop_context_validates_private_endpoint_and_rejects_dead_context(tmp_path, monkeypatch, request):
    import socket
    import tempfile
    from pathlib import Path
    api = importlib.import_module("hermes_cli.session_execution")
    temporary = tempfile.TemporaryDirectory(prefix="hse-")
    request.addfinalizer(temporary.cleanup)
    runtime = Path(temporary.name)
    endpoint = runtime / "wayland-test"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(endpoint))
    try:
        launch = api.ComputerUseLaunchContext(driver_command="/bin/true", private_daemon=True,
                                             desktop_only=True, no_overlay=False,
                                             session_name="Test desktop", theme="blue")
        context = api.SessionExecutionContext(
            env_set={"XDG_RUNTIME_DIR": str(runtime), "WAYLAND_DISPLAY": endpoint.name, "CUA_DRIVER_RS_ENABLE_WAYLAND": "1"},
            computer_use=launch)
        api.register_session_execution_context("desktop", context)
        lease = api.resolve_session_execution_context(session_id="desktop")
        lease.check()
        with monkeypatch.context() as scoped:
            scoped.setenv("XDG_RUNTIME_DIR", str(runtime))
            scoped.setenv("WAYLAND_DISPLAY", endpoint.name)
            with pytest.raises(api.SessionExecutionError, match="host"):
                api.register_session_execution_context("host", context)
        endpoint.unlink()
        with pytest.raises(api.SessionExecutionError, match="desktop"):
            lease.check()
        with pytest.raises(ValueError, match="private"):
            api.ComputerUseLaunchContext(desktop_only=True)
        for kwargs in ({"env_set": {"A": "x"}, "env_unset": {"A"}},
                       {"env_set": {"BAD=KEY": "x"}}, {"env_set": {"A": "a\u0000b"}}):
            with pytest.raises(ValueError):
                api.SessionExecutionContext(**kwargs)
        for validate in (lambda: False, lambda: 1):
            with pytest.raises(api.SessionExecutionError, match="validation"):
                api.register_session_execution_context("bad", api.SessionExecutionContext(validate=validate))
        env = {"XDG_RUNTIME_DIR": str(runtime), "WAYLAND_DISPLAY": "wl-native"}
        native_sock = socket.socket(socket.AF_UNIX)
        native_sock.bind(str(runtime / "wl-native"))
        try:
            with pytest.raises(api.SessionExecutionError, match="native Wayland"):
                api.register_session_execution_context("unsafe", api.SessionExecutionContext(env_set=env, computer_use=launch))
            env["CUA_DRIVER_RS_ENABLE_WAYLAND"] = "1"
            with monkeypatch.context() as scoped:
                scoped.setenv("DISPLAY", ":host")
                with pytest.raises(api.SessionExecutionError, match="host.*DISPLAY"):
                    api.register_session_execution_context("unsafe", api.SessionExecutionContext(env_set=env, computer_use=launch))
        finally:
            native_sock.close()
    finally:
        sock.close()
        api.remove_session_execution_context("desktop")


@pytest.mark.linux_only
def test_prefix_and_runtime_validation_fail_closed(tmp_path):
    import sys
    from hermes_cli import session_execution as api
    for invalid in ('/bin/echo', ('relative',), ('/bin/echo', '\0')):
        with pytest.raises(ValueError):
            api.SessionExecutionContext(command_prefix=invalid)
        with pytest.raises(ValueError):
            api.ComputerUseLaunchContext(command_prefix=invalid)
    prefix = [sys.executable, '-c', 'pass']
    context = api.SessionExecutionContext(command_prefix=prefix)
    prefix.append('changed')
    assert context.command_prefix == (sys.executable, '-c', 'pass')
    with pytest.raises(api.SessionExecutionError, match='executable unavailable'):
        api.register_session_execution_context('missing', api.SessionExecutionContext(command_prefix=(str(tmp_path / 'missing'),)))
    runtime = tmp_path / 'runtime'
    runtime.mkdir(mode=0o700)
    context = api.SessionExecutionContext(computer_use=api.ComputerUseLaunchContext(private_daemon=True, runtime_dir=str(runtime)))
    api.register_session_execution_context('runtime', context)
    try:
        runtime.chmod(0o755)
        with pytest.raises(api.SessionExecutionError, match='private directory'):
            api.get_session_execution_context(session_id='runtime')
        runtime.chmod(0o700)
        runtime.rename(tmp_path / 'old-runtime')
        runtime.mkdir(mode=0o700)
        with pytest.raises(api.SessionExecutionError, match='replaced'):
            api.get_session_execution_context(session_id='runtime')
    finally:
        api.remove_session_execution_context('runtime')
