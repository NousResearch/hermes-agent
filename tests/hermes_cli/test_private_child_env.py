"""Plugin control credentials stay private through actual child-env construction."""
import json
import subprocess
import sys
import threading

import pytest

from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.private_child_env import private_env_keys
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import build_subprocess_env, hermes_subprocess_env, _make_run_env
from hermes_cli.authorized_tool_execution import run_authorized_tool_execution_middleware as run


def context(manager, name):
    return PluginContext(PluginManifest(name=name), manager)


@pytest.fixture
def managers(tmp_path):
    values = [PluginManager(scope_key=str(tmp_path / name)) for name in ("a", "b")]
    yield values
    for manager in values:
        manager.unload()


@pytest.mark.parametrize("names", [[], "TOKEN", [""], ["WITH SPACE"], ["0BAD"], [None], ["A" * 129], ["X"] * 129])
def test_invalid_registration_is_atomic(managers, names):
    token = set_hermes_home_override(managers[0].home_path)
    try:
        before = private_env_keys()
        with pytest.raises(ValueError):
            context(managers[0], "bad").register_private_env_keys(names)
        assert private_env_keys() == before
    finally:
        reset_hermes_home_override(token)


def test_profile_a_b_a_and_overlapping_unload_ownership(managers):
    first = context(managers[0], "one").register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])
    second = context(managers[0], "two").register_private_env_keys(["example_control_token"])
    context(managers[1], "other").register_private_env_keys(["OTHER_CONTROL_TOKEN"])
    for manager, expected in ((managers[0], {"EXAMPLE_CONTROL_TOKEN"}),
                              (managers[1], {"OTHER_CONTROL_TOKEN"}),
                              (managers[0], {"EXAMPLE_CONTROL_TOKEN"})):
        token = set_hermes_home_override(manager.home_path)
        try:
            assert private_env_keys() == expected
        finally:
            reset_hermes_home_override(token)
    token = set_hermes_home_override(managers[0].home_path)
    try:
        first.dispose()
        first.dispose()
        assert private_env_keys() == {"EXAMPLE_CONTROL_TOKEN"}
        managers[0].unload("two")
        assert not private_env_keys() and not second.active
    finally:
        reset_hermes_home_override(token)


def test_private_names_block_inheritance_forced_extras_and_real_child(managers):
    token = set_hermes_home_override(managers[0].home_path)
    try:
        context(managers[0], "control").register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])
        env = build_subprocess_env(
            base={"EXAMPLE_CONTROL_TOKEN": "fake", "example_control_token": "fake", "KEEP_ME": "ok"},
            extra={"_HERMES_FORCE_EXAMPLE_CONTROL_TOKEN": "fake", "_HERMES_FORCE_KEEP_FORCED": "ok"},
        )
        assert not any(k.upper() == "EXAMPLE_CONTROL_TOKEN" for k in env)
        assert env["KEEP_ME"] == env["KEEP_FORCED"] == "ok"
        # No credential value is printed, even if the regression fails.
        code = "import os,json;print(json.dumps(any(k.upper()=='EXAMPLE_CONTROL_TOKEN' for k in os.environ)))"
        actual = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
        assert actual.returncode == 0 and json.loads(actual.stdout) is False
        assert "EXAMPLE_CONTROL_TOKEN" not in hermes_subprocess_env(
            inherit_credentials=True, base_env={"EXAMPLE_CONTROL_TOKEN": "fake"})
        assert "EXAMPLE_CONTROL_TOKEN" not in _make_run_env({"EXAMPLE_CONTROL_TOKEN": "fake"})
    finally:
        reset_hermes_home_override(token)


def test_passthrough_cannot_restore_private_key(managers, monkeypatch):
    token = set_hermes_home_override(managers[0].home_path)
    try:
        context(managers[0], "control").register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])
        monkeypatch.setattr("tools.env_passthrough.scoped_passthrough_additions", lambda env: {
            "example_control_token": "fake", "OTHER": "ok"})
        env = build_subprocess_env(base={})
        assert "example_control_token" not in env and env["OTHER"] == "ok"
    finally:
        reset_hermes_home_override(token)


def test_timed_out_plugin_cannot_register_private_names(managers):
    token = set_hermes_home_override(managers[0].home_path)
    try:
        ctx = context(managers[0], "abandoned")
        ctx._abandon_load()
        assert ctx.register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"]) is None
        assert not private_env_keys()
    finally:
        reset_hermes_home_override(token)


def test_unload_before_dispatch_keeps_key_out_of_actual_child(managers, monkeypatch):
    manager = managers[0]
    token = set_hermes_home_override(manager.home_path)
    try:
        ctx = context(manager, "control")
        ctx.register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])
        def callback(next_call, **kwargs):
            unloader = threading.Thread(target=lambda: manager.unload("control"))
            unloader.start()
            unloader.join(timeout=5)
            assert not unloader.is_alive(), "callback still holds the discovery/registry lock"
            assert not manager._middleware
            return next_call()
        ctx.register_middleware("authorized_tool_execution", callback)
        monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
        def child():
            env = build_subprocess_env(base={"EXAMPLE_CONTROL_TOKEN": "fake"})
            code = "import os,json;print(json.dumps('EXAMPLE_CONTROL_TOKEN' in os.environ))"
            result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
            assert result.returncode == 0
            return json.loads(result.stdout)
        assert run("terminal", {}, child) is False
        assert not private_env_keys(), "completed dispatch retained a stale declaration"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("fails", [False, True])
def test_retained_names_follow_nested_profile_and_restore_on_error(managers, monkeypatch, fails):
    from hermes_constants import hermes_home_key
    by_scope = {manager.scope_key: manager for manager in managers}
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: by_scope[hermes_home_key()])
    for index, manager in enumerate(managers):
        ctx = context(manager, "control")
        ctx.register_private_env_keys([f"PRIVATE_{index}"])
        def callback(next_call, manager=manager, **kwargs):
            manager.unload("control")
            return next_call()
        ctx.register_middleware("authorized_tool_execution", callback)
    token = set_hermes_home_override(managers[0].home_path)
    try:
        def outer():
            assert private_env_keys() == {"PRIVATE_0"}
            nested = set_hermes_home_override(managers[1].home_path)
            try:
                assert run("terminal", {}, private_env_keys) == {"PRIVATE_1"}
                assert not private_env_keys()
            finally:
                reset_hermes_home_override(nested)
            assert private_env_keys() == {"PRIVATE_0"}
            if fails:
                raise ValueError("tool failed")
            return "ok"
        if fails:
            with pytest.raises(ValueError, match="tool failed"):
                run("terminal", {}, outer)
        else:
            assert run("terminal", {}, outer) == "ok"
        assert not private_env_keys()
    finally:
        reset_hermes_home_override(token)


def test_callback_key_snapshot_excludes_concurrent_unload(managers, monkeypatch):
    import hermes_constants
    manager = managers[0]
    ctx = context(manager, "control")
    ctx.register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])
    entered, proceed, unloaded = threading.Event(), threading.Event(), threading.Event()
    def callback(next_call, **kwargs):
        assert unloaded.wait(5), "snapshot lock was held during callback execution"
        return next_call()
    ctx.register_middleware("authorized_tool_execution", callback)
    original = hermes_constants.hermes_home_key
    def home_key(value=None):
        if value == manager.scope_key and threading.current_thread().name == "snapshot-caller":
            entered.set()
            assert proceed.wait(5)
        return original(value)
    monkeypatch.setattr(hermes_constants, "hermes_home_key", home_key)
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    results = []
    def caller():
        token = set_hermes_home_override(manager.home_path)
        try:
            results.append(run("terminal", {}, private_env_keys))
        finally:
            reset_hermes_home_override(token)
    def unload():
        manager.unload("control")
        unloaded.set()
    thread = threading.Thread(target=caller, name="snapshot-caller")
    unloader = threading.Thread(target=unload)
    thread.start()
    try:
        assert entered.wait(5)
        unloader.start()
        assert not unloaded.wait(.05), "unload bypassed the coherent callback/key snapshot"
    finally:
        proceed.set()
        thread.join(timeout=10)
        if unloader.ident is not None:
            unloader.join(timeout=10)
    assert not thread.is_alive() and not unloader.is_alive()
    assert results == [{"EXAMPLE_CONTROL_TOKEN"}]
