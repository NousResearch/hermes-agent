"""Registered control names stay out of children across profile switches."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from hermes_cli.authorized_tool_execution import run_authorized_tool_execution_middleware as run
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.private_child_env import private_env_keys
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import build_subprocess_env, hermes_subprocess_env, _make_run_env


KEY_A = "EXAMPLE_PROFILE_ALPHA_PRIVATE"
KEY_B = "EXAMPLE_PROFILE_BETA_PRIVATE"
BUILDERS = ("factory", "foreground", "credentialed")


@contextmanager
def profile(manager):
    token = set_hermes_home_override(manager.home_path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    values = [PluginManager(scope_key=str(tmp_path / name)) for name in ("a", "b")]
    yield values
    for manager in values:
        manager.unload()


def register(manager, key):
    ctx = PluginContext(PluginManifest(name="control"), manager)
    ctx.register_private_env_keys([key])
    return ctx


def child_env(builder, extra=None):
    if builder == "factory":
        return build_subprocess_env(extra=extra)
    if builder == "foreground":
        return _make_run_env(extra or {})
    return hermes_subprocess_env(inherit_credentials=True, base_env=extra)


def contains_private(env):
    return any(key.upper().removeprefix("_HERMES_FORCE_") in {KEY_A, KEY_B} for key in env)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("b_declares", [False, True])
def test_operator_inherited_keys_are_stripped_across_a_b_a(profiles, monkeypatch, builder, b_declares):
    register(profiles[0], KEY_A)
    if b_declares:
        register(profiles[1], KEY_B)
        monkeypatch.setenv(KEY_B, "fake-beta")
    monkeypatch.setenv(KEY_A, "fake-alpha")
    monkeypatch.setenv("EXAMPLE_PUBLIC_SETTING", "keep")
    for index in (0, 1, 0):
        with profile(profiles[index]):
            expected = {KEY_A} if index == 0 else ({KEY_B} if b_declares else set())
            assert private_env_keys() == expected  # Authority never crosses profiles.
            env = child_env(builder)
            assert not contains_private(env)
            assert env["EXAMPLE_PUBLIC_SETTING"] == "keep"


@pytest.mark.parametrize("builder", BUILDERS)
def test_other_profile_private_extras_and_passthrough_cannot_restore_key(profiles, monkeypatch, builder):
    register(profiles[0], KEY_A)
    monkeypatch.setattr("tools.env_passthrough.scoped_passthrough_additions", lambda env: {
        KEY_A.lower(): "fake-passthrough", "EXAMPLE_PUBLIC_SETTING": "keep"})
    with profile(profiles[1]):
        env = child_env(builder, {KEY_A.lower(): "fake-extra", "_HERMES_FORCE_" + KEY_A: "fake-force"})
        assert not contains_private(env)
        assert not private_env_keys()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("builder", BUILDERS)
def test_windows_mixed_case_forced_private_key_is_not_in_child(profiles, builder):
    register(profiles[0], KEY_A)
    with profile(profiles[1]):
        env = child_env(builder, {
            KEY_A.lower(): "fake-direct",
            "_hErMeS_fOrCe_" + KEY_A.lower(): "fake-mixed-force",
            "_HERMES_FORCE_" + KEY_A.lower(): "fake-force",
            "_HERMES_FORCE_EXAMPLE_PUBLIC_SETTING": "keep",
        })
        assert not contains_private(env)
        if builder != "credentialed":
            assert env["EXAMPLE_PUBLIC_SETTING"] == "keep"


@pytest.mark.parametrize("callback_profile", [0, 1])
@pytest.mark.parametrize("builder", ["factory", "foreground"])
def test_captured_callback_keeps_other_profile_key_private_on_another_thread(
    profiles, monkeypatch, callback_profile, builder,
):
    owner_ctx = register(profiles[0], KEY_A)
    manager = profiles[callback_profile]
    ctx = owner_ctx if callback_profile == 0 else PluginContext(PluginManifest(name="worker"), manager)
    monkeypatch.setenv(KEY_A, "fake-inherited")
    observed = []
    errors = []

    def worker():
        try:
            with profile(profiles[1]):
                observed.append((bool(private_env_keys()), contains_private(child_env(builder))))
        except BaseException as exc:
            errors.append(type(exc).__name__)

    def callback(next_call, **kwargs):
        profiles[0].unload("control")
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
        return next_call()

    ctx.register_middleware("authorized_tool_execution", callback)
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    with profile(manager):
        assert run("terminal", {}, lambda: "done") == "done"
    assert observed == [(False, False)]
    # No permanent global blocklist: snapshots release after the captured call.
    with profile(profiles[1]):
        assert not private_env_keys()
        assert KEY_A in build_subprocess_env()


def test_captured_cross_profile_strip_snapshot_releases_on_exception(profiles, monkeypatch):
    register(profiles[0], KEY_A)
    manager = profiles[1]
    ctx = PluginContext(PluginManifest(name="worker"), manager)
    monkeypatch.setenv(KEY_A, "fake-inherited")

    def callback(next_call, **kwargs):
        profiles[0].unload("control")
        assert not contains_private(build_subprocess_env())
        raise ValueError("fake callback failure")

    ctx.register_middleware("authorized_tool_execution", callback)
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    with profile(manager):
        with pytest.raises(ValueError, match="fake callback failure"):
            run("terminal", {}, lambda: None)
        assert KEY_A in build_subprocess_env()


def test_other_profile_private_key_absent_in_real_child(profiles, monkeypatch):
    register(profiles[0], KEY_A)
    monkeypatch.setenv(KEY_A, "fake-inherited")
    with profile(profiles[1]):
        env = build_subprocess_env()
        # Only a boolean is printed, including when this regression fails.
        code = "import os,json;print(json.dumps('EXAMPLE_PROFILE_ALPHA_PRIVATE' in os.environ))"
        actual = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=10)
        assert actual.returncode == 0
        assert json.loads(actual.stdout) is False


def test_explicit_unscrubbed_factory_contract_is_unchanged(profiles):
    register(profiles[0], KEY_A)
    with profile(profiles[1]):
        env = build_subprocess_env(base={KEY_A: "fake"}, scrub_secrets=False,
                                   inherit_profile_home=False)
        assert env == {KEY_A: "fake"}


def test_one_child_build_keeps_private_snapshot_through_forced_extras(profiles, monkeypatch):
    from tools.environments import local
    register(profiles[0], KEY_A)
    original = local._filter_secret_env

    def filter_then_unload(*args, **kwargs):
        original(*args, **kwargs)
        profiles[0].unload("control")

    monkeypatch.setattr(local, "_filter_secret_env", filter_then_unload)
    with profile(profiles[1]):
        env = build_subprocess_env(base={}, extra={"_HERMES_FORCE_" + KEY_A: "fake-extra"})
        assert not contains_private(env)


def test_outer_policy_lookup_cannot_unprotect_private_force_extra(profiles, monkeypatch):
    from tools.environments import local
    register(profiles[0], KEY_A)
    original = local._plugin_terminal_env_strip_keys

    def policy_then_unload():
        names = original()
        profiles[0].unload("control")
        return names

    monkeypatch.setattr(local, "_plugin_terminal_env_strip_keys", policy_then_unload)
    with profile(profiles[1]):
        env = build_subprocess_env(base={}, extra={"_HERMES_FORCE_" + KEY_A: "fake-extra"})
        assert not contains_private(env)
