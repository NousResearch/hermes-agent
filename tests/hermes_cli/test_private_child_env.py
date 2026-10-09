"""Plugin control credentials stay private through actual child-env construction."""
import json
import subprocess
import sys

import pytest

from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.private_child_env import private_env_keys
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import build_subprocess_env, hermes_subprocess_env, _make_run_env


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
