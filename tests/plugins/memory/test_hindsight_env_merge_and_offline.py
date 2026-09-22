"""Tests for Hindsight embedded profile environment preservation, atomic writes,
managed key life-cycle, and offline environment export."""

import errno
import json
import os
import stat
from pathlib import Path

import pytest

from plugins.memory.hindsight import embedded
from plugins.memory.hindsight.embedded import (
    _build_embedded_profile_env,
    _compute_target_env,
    _embedded_profile_env_path,
    _export_daemon_offline_env,
    _load_simple_env,
    _materialize_embedded_profile_env,
    _parse_bool_setting,
    _sanitize_env_pair,
)


@pytest.fixture(autouse=True)
def _isolate_exports(monkeypatch):
    # Restore process env as well as ownership between tests.
    monkeypatch.setattr(embedded, "_DAEMON_OFFLINE_ENV_ORIGINALS", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def test_sanitize_env_pair():
    assert _sanitize_env_pair("VALID_KEY", "valid_val") == ("VALID_KEY", "valid_val")
    assert _sanitize_env_pair("INVALID-KEY", "val") is None
    assert _sanitize_env_pair("123_BAD", "val") is None
    assert _sanitize_env_pair("KEY", "val\r\nwith\ninjection") == ("KEY", "valwithinjection")


def test_parse_bool_setting():
    assert _parse_bool_setting(True) is True
    assert _parse_bool_setting("true") is True
    assert _parse_bool_setting("1") is True
    assert _parse_bool_setting(False) is False
    assert _parse_bool_setting("false") is False
    assert _parse_bool_setting("0") is False


def test_export_daemon_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    _export_daemon_offline_env({"hf_hub_offline": True, "hf_endpoint": "https://hf-mirror.com"})
    assert os.environ.get("HF_HUB_OFFLINE") == "true"
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"

    # Explicit false can turn it off
    _export_daemon_offline_env({"hf_hub_offline": False})
    assert os.environ.get("HF_HUB_OFFLINE") == "false"


def test_managed_keys_tombstone_and_unmanaged_preservation(tmp_path, monkeypatch):
    profile_env = tmp_path / "hermes.env"
    # Existing file with managed keys and custom unmanaged keys
    profile_env.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o\n"
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=true\n"
        "HF_HUB_OFFLINE=true\n"
        "CUSTOM_USER_PROXY=http://127.0.0.1:7890\n"
    )

    # Static keys are owned even without a sidecar; legacy HF keys are not.
    config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "profile": "hermes",
    }
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    target_env = _compute_target_env(profile_env, config, llm_api_key="test-key")

    # Unmanaged key must be preserved
    assert target_env.get("CUSTOM_USER_PROXY") == "http://127.0.0.1:7890"

    # Deleted managed keys must NOT resurrect
    assert "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU" not in target_env
    assert target_env["HF_HUB_OFFLINE"] == "true"

    # Active managed keys updated
    assert target_env.get("HINDSIGHT_API_LLM_MODEL") == "gpt-4o-mini"
    assert target_env.get("HINDSIGHT_API_LLM_API_KEY") == "test-key"


def test_idempotent_target_env_avoids_restart_loop(tmp_path):
    profile_env = tmp_path / "hermes.env"
    config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "profile": "hermes",
        "extra_env": {"MY_VAR": "val1"},
    }
    target_env_1 = _compute_target_env(profile_env, config, llm_api_key="test-key")
    profile_env.write_text("".join(f"{k}={v}\n" for k, v in target_env_1.items()))

    # Second pass: target env must exactly match on-disk env
    on_disk = _load_simple_env(profile_env)
    target_env_2 = _compute_target_env(profile_env, config, llm_api_key="test-key")
    assert on_disk == target_env_2


@pytest.mark.parametrize("failure", ["mkstemp", "chmod", "fdopen", "write", "fsync", "replace"])
def test_failed_write_preserves_original_and_closes_descriptor(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {"profile": "atomic"}
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True)
    original = b"# ORIGINAL\r\nHINDSIGHT_API_LLM_API_KEY=original\r\n"
    profile_env.write_bytes(original)
    opened = []
    mkstemp = embedded.tempfile.mkstemp
    fdopen = os.fdopen

    def capture_fd(*args, **kwargs):
        result = mkstemp(*args, **kwargs)
        opened.append(result[0])
        return result

    def fail(*args, **kwargs):
        raise OSError("injected pre-replace failure")

    def fail_write(fd, *args, **kwargs):
        # A real file owns the descriptor; writing fails inside its context.
        fh = fdopen(fd, *args, **kwargs)
        fh.write = fail
        return fh

    monkeypatch.setattr(embedded.tempfile, "mkstemp", capture_fd)
    if failure == "mkstemp":
        monkeypatch.setattr(embedded.tempfile, "mkstemp", fail)
    else:
        monkeypatch.setattr(embedded.os, "fdopen" if failure == "write" else failure,
                            fail_write if failure == "write" else fail)

    with pytest.raises(OSError, match="injected"):
        _materialize_embedded_profile_env(config, llm_api_key="replacement")

    assert profile_env.read_bytes() == original
    assert list(profile_env.parent.iterdir()) == [profile_env]
    for fd in opened:
        with pytest.raises(OSError) as exc:
            os.fstat(fd)
        assert exc.value.errno == errno.EBADF


@pytest.mark.parametrize("extra_setting", ["extra_env", "env_extra"])
def test_dynamic_keys_removed_but_unmanaged_keys_survive(tmp_path, monkeypatch, extra_setting):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {
        "profile": "ownership",
        "hf_hub_offline": True,
        "hf_endpoint": "https://mirror.example",
        extra_setting: {"CUSTOM_SETTING": "configured", "MANUAL_OVERRIDE": "new"},
    }
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text("MANUAL_OVERRIDE=old\nUSER_MANUAL=keep\nHINDSIGHT_API_PORT=9000\n")
    _materialize_embedded_profile_env(config, llm_api_key="test-secret")
    assert _load_simple_env(profile_env)["CUSTOM_SETTING"] == "configured"
    assert _load_simple_env(profile_env)["MANUAL_OVERRIDE"] == "new"

    sidecar = embedded._profile_env_ownership_path(profile_env)
    owned = json.loads(sidecar.read_text())
    assert set(owned) == {"CUSTOM_SETTING", "MANUAL_OVERRIDE", "HF_HUB_OFFLINE", "HF_ENDPOINT"}
    if os.name == "posix":
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600

    # Simulate the upstream register step rewriting .env and adding another key.
    with profile_env.open("a") as fh:
        fh.write("HINDSIGHT_API_HOST=127.0.0.1\n")
    config = {"profile": "ownership"}
    target = _compute_target_env(profile_env, config, llm_api_key="test-secret")
    assert not set(owned) & target.keys()
    _materialize_embedded_profile_env(config, llm_api_key="test-secret")
    assert _load_simple_env(profile_env) == target
    assert target["USER_MANUAL"] == "keep"
    assert target["HINDSIGHT_API_PORT"] == "9000"
    assert target["HINDSIGHT_API_HOST"] == "127.0.0.1"
    assert json.loads(sidecar.read_text()) == []


@pytest.mark.parametrize("original", [None, "", "pre-plugin-value"])
def test_exports_restore_pre_plugin_values_and_false_updates_target(tmp_path, monkeypatch, original):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if original is not None:
        monkeypatch.setenv("HF_HUB_OFFLINE", original)
        monkeypatch.setenv("HF_ENDPOINT", original)

    config = {"hf_hub_offline": True, "hf_endpoint": "https://mirror.example"}
    _export_daemon_offline_env(config)
    path = _materialize_embedded_profile_env(config, llm_api_key="test-key")
    assert os.environ["HF_HUB_OFFLINE"] == _load_simple_env(path)["HF_HUB_OFFLINE"] == "true"
    config["hf_hub_offline"] = False
    _export_daemon_offline_env(config)
    _materialize_embedded_profile_env(config, llm_api_key="test-key")
    assert os.environ["HF_HUB_OFFLINE"] == _load_simple_env(path)["HF_HUB_OFFLINE"] == "false"
    assert os.environ["HF_ENDPOINT"] == "https://mirror.example"

    _export_daemon_offline_env({})
    _materialize_embedded_profile_env({}, llm_api_key="test-key")
    for key in ("HF_HUB_OFFLINE", "HF_ENDPOINT"):
        assert os.environ.get(key) == original
        assert key not in _load_simple_env(path)
        assert key not in _build_embedded_profile_env({}, llm_api_key="test-key")


def test_first_upgrade_preserves_disk_hf_without_importing_ambient_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "false")
    monkeypatch.setenv("HF_ENDPOINT", "https://ambient.example")
    profile_env = _embedded_profile_env_path({})
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text("HF_HUB_OFFLINE=true\nHINDSIGHT_API_PORT=9000\n")
    for _ in range(2):
        _materialize_embedded_profile_env({}, llm_api_key="test-key")
        target = _load_simple_env(profile_env)
        assert target["HF_HUB_OFFLINE"] == "true"
        assert target["HINDSIGHT_API_PORT"] == "9000"
        assert "HF_ENDPOINT" not in target


@pytest.mark.parametrize("sidecar_content", [b"broken json", b'{}', b'[1]', b'\xff'])
def test_invalid_ownership_preserves_unknown_keys(tmp_path, monkeypatch, sidecar_content):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = _materialize_embedded_profile_env({"extra_env": {"CUSTOM": "keep"}}, llm_api_key="test-key")
    embedded._profile_env_ownership_path(path).write_bytes(sidecar_content)
    _materialize_embedded_profile_env({}, llm_api_key="test-key")
    assert _load_simple_env(path)["CUSTOM"] == "keep"


def test_ownership_io_failure_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = _embedded_profile_env_path({})
    sidecar = embedded._profile_env_ownership_path(path)
    sidecar.mkdir(parents=True)  # Cannot read or replace a directory as a sidecar.
    path.write_text("USER_KEY=keep\n")
    _materialize_embedded_profile_env({"extra_env": {"CUSTOM": "new"}}, llm_api_key="test-key")
    assert _load_simple_env(path)["USER_KEY"] == "keep"
    assert _load_simple_env(path)["CUSTOM"] == "new"
    assert set(path.parent.iterdir()) == {path, sidecar}


@pytest.mark.parametrize("b_settings", [{}, {"hf_hub_offline": False, "hf_endpoint": "https://b.example"}])
def test_profile_switches_use_own_config_and_ownership(tmp_path, monkeypatch, b_settings):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory.hindsight import _load_config

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("HF_ENDPOINT", "https://original.example")
    configs = {
        "a": {"profile": "a", "hf_hub_offline": True, "hf_endpoint": "https://a.example",
              "extra_env": {"A_ONLY": "owned"}},
        "b": {"profile": "b", **b_settings},
    }
    for name, config in configs.items():
        path = tmp_path / name / "hindsight" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(config))

    snapshots = {}
    for name in ("a", "b", "a"):
        home_token = set_hermes_home_override(tmp_path / name)
        secret_token = secret_scope.set_secret_scope({"HINDSIGHT_API_LLM_API_KEY": f"key-{name}"})
        try:
            config = _load_config()
            # Build while the previous profile's exports still occupy os.environ.
            built = _build_embedded_profile_env(config)
            _export_daemon_offline_env(config)
            path = _materialize_embedded_profile_env(config)
            actual = _load_simple_env(path)
            assert actual == built
            assert actual["HINDSIGHT_API_LLM_API_KEY"] == f"key-{name}"
            assert os.environ.get("HF_HUB_OFFLINE") == actual.get("HF_HUB_OFFLINE")
            assert os.environ["HF_ENDPOINT"] == actual.get("HF_ENDPOINT", "https://original.example")
            assert ("A_ONLY" in actual) == (name == "a")
            if name in snapshots:
                assert actual == snapshots[name]
            snapshots[name] = actual
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    _export_daemon_offline_env({})
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["HF_ENDPOINT"] == "https://original.example"
