"""Tests for Hindsight embedded profile environment preservation, atomic writes,
managed key life-cycle, and offline environment export."""

import os
from pathlib import Path
from plugins.memory.hindsight.embedded import (
    _HERMES_MANAGED_KEYS,
    _build_embedded_profile_env,
    _compute_target_env,
    _export_daemon_offline_env,
    _materialize_embedded_profile_env,
    _parse_bool_setting,
    _sanitize_env_pair,
)


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

    # Now user deleted embeddings_local_force_cpu and hf_hub_offline in config
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
    assert "HF_HUB_OFFLINE" not in target_env

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
    from plugins.memory.hindsight.embedded import _load_simple_env
    on_disk = _load_simple_env(profile_env)
    target_env_2 = _compute_target_env(profile_env, config, llm_api_key="test-key")
    assert on_disk == target_env_2
