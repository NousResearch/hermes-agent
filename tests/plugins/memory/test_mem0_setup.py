"""Tests for Mem0 setup wizard — flag parsing, config building, validation."""

import json
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from plugins.memory.mem0._setup import (
    parse_flags,
    build_oss_config,
    _write_env,
    post_setup,
    _check_qdrant_path,
    _check_ollama,
    _check_pgvector,
)


def _inject_fake_hermes_cli(monkeypatch):
    """Inject fake hermes_cli modules so yaml/curses aren't required."""
    fake_config_mod = types.ModuleType("hermes_cli.config")
    fake_config_mod.save_config = lambda c: None

    fake_setup_mod = types.ModuleType("hermes_cli.memory_setup")
    fake_setup_mod._curses_select = lambda *a, **kw: 0
    fake_setup_mod._prompt = lambda label, default=None, secret=False: default or ""

    fake_hermes_cli = types.ModuleType("hermes_cli")
    fake_hermes_cli.config = fake_config_mod
    fake_hermes_cli.memory_setup = fake_setup_mod

    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config_mod)
    monkeypatch.setitem(sys.modules, "hermes_cli.memory_setup", fake_setup_mod)

    monkeypatch.setattr("plugins.memory.mem0._setup._curses_select", lambda *a, **kw: 0)
    monkeypatch.setattr("plugins.memory.mem0._setup._prompt", lambda label, default=None, secret=False: default or "")
    return fake_config_mod


class TestParseFlags:

    def test_mode_platform(self):
        flags = parse_flags(["--mode", "platform", "--api-key", "sk-test"])
        assert flags["mode"] == "platform"
        assert flags["api_key"] == "sk-test"

    def test_mode_oss_defaults(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai"])
        assert flags["mode"] == "oss"
        assert flags["oss_llm"] == "openai"
        assert flags["oss_embedder"] == "openai"
        assert flags["oss_vector"] == "qdrant"

    def test_mode_oss_all_flags(self):
        flags = parse_flags([
            "--mode", "oss",
            "--oss-llm", "ollama",
            "--oss-llm-model", "llama3:latest",
            "--oss-embedder", "ollama",
            "--oss-embedder-model", "nomic-embed-text",
            "--oss-vector", "pgvector",
            "--oss-vector-host", "db.local",
            "--oss-vector-port", "5433",
            "--oss-vector-user", "pguser",
            "--oss-vector-password", "secret",
            "--oss-vector-dbname", "memdb",
            "--user-id", "my-user",
        ])
        assert flags["oss_llm"] == "ollama"
        assert flags["oss_llm_model"] == "llama3:latest"
        assert flags["oss_vector"] == "pgvector"
        assert flags["oss_vector_user"] == "pguser"
        assert flags["user_id"] == "my-user"

    def test_no_flags_returns_empty_mode(self):
        flags = parse_flags([])
        assert flags["mode"] == ""

    def test_oss_vector_path_flag(self):
        flags = parse_flags(["--mode", "oss", "--oss-vector-path", "/data/qdrant"])
        assert flags["oss_vector_path"] == "/data/qdrant"


class TestBuildOSSConfig:

    def test_openai_defaults(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai"])
        oss, env_writes = build_oss_config(flags)
        assert oss["llm"]["provider"] == "openai"
        assert oss["llm"]["config"]["model"] == "gpt-5-mini"
        assert oss["embedder"]["provider"] == "openai"
        assert oss["embedder"]["config"]["model"] == "text-embedding-3-small"
        assert oss["vector_store"]["provider"] == "qdrant"
        assert env_writes["OPENAI_API_KEY"] == "sk-oai"

    def test_openai_custom_urls_use_mem0_provider_specific_keys(self):
        flags = parse_flags([
            "--mode", "oss",
            "--oss-llm-key", "sk-oai",
            "--oss-llm-url", "https://llm.example/v1",
            "--oss-embedder-url", "https://embed.example/v1",
        ])

        oss, _ = build_oss_config(flags)

        assert oss["llm"]["config"]["openai_base_url"] == "https://llm.example/v1"
        assert oss["embedder"]["config"]["openai_base_url"] == "https://embed.example/v1"
        assert "api_base" not in oss["llm"]["config"]
        assert "api_base" not in oss["embedder"]["config"]

    def test_ollama_no_key_needed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm", "ollama", "--oss-embedder", "ollama"])
        oss, env_writes = build_oss_config(flags)
        assert oss["llm"]["provider"] == "ollama"
        assert "model" in oss["llm"]["config"]
        assert oss["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert oss["embedder"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert env_writes == {}

    def test_embedder_reuses_llm_key(self):
        """When LLM and embedder share same provider, key written once."""
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai"])
        _, env_writes = build_oss_config(flags)
        assert env_writes == {"OPENAI_API_KEY": "sk-oai"}

    def test_different_embedder_needs_separate_key(self):
        flags = parse_flags([
            "--mode", "oss",
            "--oss-llm", "ollama",
            "--oss-embedder", "openai", "--oss-embedder-key", "sk-oai",
        ])
        _, env_writes = build_oss_config(flags)
        assert env_writes == {"OPENAI_API_KEY": "sk-oai"}

    def test_pgvector_config(self):
        flags = parse_flags([
            "--mode", "oss", "--oss-llm-key", "sk-oai",
            "--oss-vector", "pgvector",
            "--oss-vector-host", "db.local", "--oss-vector-port", "5433",
            "--oss-vector-user", "pg", "--oss-vector-dbname", "memdb",
        ])
        oss, _ = build_oss_config(flags)
        vs = oss["vector_store"]
        assert vs["provider"] == "pgvector"
        assert vs["config"]["host"] == "db.local"
        assert vs["config"]["port"] == 5433
        assert vs["config"]["user"] == "pg"

    def test_known_dims_auto_set(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai"])
        oss, _ = build_oss_config(flags)
        dims = oss["embedder"]["config"].get("embedding_dims")
        assert dims == 1536

    def test_custom_qdrant_path(self):
        flags = parse_flags([
            "--mode", "oss", "--oss-llm-key", "sk-oai",
            "--oss-vector-path", "/data/qdrant",
        ])
        oss, _ = build_oss_config(flags)
        assert oss["vector_store"]["config"]["path"] == "/data/qdrant"


class TestWriteEnv:

    def test_write_new_vars(self, tmp_path):
        env_path = tmp_path / ".env"
        _write_env(env_path, {"OPENAI_API_KEY": "sk-test"})
        content = env_path.read_text()
        assert "OPENAI_API_KEY=sk-test" in content

    def test_update_existing_var(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OPENAI_API_KEY=old\nOTHER=keep\n")
        _write_env(env_path, {"OPENAI_API_KEY": "new"})
        content = env_path.read_text()
        assert "OPENAI_API_KEY=new" in content
        assert "OTHER=keep" in content
        assert "old" not in content


class TestPostSetup:

    def test_platform_flag_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["hermes", "--mode", "platform", "--api-key", "sk-test"])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert config["memory"]["provider"] == "mem0"
        env_content = (tmp_path / ".env").read_text()
        assert "MEM0_API_KEY=sk-test" in env_content
        mem0_json = json.loads((tmp_path / "mem0.json").read_text())
        assert mem0_json["mode"] == "platform"

    def test_platform_setup_clears_stale_host(self, tmp_path, monkeypatch):
        # A user who previously ran self-hosted has host in mem0.json. Switching
        # to platform must drop host — otherwise routing (host > platform) keeps
        # sending them to the self-hosted server despite --mode platform.
        (tmp_path / "mem0.json").write_text(
            json.dumps({"mode": "platform", "host": "http://old-selfhosted:8888"})
        )
        monkeypatch.setattr("sys.argv", ["hermes", "--mode", "platform", "--api-key", "sk-test"])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        mem0_json = json.loads((tmp_path / "mem0.json").read_text())
        assert mem0_json["mode"] == "platform"
        assert not mem0_json.get("host")  # cleared to falsy so routing → platform

    def test_oss_flag_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "hermes", "--mode", "oss", "--oss-llm-key", "sk-oai",
        ])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._install_provider_deps", lambda l, e, v: None)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert config["memory"]["provider"] == "mem0"
        mem0_json = json.loads((tmp_path / "mem0.json").read_text())
        assert mem0_json["mode"] == "oss"
        assert mem0_json["oss"]["llm"]["provider"] == "openai"

    def test_selfhosted_flag_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "hermes", "--mode", "selfhosted",
            "--host", "http://localhost:8888/", "--api-key", "admin-key",
        ])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._check_selfhosted_server", lambda h: None)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert config["memory"]["provider"] == "mem0"
        env_content = (tmp_path / ".env").read_text()
        assert "MEM0_API_KEY=admin-key" in env_content
        mem0_json = json.loads((tmp_path / "mem0.json").read_text())
        assert mem0_json["host"] == "http://localhost:8888"  # trailing slash stripped
        assert mem0_json["user_id"] == "hermes-user"

    def test_selfhosted_no_api_key_auth_disabled(self, tmp_path, monkeypatch):
        # AUTH_DISABLED servers need no key — setup must not write one.
        monkeypatch.setattr("sys.argv", [
            "hermes", "--mode", "self-hosted", "--host", "http://mem0.lan:8888",
        ])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._check_selfhosted_server", lambda h: None)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert not (tmp_path / ".env").exists()
        mem0_json = json.loads((tmp_path / "mem0.json").read_text())
        assert mem0_json["host"] == "http://mem0.lan:8888"

    def test_selfhosted_dry_run_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "hermes", "--mode", "selfhosted",
            "--host", "http://localhost:8888", "--api-key", "k", "--dry-run",
        ])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._check_selfhosted_server", lambda h: None)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert not (tmp_path / ".env").exists()
        assert not (tmp_path / "mem0.json").exists()
        assert "provider" not in config["memory"]


class TestDryRun:

    def test_dry_run_flag_parsed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai", "--dry-run"])
        assert flags["dry_run"] is True

    def test_dry_run_not_set_by_default(self):
        flags = parse_flags(["--mode", "oss"])
        assert flags["dry_run"] is False

    def test_dry_run_platform_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["hermes", "--mode", "platform", "--api-key", "sk-test", "--dry-run"])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert not (tmp_path / ".env").exists()
        assert not (tmp_path / "mem0.json").exists()
        assert "provider" not in config["memory"]

    def test_dry_run_oss_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "hermes", "--mode", "oss", "--oss-llm-key", "sk-oai", "--dry-run",
        ])
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._install_provider_deps", lambda l, e, v: None)
        config = {"memory": {}}
        post_setup(str(tmp_path), config)
        assert not (tmp_path / ".env").exists()
        assert not (tmp_path / "mem0.json").exists()
        assert "provider" not in config["memory"]


class TestConnectivityChecks:

    def test_qdrant_path_writable(self, tmp_path):
        ok, msg = _check_qdrant_path(str(tmp_path / "qdrant"))
        assert ok is True

    def test_qdrant_path_not_writable(self, tmp_path, monkeypatch):
        def _raise_oserror(*a, **kw):
            raise OSError("Permission denied")
        monkeypatch.setattr(Path, "mkdir", _raise_oserror)
        ok, msg = _check_qdrant_path(str(tmp_path / "qdrant"))
        assert ok is False
        assert "Permission denied" in msg

    def test_ollama_unreachable(self):
        ok, msg = _check_ollama("http://localhost:1")
        assert ok is False

    def test_pgvector_unreachable(self):
        ok, msg = _check_pgvector("localhost", 1)
        assert ok is False
