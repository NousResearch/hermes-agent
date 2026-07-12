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
    _prompt_api_key,
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

    def test_ollama_no_key_needed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm", "ollama", "--oss-embedder", "ollama"])
        oss, env_writes = build_oss_config(flags)
        assert oss["llm"]["provider"] == "ollama"
        assert "model" in oss["llm"]["config"]
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

    def test_preserves_non_ascii_existing_lines(self, tmp_path):
        """Existing non-ASCII .env content must survive the read-modify-write
        as UTF-8 (the locale codec would crash/mangle it on Windows)."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("PROXY_NOTE=café-zürich-完了\n".encode("utf-8"))
        _write_env(env_path, {"OPENAI_API_KEY": "sk-test"})
        content = env_path.read_text(encoding="utf-8")
        assert "PROXY_NOTE=café-zürich-完了" in content
        assert "OPENAI_API_KEY=sk-test" in content

    def test_updates_first_key_with_bom(self, tmp_path):
        """A Notepad-edited .env carries a BOM; the first key must still be
        matched/updated in place, not duplicated."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿OPENAI_API_KEY=old\n".encode("utf-8"))
        _write_env(env_path, {"OPENAI_API_KEY": "new"})
        content = env_path.read_text(encoding="utf-8")
        assert content.count("OPENAI_API_KEY=") == 1
        assert "OPENAI_API_KEY=new" in content


class TestPromptApiKey:

    def test_existing_key_found_behind_bom(self, tmp_path, monkeypatch):
        """The masked-current-value lookup must see a key on the BOM'd first
        line of a Notepad-edited .env instead of prompting from scratch."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿OPENAI_API_KEY=sk-existing\n".encode("utf-8"))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        prompts: list[str] = []

        def _fake_getpass(prompt):
            prompts.append(prompt)
            return ""

        monkeypatch.setattr("plugins.memory.mem0._setup.getpass.getpass", _fake_getpass)
        _prompt_api_key("OpenAI", "OPENAI_API_KEY", str(tmp_path))

        assert len(prompts) == 1
        assert "current: ...ting" in prompts[0]


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
