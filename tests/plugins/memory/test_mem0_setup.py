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
        assert oss["llm"]["config"]["is_reasoning_model"] is True
        assert oss["embedder"]["provider"] == "openai"
        assert oss["embedder"]["config"]["model"] == "text-embedding-3-small"
        assert oss["vector_store"]["provider"] == "qdrant"
        assert env_writes["OPENAI_API_KEY"] == "sk-oai"


    def test_explicit_gpt_5_mini_is_reasoning_model(self):
        flags = parse_flags([
            "--mode", "oss", "--oss-llm-key", "sk-oai",
            "--oss-llm-model", "gpt-5-mini",
        ])
        oss, _ = build_oss_config(flags)
        assert oss["llm"]["config"]["model"] == "gpt-5-mini"
        assert oss["llm"]["config"]["is_reasoning_model"] is True


    def test_custom_openai_model_is_not_forced_to_reasoning(self):
        flags = parse_flags([
            "--mode", "oss", "--oss-llm-key", "sk-oai",
            "--oss-llm-model", "gpt-5.2",
        ])
        oss, _ = build_oss_config(flags)
        assert oss["llm"]["config"]["model"] == "gpt-5.2"
        assert "is_reasoning_model" not in oss["llm"]["config"]


    def test_ollama_no_key_needed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm", "ollama", "--oss-embedder", "ollama"])
        oss, env_writes = build_oss_config(flags)
        assert oss["llm"]["provider"] == "ollama"
        assert "model" in oss["llm"]["config"]
        assert oss["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert "is_reasoning_model" not in oss["llm"]["config"]
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


class TestDryRun:

    def test_dry_run_flag_parsed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai", "--dry-run"])
        assert flags["dry_run"] is True


class TestConnectivityChecks:

    def test_qdrant_path_writable(self, tmp_path):
        ok, msg = _check_qdrant_path(str(tmp_path / "qdrant"))
        assert ok is True




class TestMem0JsonMergeWrite:
    """The merge-write must never turn into a full replacement.

    _save_mem0_json reads mem0.json, updates it, and writes the whole file
    back. Swallowing the read left `existing` empty, so every key the call
    does not set was dropped — api_key and host among them.
    """

    @staticmethod
    def _save(home, data):
        from plugins.memory.mem0._setup import _save_mem0_json

        _save_mem0_json(str(home), data)

    def test_bom_prefixed_config_keeps_untouched_keys(self, tmp_path):
        import json

        original = {
            "mode": "platform",
            "api_key": "sk-user-key",
            "host": "https://mem0.example",
            "custom": {"keep": "me"},
        }
        path = tmp_path / "mem0.json"
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(original, indent=2).encode())

        self._save(tmp_path, {"mode": "oss", "agent_id": "hermes"})

        after = json.loads(path.read_text(encoding="utf-8-sig"))
        assert after["api_key"] == "sk-user-key"
        assert after["host"] == "https://mem0.example"
        assert after["custom"] == {"keep": "me"}
        assert after["mode"] == "oss"
        assert after["agent_id"] == "hermes"

    def test_unreadable_config_is_not_overwritten(self, tmp_path):
        import pytest as _pytest

        path = tmp_path / "mem0.json"
        path.write_text("{ truncated", encoding="utf-8")

        with _pytest.raises(RuntimeError):
            self._save(tmp_path, {"mode": "oss"})

        assert path.read_text(encoding="utf-8") == "{ truncated"

    def test_non_object_config_is_not_overwritten(self, tmp_path):
        import pytest as _pytest

        path = tmp_path / "mem0.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with _pytest.raises(RuntimeError):
            self._save(tmp_path, {"mode": "oss"})

        assert path.read_text(encoding="utf-8") == "[1, 2, 3]"

    def test_missing_file_is_created(self, tmp_path):
        import json

        self._save(tmp_path, {"mode": "oss", "user_id": "alice"})

        after = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert after == {"mode": "oss", "user_id": "alice"}

    def test_plain_config_merges_normally(self, tmp_path):
        import json

        path = tmp_path / "mem0.json"
        path.write_text(json.dumps({"api_key": "k", "mode": "platform"}), encoding="utf-8")

        self._save(tmp_path, {"mode": "oss"})

        after = json.loads(path.read_text(encoding="utf-8"))
        assert after == {"api_key": "k", "mode": "oss"}
