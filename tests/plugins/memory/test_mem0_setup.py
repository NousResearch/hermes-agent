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
from plugins.memory.mem0 import Mem0MemoryProvider


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

    def test_recall_mode_flag(self):
        flags = parse_flags(["--mode", "oss", "--recall-mode", "tools"])
        assert flags["recall_mode"] == "tools"


class TestRecallModeSchema:

    def test_schema_exposes_choices_without_environment_setting(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = Mem0MemoryProvider()

        fields = {field["key"]: field for field in provider.get_config_schema()}

        assert fields["recall_mode"]["default"] == "hybrid"
        assert fields["recall_mode"]["choices"] == ["hybrid", "context", "tools"]
        assert "env_var" not in fields["recall_mode"]


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

    @pytest.mark.parametrize(
        ("setup_fn", "flags"),
        [
            (
                "_setup_platform",
                {"api_key": "sk-test", "recall_mode": "tools", "dry_run": False},
            ),
            (
                "_setup_selfhosted",
                {
                    "host": "http://localhost:8888/",
                    "api_key": "admin-key",
                    "user_id": "user-1",
                    "recall_mode": "context",
                    "dry_run": False,
                },
            ),
            (
                "_setup_oss",
                {
                    "_mode_from_flag": True,
                    "oss_llm": "ollama",
                    "oss_embedder": "ollama",
                    "oss_vector": "qdrant",
                    "recall_mode": "context",
                    "user_id": "user-1",
                    "dry_run": False,
                },
            ),
        ],
    )
    def test_flag_driven_setup_persists_recall_mode_and_unknown_keys(
        self, tmp_path, monkeypatch, setup_fn, flags
    ):
        from plugins.memory.mem0 import _setup

        existing = {"recall_mode": "hybrid", "future_setting": {"keep": True}}
        (tmp_path / "mem0.json").write_text(json.dumps(existing), encoding="utf-8")
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._check_selfhosted_server", lambda h: None)
        monkeypatch.setattr("plugins.memory.mem0._setup._install_provider_deps", lambda *a: None)
        monkeypatch.setattr("plugins.memory.mem0._setup._run_connectivity_checks", lambda *a: None)
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0._setup._check_min_dep_version", lambda: None)
        monkeypatch.setattr("sys.argv", ["hermes"])
        config = {"memory": {}}

        getattr(_setup, setup_fn)(str(tmp_path), config, flags)

        saved = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert saved["recall_mode"] == flags["recall_mode"]
        assert saved["future_setting"] == {"keep": True}

    def test_selfhosted_without_flag_preserves_existing_recall_mode(self, tmp_path, monkeypatch):
        from plugins.memory.mem0 import _setup

        (tmp_path / "mem0.json").write_text(
            json.dumps({"recall_mode": "tools", "future_setting": 7}), encoding="utf-8"
        )
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr(_setup, "_check_selfhosted_server", lambda h: None)
        monkeypatch.setattr("plugins.memory.mem0._setup.get_hermes_home", lambda: tmp_path)
        config = {"memory": {}}
        flags = {
            "_mode_from_flag": True,
            "host": "http://localhost:8888/",
            "api_key": "admin-key",
            "user_id": "",
        }

        _setup._setup_selfhosted(str(tmp_path), config, flags)

        saved = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert saved["recall_mode"] == "tools"
        assert saved["future_setting"] == 7

    def test_oss_interactive_persists_selected_recall_mode(self, tmp_path, monkeypatch):
        from plugins.memory.mem0 import _setup

        (tmp_path / "mem0.json").write_text(
            json.dumps({"future_setting": "keep"}), encoding="utf-8"
        )
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr(_setup, "_prompt_api_key", lambda *a: "")
        monkeypatch.setattr(_setup, "_install_provider_deps", lambda *a: None)
        monkeypatch.setattr(_setup, "_run_connectivity_checks", lambda *a: None)
        monkeypatch.setattr(
            _setup,
            "_curses_select",
            lambda title, items, default=0: 1 if "Recall mode" in title else 0,
        )
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        config = {"memory": {}}

        _setup._setup_oss_interactive(str(tmp_path), config)

        saved = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert saved["recall_mode"] == "context"
        assert saved["future_setting"] == "keep"

    @pytest.mark.parametrize("setup_fn", ["_setup_platform", "_setup_selfhosted"])
    def test_interactive_setup_persists_selected_recall_mode(
        self, tmp_path, monkeypatch, setup_fn
    ):
        from plugins.memory.mem0 import _setup

        (tmp_path / "mem0.json").write_text(
            json.dumps({"future_setting": "keep"}), encoding="utf-8"
        )
        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setenv("MEM0_API_KEY", "existing-key")
        monkeypatch.setattr(_setup, "_check_selfhosted_server", lambda h: None)
        monkeypatch.setattr(
            _setup,
            "_curses_select",
            lambda title, items, default=0: 2 if "Recall mode" in title else default,
        )
        monkeypatch.setattr(
            _setup,
            "_prompt",
            lambda label, default=None, secret=False: default or (
                "http://localhost:8888" if "server URL" in label else ""
            ),
        )
        config = {"memory": {}}

        getattr(_setup, setup_fn)(str(tmp_path), config, {})

        saved = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert saved["recall_mode"] == "tools"
        assert saved["future_setting"] == "keep"

    def test_invalid_recall_mode_flag_defaults_to_hybrid(self, tmp_path, monkeypatch):
        from plugins.memory.mem0 import _setup

        _inject_fake_hermes_cli(monkeypatch)
        monkeypatch.setattr("plugins.memory.mem0._setup._install_provider_deps", lambda *a: None)
        monkeypatch.setattr("plugins.memory.mem0._setup._run_connectivity_checks", lambda *a: None)
        config = {"memory": {}}
        flags = {
            "_mode_from_flag": True,
            "oss_llm": "ollama",
            "oss_embedder": "ollama",
            "oss_vector": "qdrant",
            "recall_mode": "not-a-mode",
            "user_id": "user-1",
            "dry_run": False,
        }

        _setup._setup_oss(str(tmp_path), config, flags)

        saved = json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8"))
        assert saved["recall_mode"] == "hybrid"


class TestDryRun:

    def test_dry_run_flag_parsed(self):
        flags = parse_flags(["--mode", "oss", "--oss-llm-key", "sk-oai", "--dry-run"])
        assert flags["dry_run"] is True

    def test_oss_dry_run_does_not_write_recall_mode_or_secrets(self, tmp_path, monkeypatch):
        from plugins.memory.mem0 import _setup

        original = {"recall_mode": "tools", "future_setting": {"keep": True}}
        config_path = tmp_path / "mem0.json"
        config_path.write_text(json.dumps(original), encoding="utf-8")
        env_path = tmp_path / ".env"
        env_path.write_text("OPENAI_API_KEY=old\n", encoding="utf-8")
        monkeypatch.setattr(_setup, "_run_connectivity_checks", lambda *a: None)
        config = {"memory": {}}
        flags = {
            "_mode_from_flag": True,
            "oss_llm": "openai",
            "oss_llm_key": "new",
            "oss_embedder": "openai",
            "oss_vector": "qdrant",
            "recall_mode": "context",
            "user_id": "user-1",
            "dry_run": True,
        }

        _setup._setup_oss(str(tmp_path), config, flags)

        assert json.loads(config_path.read_text(encoding="utf-8")) == original
        assert env_path.read_text(encoding="utf-8") == "OPENAI_API_KEY=old\n"
        assert config == {"memory": {}}


class TestConnectivityChecks:

    def test_qdrant_path_writable(self, tmp_path):
        ok, msg = _check_qdrant_path(str(tmp_path / "qdrant"))
        assert ok is True
