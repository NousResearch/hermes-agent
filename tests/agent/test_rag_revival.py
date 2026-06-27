"""Tests for agent.rag_revival.

These tests exercise the RAG Revival snapshot writer in isolation and verify
that the compression-integration hooks in agent/conversation_compression.py
invoke it exactly once per compaction boundary without duplicating notes.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir(parents=True, exist_ok=True)
    return v


@pytest.fixture
def dummy_agent():
    agent = MagicMock()
    agent.profile = "test-profile"
    agent.model = "test/model"
    agent.provider = "test-provider"
    agent.platform = "cli"
    agent.session_id = "sess-123"
    return agent


class TestWriteRagRevivalSnapshot:
    def test_writes_to_profile_subdirectory(self, vault: Path, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        os.environ["HERMES_RAG_REVIVAL_ENABLED"] = "1"
        from agent.rag_revival import write_rag_revival_snapshot

        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[{"role": "user", "content": "hello"}],
            summary_text="summary",
            old_session_id="old-sid",
            new_session_id="new-sid",
            approx_tokens=1234,
            in_place=True,
            focus_topic="feeds",
        )
        assert path is not None
        assert path.exists()
        expected_parent = vault / "01 Projetos" / "Ágora" / "RAG Revival" / "test-profile"
        assert path.parent == expected_parent
        text = path.read_text(encoding="utf-8")
        assert "RAG Revival" in text
        assert "test-profile" in text
        assert "old-sid" in text
        assert "new-sid" in text
        assert "test/model" in text
        assert "test-provider" in text
        assert "summary" in text
        assert "feeds" in text
        assert "1234" in text

    def test_uses_session_id_when_rotation_ids_missing(self, vault: Path, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent.rag_revival import write_rag_revival_snapshot

        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[],
            summary_text="s",
            old_session_id="",
            new_session_id="",
        )
        assert path is not None
        assert "sess-123" in path.name

    def test_redacts_secrets_in_summary_and_messages(self, vault: Path, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent.rag_revival import write_rag_revival_snapshot

        secret = "sk-abc123defghi"
        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[{"role": "user", "content": f"key={secret}"}],
            summary_text=f"used {secret}",
        )
        text = path.read_text(encoding="utf-8")
        assert secret not in text
        assert "***" in text or "abc1***" in text or "sk-abc...v" in text

    def test_disabled_by_config(self, vault: Path, dummy_agent, monkeypatch):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        monkeypatch.delenv("HERMES_RAG_REVIVAL_ENABLED", raising=False)
        from agent import rag_revival
        from agent.rag_revival import write_rag_revival_snapshot

        monkeypatch.setattr(rag_revival, "_load_rag_config", lambda: {"enabled": False, "vault_dir": ""})
        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[],
            summary_text="s",
        )
        assert path is None
        assert not (vault / "01 Projetos").exists()

    def test_env_override_still_disables(self, vault: Path, dummy_agent, monkeypatch):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.rag_revival import write_rag_revival_snapshot

        # config says enabled, but env override disables (back-compat bridge)
        monkeypatch.setattr(rag_revival, "_load_rag_config", lambda: {"enabled": True, "vault_dir": ""})
        monkeypatch.setenv("HERMES_RAG_REVIVAL_ENABLED", "false")
        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[],
            summary_text="s",
        )
        assert path is None

    def test_uses_config_vault_dir(self, vault: Path, dummy_agent, monkeypatch):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.rag_revival import write_rag_revival_snapshot

        monkeypatch.setattr(
            rag_revival,
            "_load_rag_config",
            lambda: {"enabled": True, "vault_dir": "Custom/RAG"},
        )
        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[],
            summary_text="s",
        )
        assert path is not None
        assert path.parent == vault / "Custom" / "RAG" / "test-profile"

    def test_missing_vault_returns_none_non_blocking(self, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = "/nonexistent/path/no-vault"
        from agent.rag_revival import write_rag_revival_snapshot

        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[],
            summary_text="s",
        )
        assert path is None

    def test_fail_open_does_not_raise(self, vault: Path, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent.rag_revival import write_rag_revival_snapshot

        # Make the directory unwritable to force an internal error.
        vault.chmod(0o555)
        try:
            path = write_rag_revival_snapshot(
                dummy_agent,
                messages=[],
                summary_text="s",
            )
            assert path is None
        finally:
            vault.chmod(0o755)

    def test_unique_filenames_for_same_session(self, vault: Path, dummy_agent):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent.rag_revival import write_rag_revival_snapshot

        paths = []
        for _ in range(3):
            p = write_rag_revival_snapshot(
                dummy_agent,
                messages=[],
                summary_text="s",
                new_session_id="same-sid",
            )
            paths.append(p)
            time.sleep(0.001)
        assert len({p.name for p in paths}) == 3

    def test_loads_agora_status(self, vault: Path, dummy_agent, tmp_path: Path):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival

        # Build a tiny Ágora DB in the redirected Hermes home.
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        db_path = hermes_home / "agora.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE agora_agent_status (
                profile TEXT PRIMARY KEY,
                state TEXT,
                current_task_id TEXT,
                current_step TEXT,
                status_text TEXT,
                last_heartbeat_at REAL,
                pid INTEGER,
                run_id INTEGER
            );
            CREATE TABLE agora_notifications (
                id INTEGER PRIMARY KEY,
                recipient TEXT,
                read_at REAL,
                ack_at REAL
            );
            INSERT INTO agora_agent_status VALUES (
                'test-profile', 'running', 't_abc', 'step1', 'busy', 0.0, 123, 1
            );
            INSERT INTO agora_notifications VALUES (1, 'test-profile', NULL, NULL);
            """
        )
        conn.commit()
        conn.close()

        with patch.object(
            rag_revival, "_vault_path", return_value=vault
        ), patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}), patch(
            "hermes_constants.get_default_hermes_root", return_value=hermes_home
        ):
            from agent.rag_revival import write_rag_revival_snapshot

            path = write_rag_revival_snapshot(
                dummy_agent,
                messages=[],
                summary_text="s",
            )
        text = path.read_text(encoding="utf-8")
        assert "running" in text
        assert "t_abc" in text
        assert "unack" in text

    def test_message_excerpt_handles_list_and_strips_large_content(
        self, vault: Path, dummy_agent
    ):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent.rag_revival import write_rag_revival_snapshot

        path = write_rag_revival_snapshot(
            dummy_agent,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "x " * 2000}]},
                {"role": "assistant", "content": "short"},
            ],
            summary_text="s",
        )
        text = path.read_text(encoding="utf-8")
        assert "## Recent live tail still in context" in text
        assert "short" in text


class TestCompressionHooks:
    def _build_agent(self, db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="cli",
                quiet_mode=True,
                session_db=db,
                session_id="PARENT_RR",
                skip_context_files=True,
                skip_memory=True,
            )
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[summary]"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_summary_auth_failure = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        compressor._last_generated_summary = "Generated handoff summary."
        agent.context_compressor = compressor
        return agent

    def test_in_place_writes_single_snapshot(self, vault: Path, tmp_path: Path):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("PARENT_RR", source="cli")
        agent = self._build_agent(db)
        agent.compression_in_place = True

        with patch.object(
            rag_revival, "_load_rag_config", return_value={"enabled": True, "vault_dir": ""}
        ):
            compress_context(
                agent,
                [{"role": "user", "content": f"m{i}"} for i in range(20)],
                "sys",
                approx_tokens=120_000,
            )
        notes = list(vault.rglob("*.md"))
        assert len(notes) == 1
        text = notes[0].read_text(encoding="utf-8")
        assert "PARENT_RR" in text
        assert "Generated handoff summary" in text
        assert "In-place compression: `True`" in text

    def test_rotation_writes_single_snapshot(self, vault: Path, tmp_path: Path):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("PARENT_RR", source="cli")
        agent = self._build_agent(db)
        agent.compression_in_place = False

        with patch.object(
            rag_revival, "_load_rag_config", return_value={"enabled": True, "vault_dir": ""}
        ):
            compress_context(
                agent,
                [{"role": "user", "content": f"m{i}"} for i in range(20)],
                "sys",
                approx_tokens=120_000,
            )
        assert agent.session_id != "PARENT_RR"
        notes = list(vault.rglob("*.md"))
        assert len(notes) == 1
        text = notes[0].read_text(encoding="utf-8")
        assert "PARENT_RR" in text
        assert agent.session_id in text
        assert "Generated handoff summary" in text
        assert "In-place compression: `False`" in text

    def test_rotation_without_session_db_writes_single_snapshot(
        self, vault: Path, tmp_path: Path
    ):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.conversation_compression import compress_context

        agent = self._build_agent(None)
        agent._session_db = None
        agent.compression_in_place = False

        with patch.object(
            rag_revival, "_load_rag_config", return_value={"enabled": True, "vault_dir": ""}
        ):
            compress_context(
                agent,
                [{"role": "user", "content": f"m{i}"} for i in range(20)],
                "sys",
                approx_tokens=120_000,
            )
        notes = list(vault.rglob("*.md"))
        assert len(notes) == 1

    def test_no_snapshot_when_no_summary_generated(self, vault: Path, tmp_path: Path):
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        from agent import rag_revival
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("PARENT_RR", source="cli")
        agent = self._build_agent(db)
        agent.compression_in_place = True
        agent.context_compressor._last_generated_summary = None

        with patch.object(
            rag_revival, "_load_rag_config", return_value={"enabled": True, "vault_dir": ""}
        ):
            compress_context(
                agent,
                [{"role": "user", "content": f"m{i}"} for i in range(20)],
                "sys",
                approx_tokens=120_000,
            )
        notes = list(vault.rglob("*.md"))
        assert len(notes) == 0
