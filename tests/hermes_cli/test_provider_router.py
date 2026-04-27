"""Tests for Phase 6 Hermes Code Mode: ProviderRouter."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock


# =============================================================================
# Helpers
# =============================================================================


def _make_workspace(db, tmp_path, name="myproject"):
    project = tmp_path / name
    project.mkdir(exist_ok=True)
    return db.upsert_workspace(
        path=str(project),
        name=name,
        is_git_repo=True,
        branch="main",
        detected_stack=["python"],
    )


def _make_session(
    db, wdb, tmp_path, provider="openrouter", model="anthropic/claude-opus-4.6"
):
    ws = _make_workspace(wdb, tmp_path)
    return db.create_session(
        workspace_id=ws["id"],
        provider=provider,
        model=model,
    )


# =============================================================================
# ProviderRouterDB — persistence layer
# =============================================================================


class TestProviderRouterDB:
    @pytest.fixture()
    def db_path(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def pdb(self, db_path):
        from hermes_state import ProviderRouterDB

        d = ProviderRouterDB(db_path=db_path)
        yield d
        d.close()

    @pytest.fixture()
    def csdb(self, db_path):
        from hermes_state import CodeSessionDB, WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        csdb = CodeSessionDB(db_path=db_path)
        yield csdb, wdb
        csdb.close()
        wdb.close()

    def test_create_preset(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        preset = pdb.create_preset(
            code_session_id=session["id"],
            name="fast",
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
        )
        assert preset["id"]
        assert preset["name"] == "fast"
        assert preset["provider"] == "openrouter"
        assert preset["model"] == "anthropic/claude-haiku-4.5"

    def test_get_preset_by_name(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        pdb.create_preset(
            code_session_id=session["id"],
            name="strong",
            provider="openrouter",
            model="anthropic/claude-opus-4.6",
        )
        found = pdb.get_preset_by_name(session["id"], "strong")
        assert found is not None
        assert found["model"] == "anthropic/claude-opus-4.6"

    def test_get_preset_by_name_not_found(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        found = pdb.get_preset_by_name(session["id"], "nonexistent")
        assert found is None

    def test_list_presets(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        pdb.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        pdb.create_preset(
            session["id"], "strong", "openrouter", "anthropic/claude-opus-4.6"
        )
        presets = pdb.list_presets(session["id"])
        assert len(presets) == 2
        names = [p["name"] for p in presets]
        assert "fast" in names
        assert "strong" in names

    def test_update_preset(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        preset = pdb.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        updated = pdb.update_preset(
            preset["id"],
            model="anthropic/claude-sonnet-4.6",
        )
        assert updated["model"] == "anthropic/claude-sonnet-4.6"

    def test_delete_preset(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        preset = pdb.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        assert pdb.delete_preset(preset["id"]) is True
        assert pdb.get_preset(preset["id"]) is None

    def test_delete_nonexistent_preset(self, pdb):
        assert pdb.delete_preset("nonexistent") is False

    def test_add_cost_entry(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        entry = pdb.add_cost_entry(
            code_session_id=session["id"],
            provider="openrouter",
            model="anthropic/claude-opus-4.6",
            task_type="strong",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        assert entry["id"]
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert entry["cost_usd"] == 0.05

    def test_get_cost_summary(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        pdb.add_cost_entry(
            session["id"],
            "openrouter",
            "anthropic/claude-opus-4.6",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        pdb.add_cost_entry(
            session["id"],
            "openrouter",
            "anthropic/claude-haiku-4.5",
            input_tokens=2000,
            output_tokens=1000,
            cost_usd=0.02,
        )
        summary = pdb.get_cost_summary(session["id"])
        assert summary["entry_count"] == 2
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 1500
        assert summary["total_cost_usd"] == 0.07

    def test_get_cost_summary_by_provider(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        pdb.add_cost_entry(
            session["id"],
            "openrouter",
            "anthropic/claude-opus-4.6",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        pdb.add_cost_entry(
            session["id"],
            "nous",
            "xiaomi/mimo-v2-pro",
            input_tokens=2000,
            output_tokens=1000,
            cost_usd=0.01,
        )
        summary = pdb.get_cost_summary(session["id"])
        assert "openrouter" in summary["by_provider"]
        assert "nous" in summary["by_provider"]
        assert summary["by_provider"]["openrouter"]["cost_usd"] == 0.05
        assert summary["by_provider"]["nous"]["cost_usd"] == 0.01

    def test_list_cost_entries(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        for i in range(5):
            pdb.add_cost_entry(
                session["id"],
                "openrouter",
                "anthropic/claude-opus-4.6",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01,
            )
        entries = pdb.list_cost_entries(session["id"], limit=3)
        assert len(entries) == 3

    def test_empty_cost_summary(self, pdb, csdb, db_path, tmp_path):
        csdb, wdb = csdb
        session = _make_session(csdb, wdb, tmp_path)
        summary = pdb.get_cost_summary(session["id"])
        assert summary["entry_count"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["by_provider"] == {}


# =============================================================================
# ProviderRouter — service layer
# =============================================================================


class TestProviderRouter:
    @pytest.fixture()
    def db_path(self, tmp_path):
        return tmp_path / "state.db"

    @pytest.fixture()
    def router(self, db_path):
        from hermes_cli.code.provider_router import ProviderRouter

        return ProviderRouter(db_path=db_path)

    @pytest.fixture()
    def dbs(self, db_path):
        from hermes_state import CodeSessionDB, WorkspaceDB

        wdb = WorkspaceDB(db_path=db_path)
        csdb = CodeSessionDB(db_path=db_path)
        yield csdb, wdb
        csdb.close()
        wdb.close()

    # ── select_model ──

    def test_select_model_default(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        result = router.select_model(session["id"], "fast")
        assert result["task_type"] == "fast"
        assert result["provider"] == "openrouter"
        assert result["model"] == "anthropic/claude-haiku-4.5"
        assert result["source"] == "default"

    def test_select_model_session_preset(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        router.create_preset(session["id"], "fast", "nous", "xiaomi/mimo-v2-pro")
        result = router.select_model(session["id"], "fast")
        assert result["task_type"] == "fast"
        assert result["provider"] == "nous"
        assert result["model"] == "xiaomi/mimo-v2-pro"
        assert result["source"] == "session_preset"

    def test_select_model_updates_session(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        result = router.select_model(session["id"], "strong")
        updated = router.get_session_model(session["id"])
        assert updated["provider"] == result["provider"]
        assert updated["model"] == result["model"]

    def test_select_model_invalid_task_type(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        with pytest.raises(ValueError, match="Invalid task_type"):
            router.select_model(session["id"], "invalid")

    def test_select_model_nonexistent_session(self, router):
        with pytest.raises(ValueError, match="CodeSession not found"):
            router.select_model("nonexistent", "fast")

    # ── get_session_model ──

    def test_get_session_model(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(
            csdb, wdb, tmp_path, provider="nous", model="xiaomi/mimo-v2-pro"
        )
        result = router.get_session_model(session["id"])
        assert result["provider"] == "nous"
        assert result["model"] == "xiaomi/mimo-v2-pro"

    def test_get_session_model_nonexistent(self, router):
        with pytest.raises(ValueError, match="CodeSession not found"):
            router.get_session_model("nonexistent")

    # ── update_session_model ──

    def test_update_session_model(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        result = router.update_session_model(
            session["id"], "openrouter", "anthropic/claude-sonnet-4.6"
        )
        assert result["provider"] == "openrouter"
        assert result["model"] == "anthropic/claude-sonnet-4.6"
        assert result["old_provider"] == "openrouter"
        assert result["old_model"] == "anthropic/claude-opus-4.6"

    # ── Presets ──

    def test_create_preset(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        preset = router.create_preset(
            session["id"], "reviewer", "nous", "xiaomi/mimo-v2-pro"
        )
        assert preset["name"] == "reviewer"
        assert preset["provider"] == "nous"

    def test_create_preset_invalid_name(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        with pytest.raises(ValueError, match="Invalid preset name"):
            router.create_preset(
                session["id"], "invalid", "openrouter", "anthropic/claude-opus-4.6"
            )

    def test_create_preset_upsert(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        p1 = router.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        p2 = router.create_preset(session["id"], "fast", "nous", "xiaomi/mimo-v2-pro")
        assert p1["id"] == p2["id"]
        assert p2["provider"] == "nous"

    def test_list_presets(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        router.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        router.create_preset(
            session["id"], "strong", "openrouter", "anthropic/claude-opus-4.6"
        )
        presets = router.list_presets(session["id"])
        assert len(presets) == 2

    def test_delete_preset(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        preset = router.create_preset(
            session["id"], "fast", "openrouter", "anthropic/claude-haiku-4.5"
        )
        assert router.delete_preset(session["id"], preset["id"]) is True
        assert router.list_presets(session["id"]) == []

    # ── Cost Tracking ──

    def test_track_cost(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        entry = router.track_cost(
            session["id"],
            provider="openrouter",
            model="anthropic/claude-opus-4.6",
            task_type="strong",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        assert entry["id"]
        assert entry["input_tokens"] == 1000

    def test_get_session_cost_summary(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        router.track_cost(
            session["id"],
            "openrouter",
            "anthropic/claude-opus-4.6",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
        )
        router.track_cost(
            session["id"],
            "openrouter",
            "anthropic/claude-haiku-4.5",
            input_tokens=2000,
            output_tokens=1000,
            cost_usd=0.02,
        )
        summary = router.get_session_cost_summary(session["id"])
        assert summary["entry_count"] == 2
        assert summary["total_cost_usd"] == 0.07

    def test_list_cost_entries(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        for _ in range(3):
            router.track_cost(
                session["id"],
                "openrouter",
                "anthropic/claude-opus-4.6",
                input_tokens=100,
                cost_usd=0.01,
            )
        entries = router.list_cost_entries(session["id"], limit=2)
        assert len(entries) == 2

    def test_get_presets_summary(self, router, dbs, tmp_path):
        csdb, wdb = dbs
        session = _make_session(csdb, wdb, tmp_path)
        router.create_preset(session["id"], "fast", "nous", "xiaomi/mimo-v2-pro")
        summary = router.get_presets_summary(session["id"])
        assert summary["fast"]["source"] == "session"
        assert summary["fast"]["provider"] == "nous"
        assert summary["strong"]["source"] == "default"
        assert summary["strong"]["model"] == "anthropic/claude-opus-4.6"


# =============================================================================
# HTTP endpoints
# =============================================================================


class TestProviderRouterEndpoints:
    @pytest.fixture()
    def client(self, tmp_path):
        import hermes_state

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        db_path = hermes_home / "state.db"

        os.environ["HERMES_HOME"] = str(hermes_home)
        os.environ["HERMES_SESSION_TOKEN"] = "test-token"

        orig_default = hermes_state.DEFAULT_DB_PATH
        hermes_state.DEFAULT_DB_PATH = db_path

        from hermes_state import WorkspaceDB, CodeSessionDB

        wdb = WorkspaceDB(db_path=db_path)
        sdb = CodeSessionDB(db_path=db_path)
        wdb.close()
        sdb.close()

        from hermes_cli.web_server import app, _SESSION_TOKEN
        from starlette.testclient import TestClient

        test_client = TestClient(app)
        test_client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"
        yield test_client

        hermes_state.DEFAULT_DB_PATH = orig_default

    def _create_session(self, client, tmp_path):
        from hermes_state import WorkspaceDB, CodeSessionDB

        ws_path = tmp_path / "proj"
        ws_path.mkdir(exist_ok=True)
        wdb = WorkspaceDB()
        ws = wdb.upsert_workspace(
            path=str(ws_path), name="proj", is_git_repo=True, branch="main"
        )
        wdb.close()

        csdb = CodeSessionDB()
        session = csdb.create_session(
            workspace_id=ws["id"],
            provider="openrouter",
            model="anthropic/claude-opus-4.6",
        )
        csdb.close()
        return session["id"]

    def test_get_session_model(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.get(f"/api/code/sessions/{sid}/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "openrouter"

    def test_get_session_model_not_found(self, client, tmp_path):
        resp = client.get("/api/code/sessions/nonexistent/model")
        assert resp.status_code == 404

    def test_select_session_model(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.post(
            f"/api/code/sessions/{sid}/model/select",
            json={"task_type": "fast"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_type"] == "fast"
        assert data["model"] == "anthropic/claude-haiku-4.5"

    def test_select_session_model_invalid_type(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.post(
            f"/api/code/sessions/{sid}/model/select",
            json={"task_type": "invalid"},
        )
        assert resp.status_code == 400

    def test_update_session_model(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.put(
            f"/api/code/sessions/{sid}/model",
            json={"provider": "nous", "model": "xiaomi/mimo-v2-pro"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "nous"
        assert data["model"] == "xiaomi/mimo-v2-pro"

    def test_create_list_presets(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.post(
            f"/api/code/sessions/{sid}/presets",
            json={"name": "fast", "provider": "nous", "model": "xiaomi/mimo-v2-pro"},
        )
        assert resp.status_code == 200
        resp = client.get(f"/api/code/sessions/{sid}/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["presets"]["fast"]["provider"] == "nous"

    def test_delete_preset(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.post(
            f"/api/code/sessions/{sid}/presets",
            json={"name": "fast", "provider": "nous", "model": "xiaomi/mimo-v2-pro"},
        )
        preset_id = resp.json()["preset"]["id"]
        resp = client.delete(f"/api/code/sessions/{sid}/presets/{preset_id}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_track_cost(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        resp = client.post(
            f"/api/code/sessions/{sid}/cost",
            json={
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.6",
                "task_type": "strong",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost_usd": 0.05,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entry"]["input_tokens"] == 1000

    def test_get_cost_summary(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        client.post(
            f"/api/code/sessions/{sid}/cost",
            json={
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.6",
                "input_tokens": 1000,
                "cost_usd": 0.05,
            },
        )
        resp = client.get(f"/api/code/sessions/{sid}/cost")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["entry_count"] == 1

    def test_list_cost_entries(self, client, tmp_path):
        sid = self._create_session(client, tmp_path)
        for _ in range(3):
            client.post(
                f"/api/code/sessions/{sid}/cost",
                json={
                    "provider": "openrouter",
                    "model": "anthropic/claude-opus-4.6",
                    "input_tokens": 100,
                },
            )
        resp = client.get(f"/api/code/sessions/{sid}/cost/entries?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
