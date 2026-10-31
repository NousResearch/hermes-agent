"""Live session provenance at creation, rotation and Desktop persistence (#108229)."""

import copy
import json
from types import SimpleNamespace

import pytest

from tests.agent.test_compression_rotation_state import _build_agent_with_db, _msgs
from hermes_state import SessionDB


@pytest.mark.parametrize("tier", ["priority", None, "auto", "cold"])
@pytest.mark.parametrize("provider", ["openai", "custom", "custom:passthrough"])
@pytest.mark.parametrize("publication", ["creation", "rotation"])
def test_live_runtime_metadata_survives_publication(tmp_path, monkeypatch, tier, provider, publication):
    import hermes_cli.runtime_provider as rp
    from tools.approval import enable_session_yolo, disable_session_yolo

    endpoint = "https://example.invalid/v1"
    monkeypatch.setattr(rp, "load_config", lambda: {
        "custom_providers": [{"name": "test-route", "base_url": endpoint}],
    })
    with SessionDB(db_path=tmp_path / "state.db") as db:
        agent = _build_agent_with_db(db, "runtime-parent")
        agent.context_compressor.context_length = 256_000
        agent.context_compressor.threshold_tokens = 128_000
        agent.service_tier = tier
        agent.provider = provider
        agent.base_url = endpoint
        agent.model = "runtime-model-0"
        agent.reasoning_config = {"effort": "high"}
        agent.max_iterations = 17
        agent.max_tokens = 321
        agent._session_init_model_config["_delegate_from"] = "spawner"
        agent.request_overrides = {"extra_body": {"user": "proxy-user"}}
        original_overrides = copy.deepcopy(agent.request_overrides)
        snapshot = copy.deepcopy(agent._session_init_model_config)
        enable_session_yolo(agent.session_id)
        try:
            if publication == "creation":
                agent._ensure_db_session()
                modes = [tier]
            else:
                # Seed the parent independently so a creation failure cannot hide a rotation failure.
                db.create_session(agent.session_id, source="telegram", model_config=snapshot)
                agent._session_db_created = True
                modes = [tier, None, "priority", "auto", "cold"]
            parent_config = db.get_session("runtime-parent")["model_config"]
            for index, current_tier in enumerate(modes):
                agent.service_tier = current_tier
                current_provider = provider if index % 2 == 0 else "openai"
                agent.provider = current_provider
                agent.model = f"runtime-model-{index}"
                agent.reasoning_config = {"effort": "high"}
                agent.max_iterations = 17
                agent.max_tokens = 321
                if publication == "rotation":
                    parent = agent.session_id
                    # The surface owns re-arming YOLO on its current identity; publication must carry it.
                    enable_session_yolo(parent)
                    agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
                    disable_session_yolo(parent)
                    assert agent.session_id != parent
                    assert db.get_session(agent.session_id)["parent_session_id"] == parent
                config = json.loads(db.get_session(agent.session_id)["model_config"])
                assert config.get("service_tier") == (current_tier or "normal")
                assert config["provider"] == ("custom:test-route" if current_provider == "custom" else current_provider)
                assert config["model"] == agent.model
                assert config["reasoning_config"] == agent.reasoning_config
                assert config["max_iterations"] == agent.max_iterations
                assert config["max_tokens"] == agent.max_tokens
                assert config["_delegate_from"] == "spawner"
                assert config["yolo_mode"] is True
                assert not {"api_key", "base_url", "request_overrides"}.intersection(config)
                assert agent.request_overrides == original_overrides
                assert agent._session_init_model_config == snapshot
            assert db.get_session("runtime-parent")["model_config"] == parent_config
        finally:
            disable_session_yolo("runtime-parent")
            disable_session_yolo(agent.session_id)
            agent.close()


def test_desktop_persistence_distinguishes_observed_normal_from_missing(tmp_path):
    from tui_gateway.server import (
        _persist_live_session_runtime,
        _runtime_model_config,
        _stored_session_runtime_overrides,
    )

    with SessionDB(db_path=tmp_path / "state.db") as db:
        agent = _build_agent_with_db(db, "normal-runtime", platform="desktop")
        agent._ensure_db_session()
        try:
            agent.service_tier = None
            _persist_live_session_runtime({"agent": agent, "session_key": agent.session_id})
            row = db.get_session(agent.session_id)
            assert json.loads(row["model_config"]).get("service_tier") == "normal"
            assert _stored_session_runtime_overrides(row)["service_tier_override"] == ""
            assert "service_tier" not in _runtime_model_config(SimpleNamespace(), {"service_tier": "priority"})
            del agent.service_tier
            agent._session_init_model_config["service_tier"] = "priority"
            assert "service_tier" not in agent._session_row_model_config()
        finally:
            agent.close()


@pytest.mark.parametrize("publication", ["creation", "rotation"])
@pytest.mark.parametrize("entry_moved", [False, True])
def test_publication_keeps_the_entry_serving_the_live_endpoint(tmp_path, monkeypatch, publication, entry_moved):
    import hermes_cli.runtime_provider as rp

    endpoint = "https://example.invalid/v1"
    monkeypatch.setattr(rp, "load_config", lambda: {
        "custom_providers": [
            {"name": "endpoint-owner", "base_url": endpoint},
            {"name": "requested", "base_url": "https://moved.invalid/v1" if entry_moved else endpoint},
        ],
    })
    with SessionDB(db_path=tmp_path / "state.db") as db:
        agent = _build_agent_with_db(db, "named-parent")
        agent.context_compressor.context_length = 256_000
        agent.context_compressor.threshold_tokens = 128_000
        try:
            agent.provider = "custom"
            agent.requested_provider = "custom:requested"
            agent.base_url = endpoint
            agent.service_tier = "priority"
            if publication == "creation":
                agent._ensure_db_session()
            else:
                db.create_session(agent.session_id, source="tool")
                agent._session_db_created = True
                agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
                assert agent.session_id != "named-parent"
                assert db.get_session(agent.session_id)["source"] == "tool"
            config = json.loads(db.get_session(agent.session_id)["model_config"])
            assert config.get("provider") == ("custom:endpoint-owner" if entry_moved else "custom:requested")
            assert config["service_tier"] == "priority"
        finally:
            agent.close()


def test_session_projection_does_not_adopt_the_borrowed_voice_route(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        agent = _build_agent_with_db(db, "voice-parent")
        try:
            agent._voice_route_state = {
                "snapshot": {"model": "session-model", "provider": "openai"},
                "reasoning_config": {"effort": "high"},
            }
            agent.model = "voice-model"
            agent.provider = "nous"
            agent.reasoning_config = {"enabled": False}
            agent._ensure_db_session()
            config = json.loads(db.get_session(agent.session_id)["model_config"])
            assert config.get("model") == "session-model"
            assert config["provider"] == "openai"
            assert config["reasoning_config"] == {"effort": "high"}
            assert (agent.model, agent.provider, agent.reasoning_config) == ("voice-model", "nous", {"enabled": False})
        finally:
            agent.close()
