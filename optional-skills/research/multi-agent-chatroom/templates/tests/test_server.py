# tests/test_server.py
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "channels" in data


@pytest.mark.asyncio
async def test_models():
    from server.models import Message
    msg = Message(
        channel="#tasks",
        sender="supervisor",
        content="Test task",
        msg_type="task",
        metadata={"task_id": "T001"}
    )
    json_str = msg.to_json()
    parsed = Message.from_json(json_str)
    assert parsed.channel == "#tasks"
    assert parsed.sender == "supervisor"
    assert parsed.metadata["task_id"] == "T001"


@pytest.mark.asyncio
async def test_config_channels():
    from server.config import get_channels
    channels = get_channels()
    assert "#tasks" in channels.values()
    assert "#review" in channels.values()


def test_config_new_structure():
    """Test the new coding/reviewers/supervisor config structure."""
    from server.config import load_config, get_coding_config, get_reviewers_config, get_supervisor_config

    config = load_config()

    # Coding agent
    coding = get_coding_config(config)
    assert "name" in coding
    assert "provider" in coding
    assert "model" in coding
    assert coding["name"] == "deepseek-researcher"

    # Reviewers
    reviewers = get_reviewers_config(config)
    assert len(reviewers) >= 2
    # At least one synthesizer
    synthesizers = [r for r in reviewers if r.get("role") == "reviewer+synthesizer"]
    assert len(synthesizers) >= 1, "Need at least one reviewer+synthesizer"
    # At least one mathematical-rigor or reviewer
    eval_reviewers = [r for r in reviewers if r.get("role") in ("mathematical-rigor", "reviewer")]
    assert len(eval_reviewers) >= 1, "Need at least one mathematical reviewer"

    # Supervisor
    supervisor = get_supervisor_config(config)
    assert "name" in supervisor
    assert "provider" in supervisor
    assert "model" in supervisor

    # Project
    from server.config import get_project_config
    project = get_project_config(config)
    assert "workdir" in project


def test_config_model_overrides():
    """Verify that changing models in config doesn't break the loader."""
    from server.config import get_coding_config, get_reviewers_config

    # Simulate a config with different models
    test_config = {
        "coding": {
            "name": "coder",
            "provider": "openai",
            "model": "o4-mini",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "reviewers": [
            {
                "name": "reviewer-1",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "role": "reviewer+synthesizer",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
            {
                "name": "reviewer-2",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "role": "mathematical-rigor",
                "temperature": 0.4,
                "max_tokens": 4096,
            },
        ],
    }

    coding = get_coding_config(test_config)
    assert coding["provider"] == "openai"
    assert coding["model"] == "o4-mini"

    reviewers = get_reviewers_config(test_config)
    assert reviewers[0]["model"] == "deepseek-chat"
    assert reviewers[1]["model"] == "claude-sonnet-4-6"
