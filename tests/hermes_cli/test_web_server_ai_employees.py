"""Regression tests for Desktop AI employee registry APIs."""

from __future__ import annotations

import json

import pytest
import yaml


@pytest.fixture
def isolated_ai_employee_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated default home with one named Agent/profile and registry metadata."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "douyin-ingest-agent"

    for home in (default_home, worker_home):
        (home / "skills" / "demo-skill").mkdir(parents=True, exist_ok=True)
        (home / "skills" / "demo-skill" / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: demo\n---\n\n# Demo\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n  default: gpt-5.5\n",
            encoding="utf-8",
        )
        (home / "SOUL.md").write_text(
            "# Existing identity\n\nKeep this role body.\n",
            encoding="utf-8",
        )

    agents_dir = default_home / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "count": 3,
                "agents": [
                    {
                        "profile_id": "default",
                        "display_name_zh": "总控主理人",
                        "role_zh": "主控 Agent",
                        "mission_zh": "调度所有员工。",
                        "category": "orchestrator",
                        "emoji": "🧭",
                        "sort_order": 10,
                    },
                    {
                        "profile_id": "douyin-ingest-agent",
                        "display_name_zh": "抖音素材采集员",
                        "role_zh": "短视频链接解析专家",
                        "mission_zh": "采集短视频素材。",
                        "category": "video-pipeline",
                        "emoji": "📥",
                        "sort_order": 20,
                    },
                    {
                        "profile_id": "ghost-agent",
                        "display_name_zh": "幽灵员工",
                        "role_zh": "不存在",
                        "mission_zh": "应该被过滤。",
                        "category": "ghost",
                        "emoji": "👻",
                        "sort_order": 30,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_ai_employee_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_ai_employees_list_merges_registry_with_live_profiles(client):
    resp = client.get("/api/ai-employees")

    assert resp.status_code == 200
    payload = resp.json()
    assert [agent["profile_id"] for agent in payload["agents"]] == [
        "default",
        "douyin-ingest-agent",
    ]

    worker = payload["agents"][1]
    assert worker["display_name_zh"] == "抖音素材采集员"
    assert worker["role_zh"] == "短视频链接解析专家"
    assert worker["mission_zh"] == "采集短视频素材。"
    assert worker["model"] == "gpt-5.5"
    assert worker["provider"] == "openai-codex"
    assert worker["skill_count"] == 1
    assert worker["system_name_locked"] is True


def test_ai_employee_metadata_update_writes_registry_profile_yaml_and_soul(
    client, isolated_ai_employee_profiles
):
    resp = client.put(
        "/api/ai-employees/douyin-ingest-agent/metadata",
        json={
            "display_name_zh": "短视频采集专家",
            "role_zh": "短视频素材采集与解析",
            "mission_zh": "解析链接、提取字幕和关键帧。",
            "category": "video-pipeline",
            "emoji": "🎯",
        },
    )

    assert resp.status_code == 200
    updated = resp.json()
    assert updated["profile_id"] == "douyin-ingest-agent"
    assert updated["display_name_zh"] == "短视频采集专家"
    assert updated["mission_zh"] == "解析链接、提取字幕和关键帧。"

    home = isolated_ai_employee_profiles["default"]
    registry = json.loads((home / "agents" / "registry.json").read_text(encoding="utf-8"))
    worker_registry = next(a for a in registry["agents"] if a["profile_id"] == "douyin-ingest-agent")
    assert worker_registry["display_name_zh"] == "短视频采集专家"
    assert worker_registry["emoji"] == "🎯"

    profile_yaml = yaml.safe_load(
        (isolated_ai_employee_profiles["worker"] / "profile.yaml").read_text(encoding="utf-8")
    )
    assert profile_yaml["display_name_zh"] == "短视频采集专家"
    assert profile_yaml["role_zh"] == "短视频素材采集与解析"
    assert profile_yaml["description"] == "短视频采集专家：解析链接、提取字幕和关键帧。"
    assert profile_yaml["description_auto"] is False

    soul = (isolated_ai_employee_profiles["worker"] / "SOUL.md").read_text(encoding="utf-8")
    assert "中文显示名: 短视频采集专家" in soul
    assert "中文岗位: 短视频素材采集与解析" in soul
    assert "# Existing identity" in soul


def test_ai_employee_metadata_update_rejects_invalid_profile_id(client):
    resp = client.put(
        "/api/ai-employees/中文员工/metadata",
        json={"display_name_zh": "中文员工"},
    )

    assert resp.status_code == 400
