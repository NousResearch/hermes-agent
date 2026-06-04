"""Tests for the deterministic /task task-tree browser."""

from __future__ import annotations

import json

from task_tree import (
    build_task_index_view,
    build_task_tree_view,
    build_task_callback_view,
    find_parent_task,
    load_task_tree_state,
)


def _write_state(tmp_path):
    state = {
        "version": 1,
        "updated_at": "2026-05-11T16:59:36+08:00",
        "active": [
            {
                "id": "active-service",
                "title": "JNU IBS 服务巡检：每天检查客户访问链路",
                "section": "运维主线",
                "status": "active",
                "execution_type": "infinite_standing",
                "cron_job_id": "health123",
                "subtasks": [
                    {"id": "svc-a", "title": "每日巡检 cron", "status": "active", "cron_job_id": "health123"},
                    {"id": "svc-b", "title": "故障归因", "status": "resolved", "resolution_evidence": "probe passed"},
                ],
            },
            {
                "id": "active-mobile",
                "fingerprint": "mobile-fp",
                "title": "IBS mobile v1：余额 / 用量 / 风险解释助手",
                "section": "产品主线",
                "status": "active",
                "execution_type": "finite",
                "subtasks": [
                    {
                        "id": "mobile-auth",
                        "title": "注册/登录入口 + 一账号绑定一房间 MVP",
                        "status": "pending",
                        "reasoning_level": "high",
                        "acceptance": "登录/注册入口清晰",
                        "acceptance_steps": [
                            {"id": "auth-method", "title": "选择低成本账号方式", "status": "pending"},
                            {"id": "auth-test", "title": "测试授权状态机", "status": "resolved", "evidence": "pytest passed"},
                        ],
                    }
                ],
            },
        ],
        "pending": [
            {"id": "pending-cron", "title": "排查 cron/API 连接失败", "status": "pending"}
        ],
        "resolved_recent": [],
    }
    path = tmp_path / "todo-state.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def test_task_index_lists_only_parent_tasks_with_counts(tmp_path):
    path = _write_state(tmp_path)
    view = build_task_index_view(path=path)

    assert "Tasks" in view.text
    assert "IBS mobile v1" in view.text
    assert "注册/登录入口" not in view.text
    assert "subtasks:" in view.text
    assert "steps:" in view.text
    assert any(button.callback_data == "task:p:1" for row in view.buttons for button in row)


def test_task_query_renders_parent_tree_and_acceptance_steps(tmp_path):
    path = _write_state(tmp_path)
    match = find_parent_task("mobile", path=path)
    assert match is not None
    assert match.index == 1

    view = build_task_tree_view("task2", path=path)
    assert "IBS mobile v1" in view.text
    assert "注册/登录入口" in view.text
    assert "选择低成本账号方式" in view.text
    assert "reasoning: high" in view.text
    assert any(button.callback_data == "task:s:1:0" for row in view.buttons for button in row)


def test_callback_drills_into_subtask_and_step_details(tmp_path):
    path = _write_state(tmp_path)

    subtask = build_task_callback_view("task:s:1:0", path=path)
    assert "Subtask" in subtask.text
    assert "mobile-auth" in subtask.text
    assert "登录/注册入口清晰" in subtask.text
    assert any(button.callback_data == "task:step:1:0:1" for row in subtask.buttons for button in row)

    step = build_task_callback_view("task:step:1:0:1", path=path)
    assert "Step" in step.text
    assert "测试授权状态机" in step.text
    assert "pytest passed" in step.text


def test_task_renderer_redacts_sensitive_values(tmp_path):
    path = _write_state(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["active"][1]["subtasks"][0]["resolution_evidence"] = (
        "token=" + "abcdef123456" + " password='" + "super-secret" + "'"
    )
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    view = build_task_callback_view("task:s:1:0", path=path)
    assert "abcdef123456" not in view.text
    assert "super-secret" not in view.text
    assert "[REDACTED]" in view.text


def test_redacts_bare_provider_tokens():
    from task_tree import redact_sensitive

    text = "sk-" + "A" * 20 + " xoxb-" + "1" * 20 + " ghp_" + "B" * 20
    redacted = redact_sensitive(text)
    assert "sk-" not in redacted
    assert "xoxb-" not in redacted
    assert "ghp_" not in redacted
    assert "A" * 20 not in redacted
    assert "1" * 20 not in redacted
    assert "B" * 20 not in redacted
    assert redacted.count("[REDACTED]") == 3


def test_redacts_json_env_and_bearer_token_shapes():
    from task_tree import redact_sensitive

    secret_a = "abc123def456ghi789"
    secret_b = "xyz987uvw654rst321"
    secret_c = "header.payload.signaturetoken"
    secret_d = "AIza" + "C" * 32
    env_token = ("SLACK_BOT_" + "TOKEN") + chr(61) + secret_a
    env_api_key = ("GOOGLE_" + "API_KEY") + chr(61) + secret_d
    text = (
        '{"token":"' + secret_a + '", '
        '"api_key":"' + secret_b + '"} '
        + env_token + ' '
        + env_api_key + ' '
        + 'Authorization: Bearer ' + secret_c
    )

    redacted = redact_sensitive(text)

    for secret in (secret_a, secret_b, secret_c, secret_d):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") >= 5


def test_parent_view_is_bounded_for_telegram(tmp_path):
    path = _write_state(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["active"][1]["subtasks"] = [
        {
            "id": f"s{i}",
            "title": "Very long subtask title " + ("x" * 220),
            "status": "pending",
            "acceptance_steps": [
                {"id": f"s{i}-step{j}", "title": "Very long step " + ("y" * 220), "status": "pending"}
                for j in range(8)
            ],
        }
        for i in range(25)
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    view = build_task_callback_view("task:p:1", path=path)
    assert len(view.text) < 3800
    assert "truncated" in view.text.lower() or "more" in view.text.lower()


def test_no_match_query_is_bounded(tmp_path):
    path = _write_state(tmp_path)
    query = "not-found-" + "x" * 5000
    view = build_task_tree_view(query, path=path)

    assert len(view.text) <= 4096
    assert "No matching parent task" in view.text
    assert "…" in view.text


def test_load_task_tree_state_handles_missing_file(tmp_path):
    snapshot = load_task_tree_state(tmp_path / "missing.json")
    assert snapshot.parents == []
    assert "missing.json" in str(snapshot.source_path)
