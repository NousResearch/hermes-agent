from types import SimpleNamespace

import json

from gateway.auto_background_runtime_service import (
    format_auto_background_ack,
    history_suggests_auto_background_work,
    is_auto_background_shortcut,
    looks_like_auto_background_work_request,
    resolve_auto_background_dispatch,
    resolve_employee_background_dispatch,
)
from gateway.platforms.base import MessageType
from gateway.session import SessionSource
from gateway.config import Platform


def _make_event(text: str, *, chat_type: str = "dm"):
    return SimpleNamespace(
        text=text,
        source=SessionSource(
            platform=Platform.QQ_NAPCAT,
            user_id="179033731",
            user_name="發發發",
            chat_id="726109087" if chat_type == "group" else "179033731",
            chat_type=chat_type,
        ),
        message_type=MessageType.TEXT,
        media_urls=[],
        get_command=lambda: None,
    )


def test_history_suggests_auto_background_work_from_recent_task_context():
    history = [
        {"role": "user", "content": "继续排查服务器日志里的 gateway 故障"},
        {"role": "assistant", "content": "收到，我继续查。"},
    ]

    assert history_suggests_auto_background_work(history) is True


def test_history_suggests_auto_background_work_ignores_assistant_only_task_text():
    history = [
        {"role": "assistant", "content": "我继续排查服务器日志里的 gateway 故障。"},
        {"role": "assistant", "content": "查完我回来汇报。"},
    ]

    assert history_suggests_auto_background_work(history) is False


def test_auto_background_intents_load_from_json_data(monkeypatch, tmp_path):
    import gateway.auto_background_runtime_service as service

    data_path = tmp_path / "auto_background_intents.json"
    data_path.write_text(
        json.dumps(
            {
                "shortcuts": ["开干"],
                "action_terms": ["攻坚"],
                "domain_terms": ["线路"],
                "worker_assignment": {
                    "lead_markers": ["安排"],
                    "tail_markers": ["攻坚"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "_DATA_PATH", data_path)
    service._load_auto_background_intents_data.cache_clear()

    assert is_auto_background_shortcut("开干") is True
    assert looks_like_auto_background_work_request("马上攻坚这条线路") is True

    service._load_auto_background_intents_data.cache_clear()


def test_auto_background_intents_fall_back_to_defaults_when_file_is_invalid(monkeypatch, tmp_path):
    import gateway.auto_background_runtime_service as service

    data_path = tmp_path / "auto_background_intents.json"
    data_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(service, "_DATA_PATH", data_path)
    service._load_auto_background_intents_data.cache_clear()

    assert is_auto_background_shortcut("继续") is True
    assert looks_like_auto_background_work_request("继续排查服务器日志里的 gateway 故障") is True

    service._load_auto_background_intents_data.cache_clear()


def test_resolve_employee_background_dispatch_prefers_explicit_worker_assignment():
    routes = [
        {
            "worker_name": "铁柱",
            "aliases": ["老铁"],
            "match_modes": ["explicit", "heuristic"],
            "action_terms": ["优化", "打磨"],
            "subject_terms": ["主页", "页面"],
            "pain_terms": ["粗糙"],
            "preloaded_skills": ["frontend-design"],
        }
    ]

    result = resolve_employee_background_dispatch(
        "让铁柱继续优化公司主页",
        employee_routes=routes,
        conversation_history=None,
    )

    assert result == {"worker_name": "铁柱", "preloaded_skills": ["frontend-design"]}


def test_resolve_auto_background_dispatch_requires_visible_group_address_for_new_group_work():
    event = _make_event("继续优化公司主页", chat_type="group")

    result = resolve_auto_background_dispatch(
        event,
        event.text,
        auto_background_work_enabled=True,
        employee_routes=[],
        conversation_history=None,
    )

    assert result is None


def test_resolve_auto_background_dispatch_allows_group_followup_shortcut_with_task_history():
    event = _make_event("继续", chat_type="group")
    history = [
        {"role": "user", "content": "帮我继续排查服务器 gateway 并发问题"},
        {"role": "assistant", "content": "收到，我继续查。"},
    ]

    result = resolve_auto_background_dispatch(
        event,
        event.text,
        auto_background_work_enabled=True,
        employee_routes=[],
        conversation_history=history,
    )

    assert result == {"worker_name": "", "preloaded_skills": []}


def test_resolve_auto_background_dispatch_uses_injected_group_visible_address_checker():
    event = _make_event("上服务器看看日志，把这个问题查清楚。", chat_type="group")

    result = resolve_auto_background_dispatch(
        event,
        event.text,
        auto_background_work_enabled=True,
        employee_routes=[],
        conversation_history=None,
        group_visible_address_checker=lambda body: True,
    )

    assert result == {"worker_name": "", "preloaded_skills": []}


def test_resolve_auto_background_dispatch_ignores_group_shortcut_with_assistant_only_task_history():
    event = _make_event("继续", chat_type="group")
    history = [
        {"role": "assistant", "content": "我继续排查服务器 gateway 并发问题。"},
        {"role": "assistant", "content": "查完回来汇报。"},
    ]

    result = resolve_auto_background_dispatch(
        event,
        event.text,
        auto_background_work_enabled=True,
        employee_routes=[],
        conversation_history=history,
    )

    assert result is None


def test_format_auto_background_ack_includes_worker_name():
    result = format_auto_background_ack("帮我继续优化公司主页", "bg_123", worker_name="铁柱")

    assert "铁柱" in result
    assert "bg_123" in result
