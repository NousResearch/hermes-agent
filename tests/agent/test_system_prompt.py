from types import SimpleNamespace

import pytest

from agent import system_prompt


@pytest.fixture(autouse=True)
def _stub_prompt_dependencies(monkeypatch):
    monkeypatch.setattr(
        system_prompt,
        "_ra",
        lambda: SimpleNamespace(
            load_soul_md=lambda: None,
            build_nous_subscription_prompt=lambda tool_names: "",
            build_environment_hints=lambda: "",
            build_context_files_prompt=lambda cwd=None, skip_soul=False: "",
            build_skills_system_prompt=lambda **kwargs: "",
            get_toolset_for_tool=lambda tool_name: None,
        ),
    )
    monkeypatch.setattr(
        "agent.file_safety._resolve_active_profile_name",
        lambda: "default",
        raising=False,
    )


def _agent(*, platform="telegram", rich_messages=False, overrides=None):
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=set(),
        _kanban_worker_guidance=False,
        _tool_use_enforcement="auto",
        provider="test",
        model="test-model",
        platform=platform,
        _telegram_rich_messages_enabled=rich_messages,
        _platform_hint_overrides=overrides or {},
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_manager=None,
        pass_session_id=False,
        session_id=None,
    )


def _stable_prompt(agent):
    return system_prompt.build_system_prompt_parts(agent)["stable"]


def test_telegram_missing_or_false_rich_uses_legacy_hint():
    prompt = _stable_prompt(_agent(rich_messages=False))

    assert "Telegram has NO table syntax" in prompt
    assert "rich messages enabled" not in prompt
    assert "status/update" not in prompt
    assert "task-list checklist" not in prompt


def test_telegram_rich_enabled_uses_rich_hint_with_media_guidance():
    prompt = _stable_prompt(_agent(rich_messages=True))

    assert "rich messages enabled" in prompt
    assert "status/update" in prompt
    assert "task-list checklist" in prompt
    assert "MEDIA:/absolute/path/to/file" in prompt
    assert "Telegram has NO table syntax" not in prompt


def test_platform_hint_replace_overrides_selected_telegram_default():
    prompt = _stable_prompt(
        _agent(
            rich_messages=True,
            overrides={"telegram": {"replace": "CUSTOM TELEGRAM HINT"}},
        )
    )

    assert "CUSTOM TELEGRAM HINT" in prompt
    assert "rich messages enabled" not in prompt
    assert "Telegram has NO table syntax" not in prompt


def test_platform_hint_append_extends_selected_telegram_default():
    prompt = _stable_prompt(
        _agent(
            rich_messages=True,
            overrides={"telegram": {"append": "CUSTOM APPEND"}},
        )
    )

    assert "rich messages enabled" in prompt
    assert "CUSTOM APPEND" in prompt


def test_non_telegram_platform_hint_unaffected_by_telegram_rich_flag():
    prompt = _stable_prompt(_agent(platform="discord", rich_messages=True))

    assert "Discord" in prompt
    assert "rich messages enabled" not in prompt
    assert "Telegram has NO table syntax" not in prompt
