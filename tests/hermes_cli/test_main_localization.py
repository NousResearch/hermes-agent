import sys

import pytest

from hermes_cli.commands import get_category_label
from hermes_cli.main import (
    _build_web_ui,
    _model_flow_openrouter,
    _model_flow_qwen_oauth,
    _prompt_provider_choice,
    main,
)


def test_get_category_label_localizes_builtin_categories():
    assert get_category_label("Session") == "세션"
    assert get_category_label("Configuration") == "설정"
    assert get_category_label("Tools & Skills") == "도구와 스킬"
    assert get_category_label("Info") == "정보"
    assert get_category_label("Exit") == "종료"


def test_main_help_is_localized_to_korean(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hermes", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "도구 호출 기능을 갖춘 AI 어시스턴트" in out
    assert "예시:" in out
    assert "실행할 명령어" in out
    assert "버전을 표시하고 종료" in out
    assert "대화형 채팅 시작" in out


def test_prompt_provider_choice_fallback_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    selected = _prompt_provider_choice(["OpenRouter", "Nous Portal"], default=1)

    out = capsys.readouterr().out
    assert selected == 1
    assert "provider 선택:" in out


def test_model_flow_openrouter_cancel_is_localized(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key: "" if key == "OPENROUTER_API_KEY" else "")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "")

    _model_flow_openrouter({}, current_model="")

    out = capsys.readouterr().out
    assert "OpenRouter API key가 설정되어 있지 않아요." in out
    assert "발급 링크:" in out
    assert "취소했어요." in out


def test_model_flow_qwen_not_logged_in_is_localized(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.auth.get_qwen_auth_status",
        lambda: {
            "logged_in": False,
            "auth_file": "/tmp/qwen-auth.json",
            "error": "missing token",
        },
    )

    _model_flow_qwen_oauth({}, current_model="")

    out = capsys.readouterr().out
    assert "Qwen CLI OAuth에 로그인되어 있지 않아요." in out
    assert "실행: qwen auth qwen-oauth" in out
    assert "예상 자격 증명 파일 위치: /tmp/qwen-auth.json" in out
    assert "오류: missing token" in out


def test_build_web_ui_fatal_missing_npm_is_localized(monkeypatch, capsys, tmp_path):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "package.json").write_text("{}")

    monkeypatch.setattr("shutil.which", lambda _name: None)

    assert _build_web_ui(web_dir, fatal=True) is False

    out = capsys.readouterr().out
    assert "Web UI 프런트엔드가 빌드되지 않았고 npm도 사용할 수 없어요." in out
    assert "Node.js를 설치한 뒤 다음을 실행하세요" in out
