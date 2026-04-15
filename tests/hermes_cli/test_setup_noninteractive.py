"""Tests for non-interactive setup and first-run headless behavior."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hermes_cli.config import DEFAULT_CONFIG, load_config, save_config


def _make_setup_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", False),
        section=overrides.get("section", None),
        reset=overrides.get("reset", False),
    )


def _make_chat_args(**overrides):
    return Namespace(
        continue_last=overrides.get("continue_last", None),
        resume=overrides.get("resume", None),
        model=overrides.get("model", None),
        provider=overrides.get("provider", None),
        toolsets=overrides.get("toolsets", None),
        verbose=overrides.get("verbose", False),
        query=overrides.get("query", None),
        worktree=overrides.get("worktree", False),
        yolo=overrides.get("yolo", False),
        pass_session_id=overrides.get("pass_session_id", False),
        quiet=overrides.get("quiet", False),
        checkpoints=overrides.get("checkpoints", False),
    )


class TestNonInteractiveSetup:
    """Verify setup paths exit cleanly in headless/non-interactive environments."""

    def test_noninteractive_guidance_is_localized_to_korean(self, capsys):
        from hermes_cli.setup import print_noninteractive_setup_guidance

        print_noninteractive_setup_guidance("TTY를 사용할 수 없습니다.")
        out = capsys.readouterr().out

        assert "⚕ Hermes 설정 — 비대화형 모드" in out
        assert "대화형 마법사는 여기서 사용할 수 없습니다." in out
        assert "환경 변수 또는 config 명령으로 Hermes를 설정하세요:" in out

    def test_prompt_choice_fallback_is_localized_to_korean(self, capsys):
        from hermes_cli.setup import prompt_choice

        with (
            patch("hermes_cli.setup._curses_prompt_choice", return_value=-1),
            patch("builtins.input", side_effect=["2"]),
        ):
            idx = prompt_choice("Provider를 선택하세요", ["첫 번째", "두 번째"], 0)

        out = capsys.readouterr().out
        assert idx == 1
        assert "Provider를 선택하세요" in out
        assert "기본값은 Enter" in out

    def test_prompt_yes_no_error_message_is_localized_to_korean(self, capsys):
        from hermes_cli.setup import prompt_yes_no

        with patch("builtins.input", side_effect=["maybe", "y"]):
            value = prompt_yes_no("계속할까요?", default=True)

        out = capsys.readouterr().out
        assert value is True
        assert "'y' 또는 'n'을 입력하세요" in out

    def test_setup_source_contains_korean_section_headers(self):
        source = Path("hermes_cli/setup.py").read_text(encoding="utf-8")

        assert 'print_header("모델 및 Provider")' in source
        assert 'print_header("터미널 백엔드")' in source
        assert 'print_header("에이전트 설정")' in source
        assert 'print_header("빠른 설정 — 누락된 항목만")' in source

    def test_setup_source_contains_korean_intro_copy(self):
        source = Path("hermes_cli/setup.py").read_text(encoding="utf-8")

        assert '주요 채팅 모델에 연결하는 방법을 선택하세요.' in source
        assert 'Hermes가 셸 명령과 코드를 어디서 실행할지 선택하세요.' in source
        assert 'Hermes Agent 설치를 함께 설정해볼게요.' in source
        assert '다시 오신 것을 환영합니다!' in source
        assert '무엇을 하시겠어요?' in source
        assert '도구 사용 가능 요약' in source
        assert '도구 API 키' in source
        assert '어떤 도구를 설정하시겠어요?' in source
        assert '어떤 플랫폼을 설정하시겠어요?' in source
        assert '모든 항목이 설정되어 있습니다! 할 일이 없습니다.' in source
        assert '필수 설정이 누락되었습니다:' in source
        assert '메시징 플랫폼' in source
        assert '어디서든 Hermes와 대화할 수 있도록 메시징 플랫폼에 연결하세요.' in source
        assert '메시징 플랫폼을 연결할까요? (Telegram, Discord 등)' in source
        assert '설정이 완료되었습니다! 바로 사용할 수 있어요.' in source
        assert '모든 설정 구성:' in source
        assert '비전 (이미지 분석)' in source
        assert '브라우저 자동화' in source
        assert '텍스트 음성 변환' in source
        assert '웹 검색 및 추출' in source
        assert '이제 사용할 준비가 되었습니다!' in source
        assert '설정을 수정하려면:' in source
        assert '텍스트 음성 변환 provider (선택 사항)' in source
        assert 'TTS provider를 선택하세요:' in source
        assert '터미널 백엔드를 선택하세요:' in source
        assert 'Modal 실행 과금 방식을 선택하세요:' in source
        assert '컨테이너 리소스 설정:' in source
        assert '세션 간 파일시스템을 유지할까요? (yes/no)' in source
        assert '현재 백엔드 유지:' in source
        assert '터미널 백엔드: Local' in source
        assert '터미널 백엔드: Docker' in source
        assert '터미널 백엔드: Singularity/Apptainer' in source
        assert '터미널 백엔드: Modal' in source
        assert '메시징 세션의 작업 디렉터리:' in source
        assert '터미널 백엔드: Daytona' in source
        assert '터미널 백엔드: SSH' in source
        assert '원격 머신에서 SSH로 명령어를 실행합니다.' in source
        assert 'Daytona 클라우드 개발 환경입니다.' in source
        assert 'SSH 호스트 (hostname 또는 IP)' in source

    def test_cmd_setup_allows_noninteractive_flag_without_tty(self):
        """The CLI entrypoint should not block --non-interactive before setup.py handles it."""
        from hermes_cli.main import cmd_setup

        args = _make_setup_args(non_interactive=True)

        with (
            patch("hermes_cli.setup.run_setup_wizard") as mock_run_setup,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            cmd_setup(args)

        mock_run_setup.assert_called_once_with(args)

    def test_cmd_setup_defers_no_tty_handling_to_setup_wizard(self):
        """Bare `hermes setup` should reach setup.py, which prints headless guidance."""
        from hermes_cli.main import cmd_setup

        args = _make_setup_args(non_interactive=False)

        with (
            patch("hermes_cli.setup.run_setup_wizard") as mock_run_setup,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            cmd_setup(args)

        mock_run_setup.assert_called_once_with(args)

    def test_non_interactive_flag_skips_wizard(self, capsys):
        """--non-interactive should print guidance and not enter the wizard."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=True)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_no_tty_skips_wizard(self, capsys):
        """When stdin has no TTY, the setup wizard should print guidance and return."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=False)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_reset_flag_rewrites_config_before_noninteractive_exit(self, tmp_path, monkeypatch, capsys):
        """--reset should rewrite config.yaml even when the wizard cannot run interactively."""
        from hermes_cli.setup import run_setup_wizard

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = load_config()
        cfg["model"] = {"provider": "custom", "base_url": "http://localhost:8080/v1", "default": "llama3"}
        cfg["agent"]["max_turns"] = 12
        save_config(cfg)

        args = _make_setup_args(non_interactive=True, reset=True)

        run_setup_wizard(args)

        reloaded = load_config()
        assert reloaded["model"] == DEFAULT_CONFIG["model"]
        assert reloaded["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
        out = capsys.readouterr().out
        assert "Configuration reset to defaults." in out

    def test_chat_first_run_headless_skips_setup_prompt(self, capsys):
        """Bare `hermes` should not prompt for input when no provider exists and stdin is headless."""
        from hermes_cli.main import cmd_chat

        args = _make_chat_args()

        with (
            patch("hermes_cli.main._has_any_provider_configured", return_value=False),
            patch("hermes_cli.main.cmd_setup") as mock_setup,
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc:
                cmd_chat(args)

        assert exc.value.code == 1
        mock_setup.assert_not_called()
        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_returning_user_terminal_menu_choice_dispatches_terminal_section(self, tmp_path):
        """Returning-user menu should map Terminal Backend to the terminal setup, not TTS."""
        from hermes_cli import setup as setup_mod

        args = _make_setup_args()
        config = {}
        model_section = MagicMock()
        tts_section = MagicMock()
        terminal_section = MagicMock()
        gateway_section = MagicMock()
        tools_section = MagicMock()
        agent_section = MagicMock()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value=config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(setup_mod, "prompt_choice", return_value=3),
            patch.object(
                setup_mod,
                "SETUP_SECTIONS",
                [
                    ("model", "모델 및 Provider", model_section),
                    ("tts", "텍스트 음성 변환", tts_section),
                    ("terminal", "터미널 백엔드", terminal_section),
                    ("gateway", "메시징 플랫폼 (Gateway)", gateway_section),
                    ("tools", "도구", tools_section),
                    ("agent", "에이전트 설정", agent_section),
                ],
            ),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        terminal_section.assert_called_once_with(config)
        tts_section.assert_not_called()

    def test_returning_user_menu_does_not_show_separator_rows(self, tmp_path):
        """Returning-user menu should only show selectable actions."""
        from hermes_cli import setup as setup_mod

        args = _make_setup_args()
        captured = {}

        def fake_prompt_choice(question, choices, default=0):
            captured["question"] = question
            captured["choices"] = list(choices)
            return len(choices) - 1

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(setup_mod, "prompt_choice", side_effect=fake_prompt_choice),
        ):
            setup_mod.run_setup_wizard(args)

        assert captured["question"] == "무엇을 하시겠어요?"
        assert "---" not in captured["choices"]
        assert captured["choices"] == [
            "빠른 설정 - 누락된 항목만 설정",
            "전체 설정 - 모든 항목 다시 설정",
            "모델 및 Provider",
            "터미널 백엔드",
            "메시징 플랫폼 (Gateway)",
            "도구",
            "에이전트 설정",
            "종료",
        ]

    def test_main_accepts_tts_setup_section(self, monkeypatch):
        """`hermes setup tts` should parse and dispatch like other setup sections."""
        from hermes_cli import main as main_mod

        received = {}

        def fake_cmd_setup(args):
            received["section"] = args.section

        monkeypatch.setattr(main_mod, "cmd_setup", fake_cmd_setup)
        monkeypatch.setattr("sys.argv", ["hermes", "setup", "tts"])

        main_mod.main()

        assert received["section"] == "tts"
