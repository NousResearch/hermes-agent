from argparse import Namespace
import sys
import types

import pytest


class _NonInteractiveStdin:
    def isatty(self):
        return False


def _chat_args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "no_restore_cwd": False,
        "toolsets": None,
        "skills": None,
        "tui": False,
        "tui_dev": False,
        "cli": True,
        "verbose": None,
        "quiet": True,
        "query": "hello",
        "image": None,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "safe_mode": False,
        "compact": False,
        "source": None,
        "yolo": False,
        "accept_hooks": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod


@pytest.fixture
def fake_cli(monkeypatch):
    captured = {}

    def fake_cli_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=fake_cli_main))
    return captured


@pytest.fixture
def codex_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "provider": "openai-codex",
                "default": "gpt-5.5",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        },
    )


def test_cmd_chat_rejects_noninteractive_gpt55_pro_startup_override(
    main_mod, fake_cli, codex_config, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro"))

    assert excinfo.value.code == 1
    assert not fake_cli
    err = capsys.readouterr().err
    assert "EXPENSIVE MODEL WARNING" in err
    assert "did you mean to select openai/gpt-5.5?" in err
    assert "non-interactive" in err


def test_cmd_chat_rejects_noninteractive_gpt55_pro_even_with_yolo(
    main_mod, fake_cli, codex_config, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro", yolo=True))

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "EXPENSIVE MODEL WARNING" in capsys.readouterr().err


def test_cmd_chat_allows_interactive_gpt55_pro_when_confirmed(
    main_mod, fake_cli, codex_config, monkeypatch
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro"))

    assert fake_cli["model"] == "openai/gpt-5.5-pro"


def test_cmd_chat_cancels_interactive_gpt55_pro_when_not_confirmed(
    main_mod, fake_cli, codex_config, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro"))

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "Model override cancelled" in capsys.readouterr().err


def test_cmd_chat_cancels_interactive_gpt55_pro_on_eof(
    main_mod, fake_cli, codex_config, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro"))

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "Model override cancelled" in capsys.readouterr().err


def test_cmd_chat_rejects_noninteractive_provider_only_override_when_default_is_expensive(
    main_mod, fake_cli, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openai-codex", "default": "openai/gpt-5.5-pro"}},
    )

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model=None, provider="nous"))

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "EXPENSIVE MODEL WARNING" in capsys.readouterr().err


def test_cmd_chat_allows_noninteractive_safe_codex_startup_override(
    main_mod, fake_cli, monkeypatch
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.5"}},
    )

    main_mod.cmd_chat(_chat_args(model="gpt-5.5", provider="openai-codex"))

    assert fake_cli["model"] == "gpt-5.5"
    assert fake_cli["provider"] == "openai-codex"
