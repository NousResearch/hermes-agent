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


def _set_startup_config(
    monkeypatch, *, provider="custom", default="safe-model", base_url="", ack=None
):
    model = {"provider": provider, "default": default}
    if base_url:
        model["base_url"] = base_url
    config = {"model": model}
    if ack is not None:
        config["security"] = {
            "allow_data_training_tiers_noninteractive": ack,
        }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


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


def test_cmd_chat_allows_acknowledged_data_training_tier_noninteractively(
    main_mod, fake_cli, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    _set_startup_config(monkeypatch, ack=True)

    main_mod.cmd_chat(
        _chat_args(model="muse-spark-1.2-contributor", provider="custom")
    )

    assert fake_cli["model"] == "muse-spark-1.2-contributor"
    err = capsys.readouterr().err
    assert "TRAINS ON YOUR DATA" in err
    assert "security.allow_data_training_tiers_noninteractive" in err


def test_cmd_chat_rejects_unacknowledged_data_training_tier_with_opt_in_hint(
    main_mod, fake_cli, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    _set_startup_config(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(
            _chat_args(model="muse-spark-1.2-contributor", provider="custom")
        )

    assert excinfo.value.code == 1
    assert not fake_cli
    err = capsys.readouterr().err
    assert "TRAINS ON YOUR DATA" in err
    assert "security.allow_data_training_tiers_noninteractive" in err


def test_data_training_acknowledgement_does_not_bypass_cost_guard(
    main_mod, fake_cli, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    _set_startup_config(
        monkeypatch,
        provider="openai-codex",
        default="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        ack=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args(model="openai/gpt-5.5-pro"))

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "EXPENSIVE MODEL WARNING" in capsys.readouterr().err


def test_data_training_acknowledgement_requires_literal_true(
    main_mod, fake_cli, monkeypatch
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    _set_startup_config(monkeypatch, ack="true")

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(
            _chat_args(model="muse-spark-1.2-contributor", provider="custom")
        )

    assert excinfo.value.code == 1
    assert not fake_cli


def test_data_training_acknowledgement_does_not_skip_interactive_confirmation(
    main_mod, fake_cli, monkeypatch, capsys
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    _set_startup_config(monkeypatch, ack=True)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(
            _chat_args(model="muse-spark-1.2-contributor", provider="custom")
        )

    assert excinfo.value.code == 1
    assert not fake_cli
    assert "Model override cancelled" in capsys.readouterr().err


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


def test_top_level_oneshot_rejects_noninteractive_gpt55_pro_startup_override(
    main_mod, codex_config, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "stdin", _NonInteractiveStdin())
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "hello", "-m", "openai/gpt-5.5-pro"])
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)

    called = False

    def fake_run_and_exit_oneshot(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", fake_run_and_exit_oneshot)

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main()

    assert excinfo.value.code == 1
    assert called is False
    err = capsys.readouterr().err
    assert "EXPENSIVE MODEL WARNING" in err
    assert "non-interactive" in err
