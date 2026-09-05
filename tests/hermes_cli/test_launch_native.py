from pathlib import Path
import stat

import pytest

from hermes_cli.launch_native import native_tui_argv, native_tui_bin, wants_native


def _exe(tmp_path: Path, name: str = "hermes-tui-native") -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_wants_native_env(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_NATIVE", "1")
    assert wants_native(argv=["--tui"]) is True


def test_wants_native_flag(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_NATIVE", raising=False)
    assert wants_native(argv=["--tui", "--native"]) is True
    assert wants_native(argv=["--tui"]) is False


def test_native_tui_bin_explicit(tmp_path):
    bin_path = _exe(tmp_path)
    env = {"HERMES_TUI_NATIVE_BIN": str(bin_path)}
    assert native_tui_bin(env) == str(bin_path)


def test_native_tui_bin_missing_explicit(tmp_path):
    env = {"HERMES_TUI_NATIVE_BIN": str(tmp_path / "missing")}
    assert native_tui_bin(env) is None


def test_native_tui_argv_none_when_not_requested():
    assert native_tui_argv({}) is None


def test_native_tui_argv_skips_dashboard(tmp_path):
    bin_path = _exe(tmp_path)
    env = {
        "HERMES_TUI_NATIVE": "1",
        "HERMES_TUI_NATIVE_BIN": str(bin_path),
        "HERMES_TUI_DASHBOARD": "1",
    }
    assert native_tui_argv(env) is None


def test_native_tui_argv_resume_and_title(tmp_path):
    bin_path = _exe(tmp_path)
    env = {
        "HERMES_TUI_NATIVE": "1",
        "HERMES_TUI_NATIVE_BIN": str(bin_path),
        "HERMES_TUI_RESUME": "sess-1",
        "HERMES_TUI_TITLE": "Gold",
    }
    assert native_tui_argv(env) == [
        str(bin_path),
        "--resume",
        "sess-1",
        "--title",
        "Gold",
    ]


def test_native_tui_argv_fallback_when_bin_missing(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_NATIVE", "1")
    monkeypatch.delenv("HERMES_TUI_NATIVE_BIN", raising=False)
    monkeypatch.setattr("hermes_cli.launch_native.shutil.which", lambda *a, **k: None)
    assert native_tui_argv({"HERMES_TUI_NATIVE": "1"}) is None
    err = capsys.readouterr().err
    assert "hermes-tui-native was not found" in err
    assert "Falling back to the Ink TUI" in err


def test_parser_accepts_native():
    from hermes_cli._parser import build_top_level_parser

    parser, _sub, chat_parser = build_top_level_parser()
    args = parser.parse_args(["--tui", "--native"])
    assert args.native is True
    chat_args = chat_parser.parse_args(["--native"])
    assert chat_args.native is True


def test_launch_tui_uses_native_after_resume_env(monkeypatch, tmp_path):
    import hermes_cli.main_tui_launch as launch_mod

    captured = {}
    bin_path = _exe(tmp_path)
    monkeypatch.setenv("HERMES_TUI_NATIVE", "1")
    monkeypatch.setenv("HERMES_TUI_NATIVE_BIN", str(bin_path))
    monkeypatch.setenv("HERMES_TUI_RESUME", "stale-missing-session")

    def boom(*_a, **_k):
        raise AssertionError("Ink argv must not be built for native launch")

    monkeypatch.setattr(launch_mod, "_make_tui_argv", boom)
    monkeypatch.setattr(
        launch_mod.subprocess,
        "call",
        lambda argv, cwd=None, env=None: captured.update(
            {"argv": argv, "cwd": cwd, "env": env}
        )
        or 1,
    )

    with pytest.raises(SystemExit):
        launch_mod._launch_tui(resume_session_id="session-good-1")

    assert captured["argv"][0] == str(bin_path)
    assert captured["argv"][captured["argv"].index("--resume") + 1] == (
        "session-good-1"
    )
    assert captured["env"]["HERMES_TUI_RESUME"] == "session-good-1"


def test_launch_tui_falls_back_to_ink_without_binary(monkeypatch):
    import hermes_cli.main_tui_launch as launch_mod

    captured = {}
    monkeypatch.setenv("HERMES_TUI_NATIVE", "1")
    monkeypatch.delenv("HERMES_TUI_NATIVE_BIN", raising=False)
    monkeypatch.setattr("hermes_cli.launch_native.shutil.which", lambda *a, **k: None)
    monkeypatch.setattr(
        launch_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(
        launch_mod.subprocess,
        "call",
        lambda argv, cwd=None, env=None: captured.update({"argv": argv}) or 1,
    )

    with pytest.raises(SystemExit):
        launch_mod._launch_tui()

    assert captured["argv"] == ["node", "dist/entry.js"]


def test_launch_tui_dev_stays_on_ink(monkeypatch, tmp_path):
    import hermes_cli.main_tui_launch as launch_mod

    captured = {}
    bin_path = _exe(tmp_path)
    monkeypatch.setenv("HERMES_TUI_NATIVE", "1")
    monkeypatch.setenv("HERMES_TUI_NATIVE_BIN", str(bin_path))
    monkeypatch.setattr(
        launch_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["tsx", "src/entry.tsx"], Path(".")),
    )
    monkeypatch.setattr(
        launch_mod.subprocess,
        "call",
        lambda argv, cwd=None, env=None: captured.update({"argv": argv}) or 1,
    )

    with pytest.raises(SystemExit):
        launch_mod._launch_tui(tui_dev=True)

    assert captured["argv"] == ["tsx", "src/entry.tsx"]
