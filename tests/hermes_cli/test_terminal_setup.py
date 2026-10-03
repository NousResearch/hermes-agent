"""Tests for the safe classic-CLI ``hermes terminal-setup`` helper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def unix_input_backend(monkeypatch):
    monkeypatch.setattr("hermes_cli.terminal_setup.is_native_windows", lambda: False)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"WT_SESSION": "session-id", "TERM_PROGRAM": "vscode"}, "windows-terminal"),
        ({"VSCODE_PID": "123"}, "vscode"),
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm2"),
        ({"TERM_PROGRAM": "Apple_Terminal"}, "apple-terminal"),
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"TERM_PROGRAM": "WezTerm"}, "wezterm"),
        ({"TERM_PROGRAM": "ghostty"}, "ghostty"),
        ({}, "unknown"),
    ],
)
def test_detect_terminal(environment, expected, monkeypatch):
    from hermes_cli.terminal_setup import detect_terminal

    for name in (
        "WT_SESSION",
        "VSCODE_PID",
        "TERM_PROGRAM",
        "LC_TERMINAL",
        "TERM",
        "KITTY_WINDOW_ID",
        "TERM_EMULATOR",
        "GHOSTTY_RESOURCES_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert detect_terminal() == expected


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        ("kitty", Path(".config/kitty/kitty.conf")),
        ("ghostty", Path(".config/ghostty/config")),
    ],
)
def test_managed_config_path_uses_xdg_home(terminal, expected, monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import managed_config_path

    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.Path.home", lambda: tmp_path / "home"
    )

    assert managed_config_path(terminal) == xdg / expected.relative_to(".config")


def test_ghostty_uses_macos_application_support(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import managed_config_path

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("hermes_cli.terminal_setup.sys.platform", "darwin")
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.Path.home", lambda: tmp_path / "home"
    )

    assert (
        managed_config_path("ghostty")
        == tmp_path / "home/Library/Application Support/com.mitchellh.ghostty/config"
    )


@pytest.mark.parametrize("terminal", ["kitty", "ghostty"])
def test_update_managed_block_preserves_content_and_is_idempotent(terminal):
    from hermes_cli.terminal_setup import managed_block, update_managed_block

    original = "font_size 14\n# personal setting\n"
    updated = update_managed_block(original, terminal)

    assert updated.startswith(original)
    assert managed_block(terminal) in updated
    assert update_managed_block(updated, terminal) == updated


def test_update_managed_block_replaces_only_existing_managed_block():
    from hermes_cli.terminal_setup import managed_block, update_managed_block

    existing = "before\n# >>> hermes terminal-setup >>>\nstale\n# <<< hermes terminal-setup <<<\nafter\n"
    updated = update_managed_block(existing, "kitty")

    assert updated == f"before\n{managed_block('kitty')}\nafter\n"


def test_update_managed_block_refuses_an_unterminated_owned_block():
    from hermes_cli.terminal_setup import update_managed_block

    with pytest.raises(ValueError, match="markers"):
        update_managed_block("user setting\n# >>> hermes terminal-setup >>>\n", "kitty")


@pytest.mark.parametrize(
    "content",
    [
        "# <<< hermes terminal-setup <<<\n",
        "# >>> hermes terminal-setup >>>\n",
        "# <<< hermes terminal-setup <<<\n# >>> hermes terminal-setup >>>\n",
        "# >>> hermes terminal-setup >>>\n# >>> hermes terminal-setup >>>\n# <<< hermes terminal-setup <<<\n# <<< hermes terminal-setup <<<\n",
        "# >>> hermes terminal-setup >>>\n# <<< hermes terminal-setup <<<\n# >>> hermes terminal-setup >>>\n# <<< hermes terminal-setup <<<\n",
        "before # >>> hermes terminal-setup >>>\n# <<< hermes terminal-setup <<<\n",
    ],
)
def test_invalid_managed_markers_are_rejected_without_writing(
    monkeypatch, tmp_path, content
):
    from hermes_cli.terminal_setup import run_terminal_setup

    config = tmp_path / "xdg/kitty/kitty.conf"
    config.parent.mkdir(parents=True)
    config.write_text(content, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="markers"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert config.read_text(encoding="utf-8") == content


def test_relative_xdg_config_home_is_rejected_without_writing(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative-config")
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="absolute"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (tmp_path / "relative-config").exists()


def test_xdg_config_root_symlink_is_rejected_without_writing(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "xdg-link"
    root_link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root_link))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="config root.*symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (outside / "kitty").exists()


def test_default_config_root_symlink_is_rejected_without_writing(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / ".config").symlink_to(outside, target_is_directory=True)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("hermes_cli.terminal_setup.Path.home", lambda: home)
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="config root.*symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (outside / "kitty").exists()


def test_xdg_config_root_ancestor_symlink_is_rejected_without_writing(
    monkeypatch, tmp_path
):
    from hermes_cli.terminal_setup import run_terminal_setup

    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root_link / "declared"))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="config root.*symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (outside / "declared").exists()


def test_macos_ghostty_config_root_symlink_is_rejected_without_writing(
    monkeypatch, tmp_path
):
    from hermes_cli.terminal_setup import run_terminal_setup

    home = tmp_path / "home"
    config_root = home / "Library/Application Support/com.mitchellh.ghostty"
    outside = tmp_path / "outside"
    config_root.parent.mkdir(parents=True)
    outside.mkdir()
    config_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("hermes_cli.terminal_setup.Path.home", lambda: home)
    monkeypatch.setattr("hermes_cli.terminal_setup.sys.platform", "darwin")
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "ghostty")

    with pytest.raises(ValueError, match="config root.*symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (outside / "config").exists()


def test_symlinked_config_target_is_rejected_without_writing(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    xdg = tmp_path / "xdg"
    target = tmp_path / "outside.conf"
    target.write_text("font_size 14\n", encoding="utf-8")
    config = xdg / "kitty/kitty.conf"
    config.parent.mkdir(parents=True)
    config.symlink_to(target)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert target.read_text(encoding="utf-8") == "font_size 14\n"


def test_parent_symlink_escape_is_rejected_without_writing(monkeypatch, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    xdg = tmp_path / "xdg"
    outside = tmp_path / "outside"
    outside.mkdir()
    xdg.mkdir()
    (xdg / "kitty").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    with pytest.raises(ValueError, match="symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (outside / "kitty.conf").exists()


@pytest.mark.parametrize(
    ("terminal", "parent", "config"),
    [
        ("kitty", "kitty", "kitty.conf"),
        ("ghostty", "ghostty", "config"),
    ],
)
def test_in_root_config_parent_symlink_is_rejected_without_writing(
    monkeypatch, tmp_path, terminal, parent, config
):
    from hermes_cli.terminal_setup import run_terminal_setup

    xdg = tmp_path / "xdg"
    redirected = xdg / "redirected"
    xdg.mkdir()
    redirected.mkdir()
    (xdg / parent).symlink_to(redirected, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: terminal)

    with pytest.raises(ValueError, match="symlink"):
        run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    assert not (redirected / config).exists()


def test_dry_run_shows_proposed_change_without_writing(monkeypatch, capsys, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    config = tmp_path / "kitty.conf"
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_path", lambda _terminal: config
    )
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_root", lambda _terminal: tmp_path
    )

    run_terminal_setup(argparse.Namespace(dry_run=True, print_config=False))

    output = capsys.readouterr().out
    assert f"Would update {config}" in output
    assert "map shift+enter send_text all \\x1b[13;2u" in output
    assert not config.exists()


def test_print_emits_pasteable_config_without_writing(monkeypatch, capsys, tmp_path):
    from hermes_cli.terminal_setup import run_terminal_setup

    config = tmp_path / "kitty.conf"
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_path", lambda _terminal: config
    )
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_root", lambda _terminal: tmp_path
    )

    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=True))

    output = capsys.readouterr().out
    assert "Paste into" in output
    assert "map shift+enter send_text all \\x1b[13;2u" in output
    assert not config.exists()


def test_dry_run_and_print_are_compatible_and_do_not_write(
    monkeypatch, capsys, tmp_path
):
    from hermes_cli.terminal_setup import run_terminal_setup

    config = tmp_path / "ghostty/config"
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "ghostty")
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_path", lambda _terminal: config
    )
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_root", lambda _terminal: tmp_path
    )

    run_terminal_setup(argparse.Namespace(dry_run=True, print_config=True))

    output = capsys.readouterr().out
    assert "Would update" in output
    assert "Paste into" in output
    assert not config.exists()


def test_unavailable_secure_persistence_prints_guidance_without_writing(
    monkeypatch, capsys, tmp_path
):
    from hermes_cli import terminal_setup

    config = tmp_path / "xdg/kitty/kitty.conf"
    config.parent.mkdir(parents=True)
    config.write_text("font_size 14\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")
    monkeypatch.setattr(
        "hermes_cli.terminal_setup._secure_persistence_available", lambda: False
    )

    terminal_setup.run_terminal_setup(
        argparse.Namespace(dry_run=False, print_config=False)
    )

    output = capsys.readouterr().out
    assert config.read_text(encoding="utf-8") == "font_size 14\n"
    assert "Cannot safely update" in output
    assert "--print" in output


def test_secure_persistence_accepts_linux_replace_dir_fds_not_in_support_set(
    monkeypatch,
):
    """Linux exposes replace's dir fds in its signature, not supports_dir_fd."""
    from types import SimpleNamespace

    from hermes_cli import terminal_setup

    def replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        pass

    open_operation = object()
    mkdir_operation = object()
    unlink_operation = object()
    linux_os = SimpleNamespace(
        name="posix",
        O_DIRECTORY=object(),
        O_NOFOLLOW=object(),
        O_CREAT=object(),
        O_EXCL=object(),
        open=open_operation,
        mkdir=mkdir_operation,
        replace=replace,
        unlink=unlink_operation,
        fchmod=object(),
        fsync=object(),
        supports_dir_fd={open_operation, mkdir_operation, unlink_operation},
    )
    monkeypatch.setattr(terminal_setup, "os", linux_os)

    assert terminal_setup._secure_persistence_available()


def test_secure_persistence_requires_keyword_replace_dir_fds(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli import terminal_setup

    def replace(src, dst, src_dir_fd, dst_dir_fd, /):
        pass

    open_operation = object()
    mkdir_operation = object()
    unlink_operation = object()
    linux_os = SimpleNamespace(
        name="posix",
        O_DIRECTORY=object(),
        O_NOFOLLOW=object(),
        O_CREAT=object(),
        O_EXCL=object(),
        open=open_operation,
        mkdir=mkdir_operation,
        replace=replace,
        unlink=unlink_operation,
        fchmod=object(),
        fsync=object(),
        supports_dir_fd={open_operation, mkdir_operation, unlink_operation},
    )
    monkeypatch.setattr(terminal_setup, "os", linux_os)

    assert not terminal_setup._secure_persistence_available()


def test_secure_persistence_does_not_follow_parent_swapped_at_write_seam(
    monkeypatch, tmp_path
):
    """A parent replaced by a symlink at persistence cannot redirect the write."""
    import os

    from hermes_cli import terminal_setup

    if not terminal_setup._secure_persistence_available():
        pytest.skip("descriptor-relative no-follow persistence is Unix-only")

    xdg = tmp_path / "xdg"
    target_parent = xdg / "kitty"
    outside = tmp_path / "outside"
    displaced = tmp_path / "displaced-kitty"
    target_parent.mkdir(parents=True)
    outside.mkdir()
    config = target_parent / "kitty.conf"
    config.write_text("font_size 14\n", encoding="utf-8")
    swapped = False
    original_replace = os.replace

    def swap_parent_before_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped and dst == "kitty.conf" and dst_dir_fd is not None:
            target_parent.rename(displaced)
            target_parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "replace", swap_parent_before_replace)

    terminal_setup._secure_update_managed_config(config, "kitty")

    assert swapped
    assert not (outside / "kitty.conf").exists()
    assert "font_size 14" in (displaced / "kitty.conf").read_text(encoding="utf-8")
    assert terminal_setup.managed_block("kitty") in (
        displaced / "kitty.conf"
    ).read_text(encoding="utf-8")


def test_default_managed_setup_writes_once_and_preserves_user_content(
    monkeypatch, capsys, tmp_path
):
    import stat

    from hermes_cli import terminal_setup

    if not terminal_setup._secure_persistence_available():
        pytest.skip("descriptor-relative no-follow persistence is Unix-only")

    run_terminal_setup = terminal_setup.run_terminal_setup

    xdg = tmp_path / "xdg"
    config = xdg / "kitty/kitty.conf"
    config.parent.mkdir(parents=True)
    config.write_text("font_size 14\n", encoding="utf-8")
    config.chmod(0o640)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "kitty")

    args = argparse.Namespace(dry_run=False, print_config=False)
    run_terminal_setup(args)
    first = config.read_text(encoding="utf-8")
    run_terminal_setup(args)

    assert "font_size 14" in first
    assert first.count("# >>> hermes terminal-setup >>>") == 1
    assert config.read_text(encoding="utf-8") == first
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    assert "Updated" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("terminal", "snippet"),
    [
        ("wezterm", "wezterm.action.SendString '\\x1b[13;2u'"),
        ("iterm2", "[13;2u"),
        ("windows-terminal", '"action": "sendInput"'),
        ("vscode", '"command": "workbench.action.terminal.sendSequence"'),
    ],
)
def test_print_includes_supported_terminal_snippets(
    monkeypatch, capsys, terminal, snippet
):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: terminal)
    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=True))

    assert snippet in capsys.readouterr().out


def test_apple_terminal_and_unknown_explain_fallback(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.setattr(
        "hermes_cli.terminal_setup.detect_terminal", lambda: "apple-terminal"
    )
    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))
    apple = capsys.readouterr().out
    assert "cannot distinguish Shift+Enter" in apple
    assert "Option+Enter" in apple

    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: "unknown")
    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))
    assert "Option+Enter" in capsys.readouterr().out


def test_native_windows_says_app_side_shift_enter_is_unavailable(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.setattr(
        "hermes_cli.terminal_setup.detect_terminal", lambda: "windows-terminal"
    )
    monkeypatch.setattr("hermes_cli.terminal_setup.is_native_windows", lambda: True)

    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    output = capsys.readouterr().out
    assert "prompt_toolkit Win32 backend" in output
    assert "Ctrl+Enter" in output
    assert "WSL or a remote Unix host" in output
    assert "sendInput" not in output


@pytest.mark.parametrize("terminal", ["kitty", "ghostty"])
def test_native_windows_never_writes_managed_mappings(
    monkeypatch, capsys, terminal, tmp_path
):
    from hermes_cli.terminal_setup import run_terminal_setup

    config = tmp_path / terminal / "config"
    monkeypatch.setattr("hermes_cli.terminal_setup.detect_terminal", lambda: terminal)
    monkeypatch.setattr("hermes_cli.terminal_setup.is_native_windows", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_path", lambda _terminal: config
    )
    monkeypatch.setattr(
        "hermes_cli.terminal_setup.managed_config_root", lambda _terminal: tmp_path
    )

    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=False))

    output = capsys.readouterr().out
    assert "prompt_toolkit Win32 backend" in output
    assert "map shift+enter" not in output
    assert not config.exists()


def test_terminal_setup_does_not_launch_external_processes(monkeypatch):
    from hermes_cli.terminal_setup import run_terminal_setup

    def fail(*_args, **_kwargs):
        raise AssertionError("terminal-setup must not invoke external processes")

    monkeypatch.setattr("subprocess.run", fail)
    monkeypatch.setattr("os.system", fail)
    run_terminal_setup(argparse.Namespace(dry_run=False, print_config=True))


def test_terminal_setup_parser_registers_flags_and_handler():
    from hermes_cli.subcommands.terminal_setup import build_terminal_setup_parser

    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    handler = lambda _args: None
    build_terminal_setup_parser(subparsers, cmd_terminal_setup=handler)

    args = root.parse_args(["terminal-setup", "--dry-run", "--print"])
    assert args.command == "terminal-setup"
    assert args.dry_run is True
    assert args.print_config is True
    assert args.func is handler


def test_terminal_setup_dispatches_through_real_top_level_parser(monkeypatch):
    import hermes_cli.main as main

    called = []
    monkeypatch.setattr(main, "cmd_terminal_setup", lambda args: called.append(args))

    parser, _ = main._build_cli_parser()
    args = parser.parse_args(["terminal-setup", "--print"])
    args.func(args)

    assert called and called[0].print_config is True
