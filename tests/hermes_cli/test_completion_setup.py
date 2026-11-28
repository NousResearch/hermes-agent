"""Behavior and real-import I/O coverage for completion setup (#157893)."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from hermes_cli import completion
from hermes_cli.main import _build_cli_parser


SHELLS = ("bash", "zsh", "fish")
GENERATORS = dict(zip(SHELLS, (completion.generate_bash, completion.generate_zsh,
                             completion.generate_fish)))
START = b"# >>> hermes completion >>>"
END = b"# <<< hermes completion <<<"


@pytest.fixture
def shell_home(tmp_path, monkeypatch):
    home = tmp_path / "home with ' quote $() `backtick` \\ slash"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "agent-state"))
    for name in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "ZDOTDIR"):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def parser(shell_home):
    return _build_cli_parser()[0]


def invoke(parser, *options):
    args = parser.parse_args(["completion", *options])
    return args.func(args)


def snapshot(home):
    return {str(path.relative_to(home)): path.read_bytes()
            for path in home.rglob("*") if path.is_file()}


def installed_paths(home, shell):
    return {
        "bash": (home / ".local/share/bash-completion/completions/hermes", home / ".bashrc"),
        "zsh": (home / ".local/share/zsh/completions/_hermes", home / ".zshrc"),
        "fish": (home / ".config/fish/completions/hermes.fish", None),
    }[shell]


@pytest.mark.parametrize("shell", SHELLS)
def test_raw_and_legacy_are_exact_generator_output(parser, shell_home, capsys, shell):
    before = snapshot(shell_home)
    for options in (("--shell", shell, "--raw"), (shell,), (shell, "--raw"),
                    (shell, "--shell", shell, "--raw")):
        assert invoke(parser, *options) in (None, 0)
        captured = capsys.readouterr()
        assert captured.out == GENERATORS[shell](parser)
        assert not captured.err
        assert snapshot(shell_home) == before


@pytest.mark.parametrize("shell", SHELLS)
def test_manual_is_read_only_and_provides_static_setup(parser, shell_home, capsys, shell):
    (shell_home / ".bashrc").write_bytes(b"# unrelated\r\n")
    before = snapshot(shell_home)
    assert invoke(parser, "--shell", shell, "--manual") in (None, 0)
    output = capsys.readouterr().out
    assert "--raw" in output
    assert "mkdir" in output
    assert "source" in output
    assert "eval" not in output
    assert snapshot(shell_home) == before


@pytest.mark.parametrize("shell", SHELLS)
def test_install_static_idempotent_and_preserves_unrelated_bytes(parser, shell_home, capsys, shell):
    script, rc = installed_paths(shell_home, shell)
    original = b"# keep comments and bytes \xff\r\nexport KEPT=1\r\n# final without newline"
    if rc:
        rc.write_bytes(original)
    assert invoke(parser, "--shell", shell, "--install") in (None, 0)
    output = capsys.readouterr().out
    assert str(script) in output
    assert script.read_text() == GENERATORS[shell](parser)
    if rc:
        assert rc.read_bytes().startswith(original)
        assert rc.read_bytes().count(START) == 1
        assert rc.read_bytes().count(END) == 1
        assert b"eval" not in rc.read_bytes()
    else:
        assert not (shell_home / ".config/fish/config.fish").exists()
    before = snapshot(shell_home)
    mtimes = {p: p.stat().st_mtime_ns for p in (script, rc) if p}
    assert invoke(parser, shell, "--install") in (None, 0)
    assert snapshot(shell_home) == before
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes


@pytest.mark.parametrize("options", [(), ("--raw",), ("--install",), ("--manual",),
                                     ("--shell", "bash")])
@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, False), (True, False), (False, True)])
def test_missing_choices_fail_without_menus(parser, monkeypatch, capsys, options,
                                           stdin_tty, stdout_tty):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select",
                        lambda *a, **kw: pytest.fail("non-TTY must not open menus"))
    assert invoke(parser, *options) != 0
    output = capsys.readouterr()
    assert not output.out
    assert "--shell" in output.err or "--install" in output.err


@pytest.mark.parametrize("options", [("bash", "--shell", "fish", "--raw"),
                                     ("--shell", "bash", "--manual", "--install"),
                                     ("--shell", "fish", "--raw", "--manual")])
def test_conflicts_fail_clearly(parser, capsys, options):
    try:
        result = invoke(parser, *options)
    except SystemExit as exc:
        result = exc.code
    assert result != 0
    assert capsys.readouterr().err


@pytest.mark.parametrize("choices", [(None,), (0, None), (0, 0, None)])
def test_interactive_cancel_writes_nothing(parser, shell_home, monkeypatch, choices):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(choices)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", lambda *a, **kw: next(answers))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **kw: next(answers))
    before = snapshot(shell_home)
    assert invoke(parser) in (None, 0)
    assert snapshot(shell_home) == before


@pytest.mark.parametrize("shell_index", range(3))
def test_interactive_install_shows_plan_before_consent(parser, shell_home, monkeypatch,
                                                     capsys, shell_index):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter((shell_index, 0, 0))
    calls = []

    def choose(title, items, **kwargs):
        calls.append((title, items))
        if len(calls) == 3:
            script, rc = installed_paths(shell_home, SHELLS[shell_index])
            text = kwargs.get("description", "")
            assert text, "preview must survive the curses screen clear on the consent screen"
            assert str(script) in text
            if rc:
                assert str(rc) in text
                assert "source" in text
            assert not script.exists()
        return next(answers)

    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", choose)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", choose)
    assert invoke(parser) in (None, 0)
    assert installed_paths(shell_home, SHELLS[shell_index])[0].exists()


def test_explicit_shell_interactive_manual(parser, shell_home, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    calls = []

    def choose(title, items, **kwargs):
        calls.append(title)
        return 1

    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", choose)
    assert invoke(parser, "--shell", "zsh") in (None, 0)
    assert len(calls) == 1
    assert snapshot(shell_home) == {}


@pytest.mark.parametrize("shell", SHELLS)
def test_xdg_and_zdotdir(parser, shell_home, monkeypatch, shell):
    data, config, zdot = (shell_home / part for part in ("custom data", "custom config", "z dot"))
    for key, value in (("XDG_DATA_HOME", data), ("XDG_CONFIG_HOME", config), ("ZDOTDIR", zdot)):
        monkeypatch.setenv(key, str(value))
    assert invoke(parser, "--shell", shell, "--install") in (None, 0)
    expected = {"bash": data / "bash-completion/completions/hermes",
                "zsh": data / "zsh/completions/_hermes",
                "fish": config / "fish/completions/hermes.fish"}[shell]
    assert expected.read_text() == GENERATORS[shell](parser)
    if shell == "zsh":
        assert (zdot / ".zshrc").exists()
        assert not (shell_home / ".zshrc").exists()


@pytest.mark.parametrize("broken", [START + b"\n", END + b"\n", END + b"\n" + START,
                                    START + b"\n" + START + b"\n" + END,
                                    b"prefix " + START + b"\n" + END + b"\n"])
def test_malformed_markers_fail_before_any_write(parser, shell_home, capsys, broken):
    (shell_home / ".bashrc").write_bytes(broken)
    before = snapshot(shell_home)
    assert invoke(parser, "--shell", "bash", "--install") != 0
    assert snapshot(shell_home) == before
    error = capsys.readouterr().err
    assert "marker" in error.lower()
    assert ".bashrc" in error


@pytest.mark.platforms("posix")
def test_symlinked_rc_preserved_and_resolved_destination_reported(parser, shell_home, capsys):
    target = shell_home / "dotfiles/bashrc"
    target.parent.mkdir()
    target.write_bytes(b"# keep me\n")
    rc = shell_home / ".bashrc"
    rc.symlink_to(target)
    assert invoke(parser, "--shell", "bash", "--install") in (None, 0)
    assert rc.is_symlink()
    assert target.read_bytes().startswith(b"# keep me\n")
    assert START in target.read_bytes()
    assert str(target) in capsys.readouterr().out


def shell_binary(shell):
    binary = shutil.which(shell)
    if not binary:
        pytest.skip(f"real {shell} executable not available")
    return binary


def shell_command(binary, shell, code):
    # Extracted distributions need their own function/module search roots, not /usr.
    prefix = Path(binary).resolve().parent.parent
    functions = prefix / "share/zsh/functions"
    if shell == "zsh" and functions.is_dir():
        modules = sorted((prefix / "lib/zsh").glob("*"))
        code = (f"fpath=({shlex.quote(str(functions))}/**/*(N/) $fpath); "
                f"module_path=({' '.join(shlex.quote(str(p)) for p in modules)} $module_path); " + code)
    return [binary, "-c", code]


def manual_sections(manual):
    save = manual.split("Run these commands to save a static script:\n", 1)[1]
    save = save.split("Add this initialization to ", 1)[0].split("Start a new shell,", 1)[0]
    activation = manual.split("Start a new shell, or activate in the current shell:\n", 1)[1]
    activation = activation.split("After updating Hermes,", 1)[0]
    return save, activation


@pytest.fixture
def completion_env(shell_home, tmp_path) -> dict[str, str]:
    state, runtime, temp = (tmp_path / name for name in ("cli-state", "cli-runtime", "cli-tmp"))
    for path in (state, runtime, temp):
        path.mkdir()
    (state / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    return {
        "HOME": str(shell_home), "USERPROFILE": str(shell_home),
        "HERMES_HOME": str(state), "HERMES_RUNTIME_DIR": str(runtime),
        "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp),
        "HERMES_DISABLE_LAZY_INSTALLS": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ["PATH"], "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }


def completion_cli(env, *options):
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "completion", *options],
                          env=env, capture_output=True, timeout=60, check=False)


def registration_check(shell):
    return {
        "bash": 'test "$?" -eq 0 || exit 8\ncomplete -p hermes || exit 9\n'
                'COMP_WORDS=(hermes pro); COMP_CWORD=1; _hermes_completion\n'
                'printf "%s\\n" "${COMPREPLY[@]}"',
        "zsh": 'test "$?" -eq 0 || exit 8\n'
               '[[ "${_comps[hermes]-}" == _hermes ]] || exit 9\n'
               '(( $+functions[_hermes] )) || exit 10\n'
               'print -r -- $_comps[hermes]',
        "fish": 'test $status -eq 0; or exit 8\ncomplete -C "hermes pro"',
    }[shell]


def run_shell(binary, shell, commands, env):
    result = subprocess.run(shell_command(binary, shell, commands), env=env, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    return result.stdout


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", ("bash", "zsh"))
@pytest.mark.parametrize("layout", ("append", "unterminated-comment", "replace"))
def test_install_crlf_comments_remains_sourceable(shell_home, completion_env, shell, layout):
    binary = shell_binary(shell)
    _, rc = installed_paths(shell_home, shell)
    prefix = b"# existing harmless comment\r\n# non-UTF8 comment \xff\r\n"
    suffix = b"# after \xfe\r\n# final without newline"
    original = prefix
    if layout == "replace":
        original += START + b"\r\n# old completion\r\n" + END + b"\r\n" + suffix
    elif layout == "unterminated-comment":
        original += suffix
    rc.write_bytes(original)
    source = f"source {shlex.quote(str(rc))}\n"
    run_shell(binary, shell, source, completion_env)

    installed = completion_cli(completion_env, "--shell", shell, "--install")
    assert installed.returncode == 0, installed.stderr
    content = rc.read_bytes()
    if layout == "replace":
        assert content.split(START, 1)[0] == prefix
        assert content.split(END, 1)[1] == b"\n" + suffix
        assert b"old completion" not in content
    else:
        assert content.startswith(original)
    output = run_shell(binary, shell, source + registration_check(shell), completion_env)
    assert ("_hermes" if shell == "zsh" else "profile") in output


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell,overrides", [
    ("bash", ()), ("bash", ("XDG_DATA_HOME",)),
    ("zsh", ()), ("zsh", ("XDG_DATA_HOME",)), ("zsh", ("ZDOTDIR",)),
    ("zsh", ("XDG_DATA_HOME", "ZDOTDIR")),
    ("fish", ()), ("fish", ("XDG_CONFIG_HOME",)),
])
def test_manual_literal_newline_paths_copy_paste(parser, shell_home, tmp_path, completion_env,
                                                shell, overrides):
    binary = shell_binary(shell)
    home = shell_home / "nested ' quote $() \\ path\nnext"
    home.mkdir()
    env = dict(completion_env, HOME=str(home), USERPROFILE=str(home))
    for name in overrides:
        path = tmp_path / (name + " ' $() \\ path\nnext")
        path.mkdir()
        env[name] = str(path)
    data = Path(env.get("XDG_DATA_HOME", home / ".local/share"))
    config = Path(env.get("XDG_CONFIG_HOME", home / ".config"))
    script = {"bash": data / "bash-completion/completions/hermes",
              "zsh": data / "zsh/completions/_hermes",
              "fish": config / "fish/completions/hermes.fish"}[shell]
    launcher = tmp_path / "bin/hermes"
    launcher.parent.mkdir()
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m hermes_cli.main "$@"\n')
    launcher.chmod(0o755)
    env["PATH"] = str(launcher.parent) + os.pathsep + env["PATH"]

    manual = completion_cli(env, "--shell", shell, "--manual")
    assert manual.returncode == 0, manual.stderr
    assert not script.exists()
    save, activation = manual_sections(manual.stdout.decode("utf-8"))
    # Do not split, strip, dedent, or otherwise rewrite these printed shell sections.
    run_shell(binary, shell, save, env)
    raw = completion_cli(env, "--shell", shell, "--raw")
    assert raw.returncode == 0, raw.stderr
    assert script.read_bytes() == raw.stdout == GENERATORS[shell](parser).encode()
    # Fresh Fish must not auto-load a good path and mask a failed printed source.
    prelude = "complete -e -c hermes\n" if shell == "fish" else ""
    output = run_shell(binary, shell, prelude + activation + registration_check(shell), env)
    assert ("_hermes" if shell == "zsh" else "profile") in output
    if shell != "fish":
        rc = Path(env.get("ZDOTDIR", home)) / ".zshrc" if shell == "zsh" else home / ".bashrc"
        initialization = manual.stdout.decode("utf-8").split(START.decode() + "\n", 1)[1]
        initialization = initialization.split(END.decode(), 1)[0]
        rc.write_text(initialization)
        output = run_shell(binary, shell,
                           f"source {shlex.quote(str(rc))}\n" + registration_check(shell), env)
        assert ("_hermes" if shell == "zsh" else "profile") in output


def test_module_entrypoint_propagates_usage_failure(completion_env):
    result = completion_cli(completion_env, "--raw")
    assert result.returncode == 1
    assert not result.stdout
    assert b"--raw requires --shell" in result.stderr


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", SHELLS)
def test_real_shell_activation_and_manual_commands(parser, shell_home, capsys, shell):
    binary = shell_binary(shell)
    assert invoke(parser, "--shell", shell, "--manual") in (None, 0)
    manual = capsys.readouterr().out
    # Copy complete command sections verbatim, including quoted path newlines.
    save_section, activate_section = manual_sections(manual)
    commands = save_section + activate_section
    launcher = shell_home / "bin/hermes"
    launcher.parent.mkdir()
    # This is a real import and argparse dispatch, not a mocked generator.
    launcher.write_text(f"#!{sys.executable}\nimport sys\nfrom hermes_cli.main import _build_cli_parser\n"
                        "p, _ = _build_cli_parser()\na = p.parse_args(sys.argv[1:])\nsys.exit(a.func(a))\n")
    launcher.chmod(0o755)
    env = dict(os.environ, PATH=str(launcher.parent) + os.pathsep + os.environ["PATH"],
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    manual_result = subprocess.run(shell_command(binary, shell, commands), env=env, capture_output=True,
                                   text=True, timeout=60, check=False)
    assert manual_result.returncode == 0, manual_result.stderr
    assert not manual_result.stderr
    script, rc = installed_paths(shell_home, shell)
    assert script.read_text() == GENERATORS[shell](parser)
    assert invoke(parser, "--shell", shell, "--install") in (None, 0)
    check = {
        "bash": 'source "$HOME/.bashrc"; complete -p hermes; '
                'COMP_WORDS=(hermes pro); COMP_CWORD=1; _hermes_completion; printf "%s\\n" "${COMPREPLY[@]}"',
        "zsh": 'source "$HOME/.zshrc"; print -r -- $_comps[hermes]; '
               '(( $+functions[_hermes] )) || exit 9; '
               'autoload +X _hermes; print -r -- $_comps[hermes]; '
               'compinit() { print -u2 -- unexpected-compinit; return 1; }; '
               'source "$HOME/.zshrc"; print -r -- $_comps[hermes]',
        "fish": 'complete -C "hermes pro"',
    }[shell]
    syntax = subprocess.run([binary, "-n", str(script)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert syntax.returncode == 0, syntax.stderr
    if rc:
        syntax = subprocess.run([binary, "-n", str(rc)], env=env, capture_output=True, text=True, timeout=30, check=False)
        assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(shell_command(binary, shell, check), env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert ("_hermes" if shell == "zsh" else "profile") in result.stdout
    if shell == "zsh":
        assert result.stdout.splitlines() == ["_hermes", "_hermes", "_hermes"]


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("mode", ("raw", "manual", "install"))
def test_real_import_subprocess_dispatch(shell_home, tmp_path, shell, mode):
    code = ("import sys; from hermes_cli.main import _build_cli_parser; "
            "p, _ = _build_cli_parser(); a = p.parse_args(sys.argv[1:]); sys.exit(a.func(a))")
    env = dict(os.environ, HOME=str(shell_home), USERPROFILE=str(shell_home),
               HERMES_HOME=str(tmp_path / "subprocess-state"),
               HERMES_RUNTIME_DIR=str(tmp_path / "runtime"))
    before = snapshot(shell_home)
    result = subprocess.run([sys.executable, "-c", code, "completion", "--shell", shell, "--" + mode],
                            env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert not result.stderr
    if mode != "install":
        assert snapshot(shell_home) == before
    else:
        assert installed_paths(shell_home, shell)[0].exists()


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("mode", ("raw", "manual", "install"))
def test_fully_explicit_never_prompts_even_on_a_terminal(parser, monkeypatch, shell, mode):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select",
                        lambda *a, **kw: pytest.fail("explicit modes must never prompt"))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist",
                        lambda *a, **kw: pytest.fail("explicit modes must never prompt"))
    assert invoke(parser, "--shell", shell, "--" + mode) in (None, 0)


def test_raw_missing_shell_fails_even_on_a_terminal(parser, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert invoke(parser, "--raw") != 0
    output = capsys.readouterr()
    assert not output.out
    assert "--shell" in output.err


@pytest.mark.parametrize("shell", ("bash", "zsh"))
def test_refreshes_managed_block_without_changing_surroundings(parser, shell_home, shell):
    _, rc = installed_paths(shell_home, shell)
    before = b"# before \xff\r\n"
    after = b"# after\r\nexport KEEP=1"
    rc.write_bytes(before + START + b"\r\n# old completion\r\n" + END + b"\r\n" + after)
    assert invoke(parser, "--shell", shell, "--install") in (None, 0)
    content = rc.read_bytes()
    assert content.startswith(before)
    assert content.endswith(after)
    assert b"old completion" not in content
    assert content.count(START) == content.count(END) == 1


@pytest.mark.platforms("posix")
def test_overlapping_symlink_destinations_fail_without_clobber(parser, shell_home, capsys):
    script, rc = installed_paths(shell_home, "bash")
    script.parent.mkdir(parents=True)
    rc.write_bytes(b"# unrelated config\n")
    script.symlink_to(rc)
    before = snapshot(shell_home)
    assert invoke(parser, "--shell", "bash", "--install") != 0
    assert snapshot(shell_home) == before
    assert script.is_symlink()
    assert "same file" in capsys.readouterr().err


def test_edit_during_confirmation_is_not_overwritten(parser, shell_home, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    script, rc = installed_paths(shell_home, "bash")

    def confirm(title, items, **kwargs):
        if title.startswith("Apply"):
            rc.write_bytes(b"# external edit\n")
        return 0

    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", confirm)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", confirm)
    assert invoke(parser, "--shell", "bash") != 0
    assert rc.read_bytes() == b"# external edit\n"
    assert not script.exists()
    assert "changed while planning" in capsys.readouterr().err
