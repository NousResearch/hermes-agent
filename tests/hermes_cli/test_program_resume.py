"""ProgramResumeReporter: one pane's OSC 7577 record (hermes_cli/program_resume.py)."""

import base64

import os

import pytest

from hermes_cli.program_resume import ProgramResumeReporter

_TOKEN = "dummy-token-replace-me"


def _argv(seq: str) -> list[str]:
    body = seq[len("\x1b]7577;") : -2]
    part = next(p for p in body.split(":") if p.startswith("argv="))
    return [base64.b64decode(item).decode() for item in part[len("argv=") :].split(",")]


@pytest.fixture(autouse=True)
def _no_claimed_token(monkeypatch):
    from hermes_cli import terminal_notify

    monkeypatch.setattr(terminal_notify, "_claimed_osc7577_token", None)


@pytest.fixture(autouse=True)
def _stock_hermes_root(monkeypatch):
    """The profiles live under ~/.hermes (no custom root): the tests' HERMES_HOME is a temp dir."""
    from pathlib import Path

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: Path.home() / ".hermes")


@pytest.fixture
def written(monkeypatch):
    monkeypatch.setenv("OSC7577_TOKEN", _TOKEN)
    return []


@pytest.fixture(autouse=True)
def _fresh_resume_prefix():
    from hermes_cli import program_resume

    program_resume._cached_prefix.cache_clear()
    yield
    program_resume._cached_prefix.cache_clear()


def _resume_env(monkeypatch, *, profile="default", argv0="/usr/local/bin/hermes", bin_path="/usr/local/bin/hermes"):
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.relaunch as relaunch_mod
    from hermes_cli import program_resume

    program_resume._cached_prefix.cache_clear()
    monkeypatch.setattr("sys.argv", [argv0])
    monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: bin_path)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: profile)
    # get_active_profile_name says "custom" for a home outside the profiles root, unless a profile has that name.
    monkeypatch.setattr(profiles_mod, "profile_matches_home", lambda name, home=None: name != "custom")
    for name in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return program_resume


def test_resume_argv_pins_the_profile_and_surface(monkeypatch):
    pr = _resume_env(monkeypatch)
    assert pr.resume_argv(None, "cli") == ("/usr/local/bin/hermes", "-p", "default", "--cli")
    assert pr.resume_argv("sess-1", "cli") == ("/usr/local/bin/hermes", "-p", "default", "--cli", "--resume", "sess-1")

    pr = _resume_env(monkeypatch, profile="work")
    assert pr.resume_argv("sess-2", "tui", "--tui-native", "--dev") == (
        "/usr/local/bin/hermes", "-p", "work", "--tui", "--tui-native", "--dev", "--resume", "sess-2"
    )


def test_resume_argv_names_a_profile_called_custom(monkeypatch):
    import hermes_cli.profiles as profiles_mod

    pr = _resume_env(monkeypatch, profile="custom")
    monkeypatch.setattr(profiles_mod, "profile_matches_home", lambda name, home=None: True)
    assert pr.resume_argv("s", "cli") == ("/usr/local/bin/hermes", "-p", "custom", "--cli", "--resume", "s")


def test_bypass_launch_repeats_its_flag_and_reports_only_a_saved_session(monkeypatch):
    pr = _resume_env(monkeypatch)
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    assert pr.resume_argv(None, "cli") is None  # restarted bare it needs its explicit model back
    assert pr.resume_argv("s", "cli") == ("/usr/local/bin/hermes", "-p", "default", "--cli", "--safe-mode", "--resume", "s")

    pr = _resume_env(monkeypatch)
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    assert pr.resume_argv("s", "tui")[-3:] == ("--ignore-user-config", "--resume", "s")


def test_resume_argv_uses_the_install_command_for_source_runs_and_bad_launchers(monkeypatch):
    from hermes_cli.chat_launch import hermes_command

    install = tuple(hermes_command())  # this install's launcher or inline bootstrap: no PYTHONPATH needed
    assert "-m" not in install
    pr = _resume_env(monkeypatch, argv0="/src/hermes/hermes_cli/main.py")
    assert pr.resume_argv("s", "cli")[:len(install)] == install
    pr = _resume_env(monkeypatch, bin_path="./relative-hermes")
    assert pr.resume_argv("s", "cli")[:len(install)] == install


def test_resume_argv_is_none_when_no_command_reaches_this_home(monkeypatch):
    import hermes_cli.profiles as profiles_mod

    pr = _resume_env(monkeypatch, profile="custom")
    assert pr.resume_argv("s", "cli") is None

    def _boom():
        raise OSError("unreadable home")

    pr = _resume_env(monkeypatch)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", _boom)
    assert pr.resume_argv("s", "cli") is None


class _Store:
    def __init__(self, saved=(), error=None):
        self.saved, self.error = set(saved), error

    def get_session(self, session_id):
        if self.error:
            raise self.error
        return {"id": session_id} if session_id in self.saved else None


def test_gateway_chat_view_reports_once_saved_and_never_when_finite(written, monkeypatch):
    from hermes_cli.gateway_chat_view import GatewayChatView

    from hermes_cli import terminal_notify

    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _resume_env(monkeypatch)

    class _DummyClient:
        pass

    import hermes_cli.gateway_session_store as store_mod

    store = _Store()
    monkeypatch.setattr(store_mod, "session_store", lambda: store)
    view = GatewayChatView(_DummyClient(), {"stored_session_id": "gw-1", "info": {"lazy": False, "title": "t"}},
                           program_resume=True)
    view._sync_program_resume()
    assert _argv(written[-1]) == ["/usr/local/bin/hermes", "-p", "default", "--cli"]

    store.saved.add("gw-1")  # the store now holds its row
    view._sync_program_resume()
    assert _argv(written[-1]) == ["/usr/local/bin/hermes", "-p", "default", "--cli", "--resume", "gw-1"]

    view._clear_program_resume()
    assert ":clear=1:mac=" in written[-1]

    before_finite = len(written)
    finite_view = GatewayChatView(_DummyClient(), {"stored_session_id": "gw-2", "info": {"lazy": False}},
                                  program_resume=True)
    finite_view.finite = True
    finite_view._sync_program_resume()
    assert len(written) == before_finite


def test_reports_once_per_command_and_again_when_forced(written):
    reporter = ProgramResumeReporter(written.append)
    reporter.sync(("hermes", "--cli"))
    reporter.sync(("hermes", "--cli"))
    assert [_argv(s) for s in written] == [["hermes", "--cli"]]
    reporter.sync(("hermes", "--cli"), force=True)
    reporter.sync(("hermes", "--cli", "--resume", "s1"))
    assert [_argv(s) for s in written[1:]] == [["hermes", "--cli"], ["hermes", "--cli", "--resume", "s1"]]


def test_unreportable_command_clears_the_earlier_record(written):
    reporter = ProgramResumeReporter(written.append)
    reporter.sync(("hermes", "--cli"))
    reporter.sync(("hermes", "--cli", "--resume", "bad\nid"))
    assert ":clear=1:mac=" in written[-1]
    reporter.sync(("hermes", "--cli", "--resume", "bad\nid"))
    assert len(written) == 2  # nothing left to clear
    assert not reporter.reported


def test_clear_sends_once_and_closes(written):
    reporter = ProgramResumeReporter(written.append)
    reporter.clear()
    assert written == []  # nothing reported, nothing to clear
    reporter = ProgramResumeReporter(written.append)
    reporter.sync(("hermes", "--cli"))
    reporter.clear()
    reporter.clear()
    reporter.sync(("hermes", "--cli"))
    assert len(written) == 2 and ":clear=1:mac=" in written[1]


def test_keep_closes_without_clearing(written):
    reporter = ProgramResumeReporter(written.append)
    reporter.sync(("hermes", "--cli"))
    reporter.keep()
    reporter.clear()
    reporter.sync(("hermes", "--tui"))
    assert len(written) == 1
    assert not reporter.active


def test_silent_without_a_token(monkeypatch):
    monkeypatch.delenv("OSC7577_TOKEN", raising=False)
    written = []
    reporter = ProgramResumeReporter(written.append)
    reporter.sync(("hermes", "--cli"))
    reporter.clear()
    assert written == [] and not reporter.active


def test_gateway_chat_view_honours_the_opt_out(written, monkeypatch):
    from hermes_cli import terminal_notify
    from hermes_cli.gateway_chat_view import GatewayChatView

    _resume_env(monkeypatch)
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    snapshot = {"stored_session_id": "gw-1", "messages": [{"role": "user"}]}
    GatewayChatView(object(), snapshot)._sync_program_resume()
    assert written == []  # program_resume defaults off: gateway_chat passes the launch's setting

    GatewayChatView(object(), snapshot, program_resume=True)._sync_program_resume()
    assert _argv(written[-1]) == ["/usr/local/bin/hermes", "-p", "default", "--cli", "--resume", "gw-1"]


def test_program_resume_enabled_reads_the_token_and_setting(written, monkeypatch):
    from argparse import Namespace

    import hermes_cli.config_effective as effective
    from hermes_cli import gateway_chat

    for name in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(effective, "load_user_config_effective", lambda path: {"display": {"program_resume": False}})
    assert gateway_chat.program_resume_enabled(Namespace()) is False
    assert gateway_chat.program_resume_enabled(Namespace(safe_mode=True)) is True  # reads no profile: the default

    monkeypatch.delenv("OSC7577_TOKEN")
    assert gateway_chat.program_resume_enabled(Namespace(safe_mode=True)) is False


def test_a_config_file_opt_out_writes_no_report(written, monkeypatch, tmp_path):
    from argparse import Namespace

    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView

    for name in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display:\n  program_resume: false\n", encoding="utf-8")
    enabled = gateway_chat.program_resume_enabled(Namespace())  # the real loader reads the file
    GatewayChatView(object(), {"stored_session_id": "gw-1", "messages": [{"role": "user"}]},
                    program_resume=enabled)._sync_program_resume()
    assert enabled is False and written == []


def _launch_tui_env(monkeypatch, *, profile):
    import sys
    from pathlib import Path

    import hermes_cli.main as main_mod
    import hermes_cli.profiles as profiles_mod
    from hermes_cli import main_tui_launch, program_resume

    program_resume._cached_prefix.cache_clear()
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: profile)
    monkeypatch.setattr(profiles_mod, "profile_matches_home", lambda name, home=None: name != "custom")
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    captured = {}
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
                        lambda tui_dir, tui_dev, notices=None: (["node", "dist/entry.js"], Path(".")))
    monkeypatch.setattr(main_mod.subprocess, "call", lambda argv, cwd=None, env=None: captured.update(env=env) or 1)
    with pytest.raises(SystemExit):
        main_mod._launch_tui()
    program_resume._cached_prefix.cache_clear()
    return captured["env"]


def test_tui_launch_hands_the_token_only_with_a_reachable_resume_command(written, monkeypatch):
    env = _launch_tui_env(monkeypatch, profile="work")
    assert env["OSC7577_TOKEN"] == _TOKEN
    assert '"-p", "work", "--tui"' in env["HERMES_TUI_RESUME_ARGV"]

    assert "HERMES_TUI_RESUME_SESSION_ONLY" not in env

    env = _launch_tui_env(monkeypatch, profile="custom")
    assert "OSC7577_TOKEN" not in env
    assert "HERMES_TUI_RESUME_ARGV" not in env and "HERMES_TUI_OWNER_PID" not in env

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    monkeypatch.setenv("HERMES_MODEL", "test/model")  # a bypass launch names its model
    env = _launch_tui_env(monkeypatch, profile="work")
    assert env["HERMES_TUI_RESUME_SESSION_ONLY"] == "1"
    assert '"--tui", "--safe-mode"' in env["HERMES_TUI_RESUME_ARGV"]


def test_tui_launch_without_a_token_resolves_nothing(monkeypatch):
    from hermes_cli import program_resume

    monkeypatch.delenv("OSC7577_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(program_resume, "_profile_args", lambda: calls.append(1) or ["-p", "default"])
    env = _launch_tui_env(monkeypatch, profile="default")
    assert calls == []
    assert "HERMES_TUI_RESUME_ARGV" not in env and "HERMES_TUI_OWNER_PID" not in env


def test_session_saved_separates_no_row_from_unreadable():
    from hermes_cli.program_resume import session_saved

    class _Store:
        def __init__(self, row=None, error=None):
            self.row, self.error = row, error

        def get_session(self, session_id):
            if self.error:
                raise self.error
            return self.row

    assert session_saved(_Store(row={"id": "s"}), "s") is True
    assert session_saved(_Store(), "s") is False
    assert session_saved(_Store(error=OSError("locked")), "s") is False
    assert session_saved(None, "s") is False


# Same table as ui-tui/src/__tests__/programResume.test.ts ("argv[0] rules match the Python reporter").
ARGV0_CASES = [
    ("/usr/local/bin/hermes", True),
    ("hermes", True),
    ("C:\\Program Files\\hermes.exe", True),
    ("c:/hermes/hermes.exe", True),
    ("\\\\server\\share\\hermes.exe", True),
    ("\\hermes.exe", False),
    ("C:hermes.exe", True),  # no separator: a bare command name
    ("./hermes", False),
    ("bin/hermes", False),
    ("-hermes", False),
    ("", False),
    ("/usr/bin/her\x1bmes", False),
]


@pytest.mark.parametrize(("arg", "valid"), ARGV0_CASES)
def test_argv0_rules_match_the_tui(arg, valid):
    from hermes_cli.terminal_notify import _is_valid_osc7577_argv0

    assert _is_valid_osc7577_argv0(arg) is valid


@pytest.mark.platforms("posix")
def test_gateway_chat_view_reports_again_after_sigcont(written, monkeypatch):
    import asyncio
    import signal

    from hermes_cli.gateway_chat_view import GatewayChatView

    calls = []
    view = GatewayChatView(object(), {"stored_session_id": "gw-1", "messages": [{"role": "user"}]})
    view.program_resume_enabled = True
    monkeypatch.setattr(view, "_sync_program_resume", lambda *, force=False: calls.append(force))

    async def _run():
        from tests.waiting import async_wait_until

        view._resync_program_resume_on_sigcont()
        try:
            signal.raise_signal(signal.SIGCONT)
            await async_wait_until(lambda: calls, timeout=5)
        finally:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGCONT)

    asyncio.run(_run())
    assert calls == [True]



@pytest.fixture
def classic(written, monkeypatch):
    """A HermesCLI with only what the OSC 7577 paths read, reporting into *written*."""
    import cli as cli_module
    from cli import HermesCLI
    from hermes_cli import terminal_notify

    _resume_env(monkeypatch)
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    for name in ("_invoke_interrupted_session_end", "_run_cleanup"):
        monkeypatch.setattr(cli_module, name, lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_signal_watchdog_armed", False)
    cli = HermesCLI.__new__(HermesCLI)
    cli.program_resume_enabled = cli._program_resume_interactive = True
    cli.session_id, cli._session_db, cli.agent = "s1", _Store(), None
    cli._should_exit = cli._viewer_disconnected = cli._terminal_io_broken = False
    cli._pending_relaunch, cli._voice_recorder, cli._program_status_closed = None, None, True
    for hook in ("_pet_stop_anim", "_persist_active_session_before_close", "_print_exit_summary",
                 "_release_active_session"):
        setattr(cli, hook, lambda: None)
    return cli


def test_classic_reports_resume_only_for_a_saved_row_even_when_resumed(classic, written):
    classic._resumed = True  # --resume of an id is not proof of its row
    classic._sync_program_resume()
    assert _argv(written[-1]) == ["/usr/local/bin/hermes", "-p", "default", "--cli"]
    classic._session_db.saved.add("s1")
    classic._sync_program_resume()
    assert _argv(written[-1])[-2:] == ["--resume", "s1"]


@pytest.mark.parametrize(("how", "cleared"), [
    ("user quit", True),
    ("exception unwinding app.run", False),
    ("SIGHUP/SIGTERM", False),
    ("terminal output broken", False),
    ("/update relaunch", False),
])
def test_classic_shutdown_clears_only_on_a_user_quit(classic, written, monkeypatch, how, cleared):
    import cli as cli_module

    classic._sync_program_resume()
    classic._should_exit = how != "exception unwinding app.run"
    if how == "SIGHUP/SIGTERM":
        monkeypatch.setattr(cli_module, "_signal_watchdog_armed", True)
    elif how == "terminal output broken":
        classic._terminal_io_broken = True
    elif how == "/update relaunch":
        classic._pending_relaunch = ["update"]
    classic._tui_shutdown()
    assert (":clear=1:mac=" in written[-1]) is cleared
    classic._clear_program_resume()  # once kept or cleared, the reporter is closed
    assert len(written) == 1 + cleared


def test_classic_sigcont_reports_again(classic, written):
    from tests.waiting import wait_until

    classic._sync_program_resume()
    classic._tui_continue_handler(None, None)
    wait_until(lambda: len(written) == 2, timeout=5)
    assert _argv(written[1]) == _argv(written[0])  # same command, reported afresh


def test_gateway_chat_ctrl_d_on_an_empty_line_is_a_quit():
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from hermes_cli.gateway_chat_view import _quit_on_ctrl_d

    def read(keys):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return PromptSession(key_bindings=_quit_on_ctrl_d()).prompt()

    assert read("\x04") == "/quit"  # Ctrl+D on an empty line
    assert read("ab\x1b[D\x04\r") == "a"  # Ctrl+D with text deletes forward, as before


def test_reporter_orders_a_threaded_report_before_the_users_clear(monkeypatch):
    import threading

    from hermes_cli.program_resume import ProgramResumeReporter

    monkeypatch.setenv("OSC7577_TOKEN", _TOKEN)
    written, entered, release = [], threading.Event(), threading.Event()

    def write(seq):
        if not written:
            entered.set()
            release.wait(5)
        written.append(seq)

    reporter = ProgramResumeReporter(write)
    report = threading.Thread(target=reporter.sync, args=(["/usr/local/bin/hermes", "--cli"],), kwargs={"force": True})
    report.start()  # a SIGCONT re-report, mid-write
    entered.wait(5)
    quit_ = threading.Thread(target=reporter.clear)
    quit_.start()
    release.set()
    report.join(5)
    quit_.join(5)
    assert len(written) == 2 and ":clear=1:mac=" in written[-1]


def test_a_data_dir_suffix_root_reports_nothing(monkeypatch):
    from pathlib import Path

    import hermes_constants

    pr = _resume_env(monkeypatch)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: Path.home() / ".hermes-canary")
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "-canary")
    assert pr.resume_argv("sess-1", "cli") is None  # a fresh -p default opens ~/.hermes, not ~/.hermes-canary


def test_reporter_resolves_a_callable_target_under_its_lock(monkeypatch):
    from hermes_cli.program_resume import ProgramResumeReporter

    monkeypatch.setenv("OSC7577_TOKEN", _TOKEN)
    written = []
    reporter = ProgramResumeReporter(written.append)
    held = []
    reporter.sync(lambda: held.append(reporter._lock.locked()) or ["/usr/local/bin/hermes", "--cli"])
    assert held == [True] and len(written) == 1


def test_a_named_gateway_reports_nothing(monkeypatch):
    pr = _resume_env(monkeypatch)
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://10.0.0.5:9119/ws")
    assert pr.resume_argv("sess-1", "cli") is None and pr.resume_argv_prefix("tui") is None


def test_the_env_load_claims_the_token_before_any_child_can_start(monkeypatch, tmp_path):
    from hermes_cli import env_loader, terminal_notify

    monkeypatch.setenv("OSC7577_TOKEN", _TOKEN)
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)  # every entry point's first env step
    assert "OSC7577_TOKEN" not in os.environ and terminal_notify.osc7577_token() == _TOKEN
    terminal_notify.restore_osc7577_token()  # the /update relaunch hands it to the new process
    assert os.environ["OSC7577_TOKEN"] == _TOKEN


def test_a_custom_hermes_root_reports_nothing(monkeypatch, tmp_path):
    import hermes_constants

    pr = _resume_env(monkeypatch)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)  # Docker's /opt/data
    assert pr.resume_argv("sess-1", "cli") is None  # its -p default would mean ~/.hermes in a fresh env


def test_gateway_chat_bypass_flags_come_from_its_args(written, monkeypatch):
    from argparse import Namespace

    from hermes_cli import terminal_notify
    from hermes_cli.gateway_chat_view import GatewayChatView

    pr = _resume_env(monkeypatch)  # a gateway chat bypass launch sets no HERMES_SAFE_MODE
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    flags = pr.bypass_flags(Namespace(safe_mode=True))
    assert flags == ("--safe-mode",)

    view = GatewayChatView(object(), {"stored_session_id": "gw-1"}, program_resume=True, resume_flags=flags)
    view._sync_program_resume()
    assert written == []  # unsaved: no bare form
    view._session_durable = True
    view._sync_program_resume()
    assert _argv(written[-1]) == ["/usr/local/bin/hermes", "-p", "default", "--cli", "--safe-mode", "--resume", "gw-1"]

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")  # flag and env together: one --safe-mode
    pr._cached_prefix.cache_clear()
    assert pr.resume_argv("s", "cli", *flags).count("--safe-mode") == 1
