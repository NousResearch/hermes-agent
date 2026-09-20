import threading
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from hermes_cli import anon_auth
from hermes_cli import cli_commands_mixin as commands


class _Thread:
    def __init__(self, target):
        self.target = target
        self.started = False

    def start(self):
        self.started = True

    def join(self):
        self.target()


def _cli(monkeypatch):
    cli = SimpleNamespace(console=MagicMock())
    workers = []

    def side_worker(produce, **kwargs):
        thread = _Thread(produce)
        workers.append((thread, kwargs))
        return thread

    cli._side_worker = side_worker
    cli._handle_login_command = commands.CLICommandsMixin._handle_login_command.__get__(cli)
    output = []
    monkeypatch.setattr(commands, "_cp", lambda *lines: output.extend(lines))
    return cli, workers, output


def test_the_cli_handler_prints_the_code_then_drains_off_thread(monkeypatch):
    cli, workers, output = _cli(monkeypatch)
    states = iter([
        anon_auth.Code("https://example.test/sign-in", "CODE-1", 900, 5),
        anon_auth.Completed(email="person@example.test", model="model-1", model_changed=True),
    ])
    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **_kwargs: states)
    monkeypatch.setattr(
        anon_auth, "render_sign_in_cli_code",
        lambda state, **kwargs: kwargs["printer"](state.link, state.code, f"  {state.copy_with_wait}"))

    cli._handle_login_command("/login")

    assert output == [
        "  Starting sign-in...",
        "https://example.test/sign-in",
        "CODE-1",
        "  Do not share this code. Waiting for sign-in, up to 15 minutes.",
    ]
    assert workers[0][0].started is True
    assert workers[0][0].target() == (
        "Signed in as person@example.test.\nDefault model is now model-1.")


@pytest.mark.parametrize(
    "terminal,initial_model,expected_model",
    [
        (anon_auth.Completed(model="model-1", model_changed=True), anon_auth.GUEST_MODEL, "model-1"),
        (anon_auth.Completed(model="model-1", model_changed=True),
         "openrouter/some-model", "openrouter/some-model"),
        (anon_auth.Declined(), anon_auth.GUEST_MODEL, anon_auth.GUEST_MODEL),
    ],
)
def test_the_drain_only_moves_the_free_tier_model_on_completion(
        monkeypatch, terminal, initial_model, expected_model):
    cli, workers, _output = _cli(monkeypatch)
    cli.model = initial_model
    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **_kwargs: iter([
        anon_auth.Code("https://example.test/sign-in", "CODE-1", 900, 5),
        terminal,
    ]))
    monkeypatch.setattr(anon_auth, "render_sign_in_cli_code", lambda *_args, **_kwargs: None)

    cli._handle_login_command("/login")
    workers[0][0].join()

    assert cli.model == expected_model


def test_a_precondition_prints_without_starting_a_thread(monkeypatch):
    cli, workers, output = _cli(monkeypatch)
    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **_kwargs: iter([anon_auth.AlreadySignedIn()]))

    cli._handle_login_command("/login")

    assert output == ["  Starting sign-in...", "  Already signed in."]
    assert workers == []


def test_ctrl_c_during_the_first_advance_prints_the_cancelled_copy(monkeypatch):
    cli, workers, output = _cli(monkeypatch)
    closed = threading.Event()

    def flow():
        try:
            raise KeyboardInterrupt
            yield
        finally:
            closed.set()

    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **_kwargs: flow())

    cli._handle_login_command("/login")

    assert output[-1] == anon_auth.UPGRADE_CANCELLED
    assert closed.is_set()
    assert workers == []


def test_the_handler_never_calls_input_and_uses_the_short_timeout(monkeypatch):
    cli, _workers, _output = _cli(monkeypatch)
    seen = []
    monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(AssertionError("input called")))

    def flow(**kwargs):
        seen.append(kwargs)
        return iter([anon_auth.AlreadySignedIn()])

    monkeypatch.setattr(anon_auth, "run_sign_in", flow)
    cli._handle_login_command("/login")

    assert seen == [{"timeout_seconds": 8.0}]


def test_the_in_chat_and_terminal_completion_use_their_own_copy():
    state = anon_auth.Completed(email="", model="", model_changed=True)
    assert "run /model to pick one" in anon_auth.drain_sign_in_copy(iter([state]), chat=True)
    assert "run `hermes model` to pick one" in anon_auth.drain_sign_in_copy(iter([state]), chat=False)


def test_the_command_resolves_through_the_cli_fallback():
    from cli import HermesCLI
    assert HermesCLI._slash_handler("login") == ("_handle_login_command", True)


def test_the_drain_writes_to_the_console_captured_at_start(monkeypatch):
    old_buf, new_buf = StringIO(), StringIO()
    gate = threading.Event()
    threads = []
    cli = SimpleNamespace(
        console=Console(file=old_buf, force_terminal=False, width=100),
        _app=None,
        bell_on_complete=False,
        final_response_markdown=False,
        _scrollback_box_width=lambda: 80,
        _invalidate=lambda **_kwargs: None,
    )
    real_side_worker = commands.CLICommandsMixin._side_worker.__get__(cli)

    def side_worker(*args, **kwargs):
        thread = real_side_worker(*args, **kwargs)
        threads.append(thread)
        return thread

    cli._side_worker = side_worker
    cli._handle_login_command = commands.CLICommandsMixin._handle_login_command.__get__(cli)
    monkeypatch.setattr(commands, "_cp", lambda *_lines: None)
    monkeypatch.setattr(anon_auth, "render_sign_in_cli_code", lambda *_args, **_kwargs: None)

    def flow(**_kwargs):
        yield anon_auth.Code("https://example.test/sign-in", "CODE", 60, 1)
        gate.wait()
        yield anon_auth.Completed(email="person@example.test")

    monkeypatch.setattr(anon_auth, "run_sign_in", flow)
    cli._handle_login_command("/login")
    cli.console = Console(file=new_buf, force_terminal=False, width=100)
    gate.set()
    threads[0].join(timeout=2)

    assert "Signed in as person@example.test." in old_buf.getvalue()
    assert new_buf.getvalue() == ""


def test_the_live_tui_drain_prints_through_cprint_instead_of_the_captured_console(monkeypatch):
    import cli as cli_module

    old_buf, new_buf = StringIO(), StringIO()
    gate = threading.Event()
    threads = []
    output = []
    cli = SimpleNamespace(
        console=Console(file=old_buf, force_terminal=False, width=100),
        _app=SimpleNamespace(invalidate=lambda: None),
        bell_on_complete=False,
        final_response_markdown=False,
        _scrollback_box_width=lambda: 80,
        _invalidate=lambda **_kwargs: None,
    )
    real_side_worker = commands.CLICommandsMixin._side_worker.__get__(cli)

    def side_worker(*args, **kwargs):
        thread = real_side_worker(*args, **kwargs)
        threads.append(thread)
        return thread

    cli._side_worker = side_worker
    cli._handle_login_command = commands.CLICommandsMixin._handle_login_command.__get__(cli)
    monkeypatch.setattr(commands, "_cp", lambda *lines: output.extend(lines))
    monkeypatch.setattr(cli_module, "_cprint", lambda *lines, **_kwargs: output.extend(lines))
    monkeypatch.setattr(anon_auth, "render_sign_in_cli_code", lambda *_args, **_kwargs: None)

    def flow(**_kwargs):
        yield anon_auth.Code("https://example.test/sign-in", "CODE", 60, 1)
        gate.wait()
        yield anon_auth.Completed(email="person@example.test")

    monkeypatch.setattr(anon_auth, "run_sign_in", flow)
    cli._handle_login_command("/login")
    cli.console = Console(file=new_buf, force_terminal=False, width=100)
    gate.set()
    threads[0].join(timeout=2)

    assert not threads[0].is_alive()
    assert old_buf.getvalue() == ""
    assert new_buf.getvalue() == ""
    assert "  Sign-in" in output
    assert any("Signed in as person@example.test." in line for line in output)


def test_upgrade_guest_keeps_the_terminal_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(anon_auth, "render_sign_in_cli", lambda **kwargs: seen.append(kwargs) or 0)
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: True)

    assert anon_auth.upgrade_guest(SimpleNamespace(timeout=None, no_browser=False)) == 0
    assert seen[0]["timeout_seconds"] == 15.0
    assert seen[0]["chat"] is False


def test_the_terminal_renderer_keeps_the_original_line_sequence(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: True)
    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **_kwargs: iter([
        anon_auth.Code("https://example.test/sign-in", "CODE-1", 900, 5),
        anon_auth.Waiting(),
        anon_auth.Completed(email="person@example.test", model="model-1", model_changed=True),
    ]))

    assert anon_auth.upgrade_guest(SimpleNamespace(timeout=None, no_browser=False)) == 0
    assert capsys.readouterr().out.splitlines() == [
        anon_auth.UPGRADE_START,
        "",
        "To continue:",
        "  1. Open: https://example.test/sign-in",
        "  2. If prompted, enter code: CODE-1",
        f"  {anon_auth.UPGRADE_DO_NOT_SHARE}",
        anon_auth.UPGRADE_WAITING,
        "Signed in as person@example.test.",
        "Default model is now model-1.",
    ]
