"""Vim Space-leader keybindings (``display.vim_mode``) on the classic CLI and gateway composers.

Keys are driven through the application's real KeyProcessor (default Vi bindings included), so
binding priority, filters and the after-key hook are exercised exactly as in a live session.
"""

from types import SimpleNamespace

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import BufferControl, HSplit, Layout, Window
from prompt_toolkit.output import DummyOutput

from hermes_cli import cli_subagent_monitor
from hermes_cli.vim_leader import (
    GATEWAY_LEADER_ACTIONS,
    VimLeaderState,
    bind_vim_leader,
    cli_leader_actions,
    configure_gateway_prompt_session_kwargs,
    is_vim_mode_enabled,
)

ENTER = KeyPress(Keys.ControlM, "\r")


def _app(kb, text="existing draft", cursor=4, read_only=False):
    buf = Buffer(document=Document(text, cursor), read_only=read_only)
    other = Window(BufferControl(Buffer()))
    main = Window(BufferControl(buf))
    app = Application(layout=Layout(HSplit([main, other]), focused_element=main), key_bindings=kb,
                      editing_mode=EditingMode.VI, input=DummyInput(), output=DummyOutput())
    app.timeoutlen = None  # no flush timer: key processing stays synchronous
    app.vi_state.input_mode = InputMode.NAVIGATION
    return app, buf, other


def _press(app, *keys):
    with set_app(app):
        app.key_processor.feed_multiple([k if isinstance(k, KeyPress) else KeyPress(k) for k in keys])
        app.key_processor.process_keys()


def _armed(app, state):
    with set_app(app):
        return state.is_armed(app)


class _FakeCLI:
    def __init__(self):
        self.calls = []
        self._open_command_palette = lambda: self.calls.append("palette")
        self._tui_handle_open_in_editor = lambda event: self.calls.append("editor")
        self._tui_handle_ctrl_c = lambda event: self.calls.append("ctrl_c")
        self._tui_submit_buffer = lambda event, buf: self.calls.append(("submit", buf.text))


def _cli_app(monkeypatch, blocked=lambda: False, **app_kw):
    cli = _FakeCLI()
    monkeypatch.setattr(cli_subagent_monitor, "open_monitor", lambda c: cli.calls.append("monitor"))
    monkeypatch.setattr(cli_subagent_monitor, "toggle_dock", lambda c: cli.calls.append("dock"))
    kb = KeyBindings()
    pastes = []
    kb.add(Keys.BracketedPaste, eager=True)(lambda event: pastes.append(event.data))  # CLI paste path
    state = bind_vim_leader(kb, cli_leader_actions(cli), VimLeaderState(blocked))
    app, buf, other = _app(kb, **app_kw)
    return cli, state, app, buf, other, pastes


def test_cli_followers_route_to_existing_handlers_without_touching_draft(monkeypatch):
    cli, state, app, buf, _, _ = _cli_app(monkeypatch)
    for key, expected in (("p", "palette"), ("e", "editor"), ("t", "monitor"), ("r", "dock"),
                          ("c", "ctrl_c"), (ENTER, ("submit", "existing draft"))):
        _press(app, " ")
        assert _armed(app, state)
        _press(app, key)
        assert cli.calls[-1] == expected
        assert not _armed(app, state)
        assert (buf.text, buf.cursor_position) == ("existing draft", 4)


def test_cli_has_no_model_follower_and_unknown_or_modified_keys_cancel_without_edit(monkeypatch):
    cli, state, app, buf, _, _ = _cli_app(monkeypatch)
    for key in ("m", "x", "d", "i", " ", Keys.ControlX, Keys.Escape):
        _press(app, " ", key)
        assert not _armed(app, state)
        assert cli.calls == []
        assert (buf.text, buf.cursor_position) == ("existing draft", 4)
        assert app.vi_state.input_mode == InputMode.NAVIGATION
        assert app.vi_state.operator_func is None


def test_paste_cancels_leader_and_keeps_original_paste_path(monkeypatch):
    _, state, app, _, _, pastes = _cli_app(monkeypatch)
    _press(app, " ", KeyPress(Keys.BracketedPaste, "pasted"))
    assert pastes == ["pasted"]
    assert not _armed(app, state)

    kb = KeyBindings()
    state = bind_vim_leader(kb, GATEWAY_LEADER_ACTIONS, VimLeaderState())
    app, buf, _ = _app(kb, text="", cursor=0)
    _press(app, " ", KeyPress(Keys.BracketedPaste, "pasted"))
    assert buf.text == "pasted"  # prompt_toolkit's default paste handler still ran
    assert not _armed(app, state)


def test_focus_mode_and_modal_changes_reset_armed_state(monkeypatch):
    modal = {"open": False}
    _, state, app, _, other, _ = _cli_app(monkeypatch, blocked=lambda: modal["open"])

    _press(app, " ")
    app.layout.focus(other)
    assert not _armed(app, state)
    app.layout.focus_previous()
    assert not _armed(app, state)

    _press(app, " ")
    app.vi_state.input_mode = InputMode.INSERT
    assert not _armed(app, state)
    app.vi_state.input_mode = InputMode.NAVIGATION
    assert not _armed(app, state)

    _press(app, " ")
    modal["open"] = True
    assert not _armed(app, state)
    modal["open"] = False
    assert not _armed(app, state)  # no stale arming after the modal closes

    modal["open"] = True
    _press(app, " ")
    assert not _armed(app, state)


def test_space_does_not_arm_with_count_operator_visual_insert_or_read_only(monkeypatch):
    cli, state, app, buf, _, _ = _cli_app(monkeypatch)
    _press(app, "2", " ")  # count: Vi's Space motion runs instead
    assert not _armed(app, state) and buf.cursor_position == 6
    _press(app, "d", " ")  # operator pending: Space is the motion
    assert not _armed(app, state)
    _press(app, Keys.Escape, "v", " ")
    assert not _armed(app, state)
    _press(app, Keys.Escape, "i", " ")
    assert not _armed(app, state) and " " in buf.text
    assert cli.calls == []

    _, state, app, buf, _, _ = _cli_app(monkeypatch, read_only=True)
    _press(app, " ")
    assert not _armed(app, state)


def test_gateway_followers_prefill_only_empty_draft_and_never_rewrite_existing_text():
    kb = KeyBindings()
    state = bind_vim_leader(kb, GATEWAY_LEADER_ACTIONS, VimLeaderState())
    for key, prefill in (("p", "/"), ("m", "/model ")):
        app, buf, _ = _app(kb)
        _press(app, " ", key)
        assert (buf.text, buf.cursor_position) == ("existing draft", 4)
        assert not _armed(app, state)

        app, buf, _ = _app(kb, text="", cursor=0)
        _press(app, " ", key)
        assert (buf.text, buf.cursor_position) == (prefill, len(prefill))
        assert app.vi_state.input_mode == InputMode.INSERT

    app, buf, _ = _app(kb)
    _press(app, " ", "c")
    assert buf.text == ""

    app, buf, _ = _app(kb)
    submitted = []
    buf.accept_handler = lambda b: submitted.append(b.text) or True
    _press(app, " ", ENTER)
    assert submitted == ["existing draft"]

    app, buf, _ = _app(kb)
    opened = []
    buf.open_in_editor = lambda *a, **k: opened.append(buf.text)
    _press(app, " ", "e")
    assert opened == ["existing draft"]

    app, buf, _ = _app(kb)
    for key in ("t", "r", "x"):
        _press(app, " ", key)
        assert (buf.text, buf.cursor_position) == ("existing draft", 4)


def test_gateway_prompt_session_kwargs_follow_vim_mode():
    off = configure_gateway_prompt_session_kwargs("test-model", "test-session",
                                                  {"display": {"vim_mode": False}})
    assert off["erase_when_done"] is True
    assert "editing_mode" not in off and "key_bindings" not in off
    assert "test-model" in off["bottom_toolbar"]()

    on = configure_gateway_prompt_session_kwargs("test-model", "test-session",
                                                 {"display": {"vim_mode": True}})
    assert on["editing_mode"] == EditingMode.VI
    assert on["key_bindings"].bindings
    assert "[LEADER]" not in on["bottom_toolbar"]()
    assert not is_vim_mode_enabled({"display": "bad"})


def test_status_bar_vim_mode_label_reflects_leader():
    import cli as cli_mod

    kb = KeyBindings()
    leader = bind_vim_leader(kb, {}, VimLeaderState())
    app, _, _ = _app(kb)
    stub = SimpleNamespace(_vim_mode=True, _vim_leader_state=leader, _app=app)
    assert cli_mod.HermesCLI._vim_mode_label(stub) == "NORMAL"

    _press(app, " ")
    assert cli_mod.HermesCLI._vim_mode_label(stub) == "NORMAL [LEADER]"

    app.vi_state.input_mode = InputMode.INSERT
    assert cli_mod.HermesCLI._vim_mode_label(stub) == "INSERT"
