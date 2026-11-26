"""display.bell_on_prompt / bell_on_complete also drive OSC 9 + Warp OSC 777 via _ring_bell."""

import json

import pytest

from cli import HermesCLI
from hermes_cli import terminal_notify

_WARP_OK = {
    "TERM_PROGRAM": "WarpTerminal",
    "WARP_CLI_AGENT_PROTOCOL_VERSION": "1",
    "WARP_CLIENT_VERSION": "v0.2026.08.01.00.00.stable_01",
}


def _ring(monkeypatch, *, flag_on, env, **kwargs):
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.bell_on_prompt = flag_on
    cli.session_id = "sess-1"
    cli._ring_bell(prompt=True, **kwargs)
    return "".join(written)


def test_osc9_body_emitted_and_sanitized_only_when_flag_on(monkeypatch):
    out = _ring(monkeypatch, flag_on=True, env={}, context="approval\x1b\x07\x00\x7f!")
    assert out == "\a\x1b]9;Hermes: approval!\x07"
    assert _ring(monkeypatch, flag_on=False, env={}, context="approval") == ""


def test_program_status_uses_st_and_encodes_safe_message():
    # Controls (C0, DEL, C1) collapse to one space: the spec drops a whole report whose msg holds one.
    seq = terminal_notify.program_status_sequence("blocked", kind="permission", message="Allow\ncommand?\x85 x")
    assert seq == "\x1b]7501;state=blocked:app=hermes:kind=permission:msg=QWxsb3cgY29tbWFuZD8geA==\x1b\\"


def test_program_status_rejects_invalid_state_and_drops_a_stray_kind():
    with pytest.raises(ValueError):
        terminal_notify.program_status_sequence("clear")
    # Same as the TUI encoder: a kind outside `blocked`, or an unknown one, is left out.
    assert terminal_notify.program_status_sequence("working", kind="permission") == (
        "\x1b]7501;state=working:app=hermes\x1b\\")
    assert terminal_notify.program_status_sequence("blocked", kind="bogus") == (
        "\x1b]7501;state=blocked:app=hermes\x1b\\")


def test_turn_outcome_matches_the_tui():
    """done never carries the reply; a failure that kept answer text is done, as in the TUI."""
    from hermes_cli.cli_chat_turn_mixin import _program_outcome

    assert _program_outcome(None) is None
    assert _program_outcome({"final_response": "the whole answer"}) == ("done", "")
    assert _program_outcome({"interrupted": True, "final_response": "half"}) == ("idle", "")
    assert _program_outcome({"failed": True, "error": "429"}) == ("error", "429")
    assert _program_outcome({"failed": True, "partial": True, "error": "429", "final_response": "429"}) == (
        "error", "429")
    assert _program_outcome({"failed": True, "partial": True, "error": "429", "final_response": "half"}) == (
        "done", "")


def test_program_status_follows_the_screen_through_one_owner(monkeypatch):
    """Prompt on screen -> blocked; prompt closed mid-turn -> working (no per-prompt bookkeeping);
    a connection card working in the background is not blocked; a lost result claims no outcome."""
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli = HermesCLI.__new__(HermesCLI)
    cli.program_status_enabled, cli._agent_running = True, True

    def states():
        out = [s.split(";", 1)[1].removesuffix("\x1b\\").split(":msg=")[0].replace(":app=hermes", "") for s in written]
        written.clear()
        return out

    cli._approval_state = {"command": "rm -rf build"}
    cli._sync_program_status()
    cli._approval_state = None
    cli._connection_state = {"phase": "waiting"}
    cli._sync_program_status()
    cli._connection_state = {"phase": "url"}
    cli._sync_program_status()
    cli._sync_program_status()  # unchanged: no repeat report
    cli._connection_state = None
    cli._slash_confirm_state = {"title": "Out of credits"}  # billing wall / slash confirm
    cli._sync_program_status()
    cli._slash_confirm_state = None
    cli._sync_program_status()
    assert states() == ["state=blocked:kind=permission", "state=working", "state=blocked:kind=auth",
                        "state=blocked:kind=question", "state=working"]

    cli._connection_state, cli._agent_running = None, False
    cli._program_turn_outcome = None  # monitor gave up on a still-running agent
    cli._sync_program_status()
    assert states() == ["state=idle"]


def test_quit_clears_the_reported_record_once_and_late_turn_ends_stay_silent(monkeypatch):
    """idle/done/error outlive the process under the spec, so quitting must clear the record or a pane
    back at its shell keeps listing Hermes as an agent. A turn that settles after quit must not re-report."""
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli = HermesCLI.__new__(HermesCLI)
    cli.program_status_enabled, cli._agent_running = True, False

    cli._clear_program_status()
    assert written == []  # never reported: nothing to clear

    cli = HermesCLI.__new__(HermesCLI)
    cli.program_status_enabled, cli._agent_running = True, False
    cli._sync_program_status()
    written.clear()
    cli._clear_program_status()
    cli._clear_program_status()
    assert written == ["\x1b]7501;state=clear\x1b\\"]

    written.clear()
    cli._program_turn_outcome = ("done", "late reply")  # an interrupted turn settling during shutdown
    cli._sync_program_status()
    assert written == []


def test_one_shot_exit_clears_the_reported_record(monkeypatch):
    """A non-quiet `-q` turn reports done/error; its exit path, not /quit, must clear the record."""
    import cli as cli_module
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    for name in ("_flush_one_shot_session_store", "_notify_single_query_session_finalize",
                 "_shutdown_agent_memory_provider", "_wait_for_oneshot_background_completions", "_run_cleanup"):
        monkeypatch.setattr(cli_module, name, lambda *a, **k: None)
    one_shot = HermesCLI.__new__(HermesCLI)
    one_shot.program_status_enabled, one_shot._agent_running = True, False
    one_shot._release_active_session = lambda: None
    one_shot._program_turn_outcome = ("done", "")
    one_shot._sync_program_status()
    cli_module._finalize_single_query(one_shot)
    assert written[-1] == "\x1b]7501;state=clear\x1b\\"


def test_quit_clear_respects_display_program_status_false(monkeypatch):
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli = HermesCLI.__new__(HermesCLI)
    cli.program_status_enabled, cli._agent_running = False, False
    cli._sync_program_status()
    cli._clear_program_status()
    assert written == []


def test_display_program_status_false_from_config_writes_nothing(monkeypatch):
    from tests.hermes_cli.test_cli_init import _make_cli

    cli = _make_cli(config_overrides={"display": {"compact": False, "program_status": False}})
    assert cli.program_status_enabled is False
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cli._approval_state = {"command": "x"}
    cli._sync_program_status()
    assert written == []


def test_warp_osc777_only_under_supported_warp_build(monkeypatch):
    out = _ring(monkeypatch, flag_on=True, env=_WARP_OK, context="approval", detail="rm -rf build")
    prefix = "\x1b]777;notify;warp://cli-agent;"
    assert out.count(prefix) == 1
    payload = json.loads(out.split(prefix, 1)[1].rstrip("\x07"))
    assert payload["agent"] == "hermes"
    assert payload["event"] == "permission_request"
    assert payload["summary"] == "rm -rf build"
    assert payload["session_id"] == "sess-1"
    assert payload["v"] == 1
    # Broken build (advertises the protocol var but can't render) → OSC 9 only.
    broken = dict(_WARP_OK, WARP_CLIENT_VERSION="v0.2026.03.25.08.24.stable_05")
    assert prefix not in _ring(monkeypatch, flag_on=True, env=broken, context="approval")
    # Not Warp at all → OSC 9 only.
    not_warp = dict(_WARP_OK, TERM_PROGRAM="ghostty")
    assert prefix not in _ring(monkeypatch, flag_on=True, env=not_warp, context="approval")


def test_running_app_gets_bell_and_osc9_on_its_loop_never_a_second_tty_writer(monkeypatch):
    """With the prompt_toolkit app live, the bell + notification must reach the tty through the
    app's output ON THE APP LOOP, never via a second writer from the calling thread."""
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(terminal_notify, "write_tty", lambda seq: pytest.fail(f"stray tty write: {seq!r}"))

    class _Output:
        raw = []

        def write_raw(self, data):
            self.raw.append(data)

        def flush(self):
            self.raw.append("<flush>")

    class _Loop:
        queued = []

        def call_soon_threadsafe(self, fn):
            self.queued.append(fn)

    class _App:
        _is_running = True
        loop = _Loop()
        output = _Output()

    cli = HermesCLI.__new__(HermesCLI)
    cli.bell_on_complete = True
    cli.session_id = "sess-1"
    cli._app = _App()
    cli._ring_bell(context="turn complete")
    # Nothing touched the tty from the calling thread; the write is queued for the loop.
    assert _Output.raw == []
    assert len(_Loop.queued) == 1
    _Loop.queued[0]()
    assert _Output.raw == ["\a\x1b]9;Hermes: turn complete\x07", "<flush>"]


_SPEC_TOKEN = "dummy-token-replace-me"
_SPEC_TIME = 1795543200
_SPEC_PID = 74877


def test_program_resume_spec_vectors():
    """Official OSC 7577 rev 0.5 test vectors match byte-for-byte."""
    assert (
        terminal_notify.program_resume_sequence(
            ["nvim", "-S", "/home/user/.local/state/nvim/sessions/8f2c1d.vim"],
            token=_SPEC_TOKEN, pid=_SPEC_PID, now=_SPEC_TIME,
        )
        == "\x1b]7577;pid=74877:time=1795543200:"
           "argv=bnZpbQ==,LVM=,L2hvbWUvdXNlci8ubG9jYWwvc3RhdGUvbnZpbS9zZXNzaW9ucy84ZjJjMWQudmlt:"
           "mac=9dbb99bfb856749a87563839d7ce9062818d78cf25909db93f309e0cd3b1909b\x1b\\"
    )
    assert (
        terminal_notify.program_resume_clear_sequence(
            token=_SPEC_TOKEN, pid=_SPEC_PID, now=_SPEC_TIME,
        )
        == "\x1b]7577;pid=74877:time=1795543200:clear=1:"
           "mac=c30603369f98fef581a4098be889175b0c74f0641e49f8b25f4326cb38f97c93\x1b\\"
    )
    assert (
        terminal_notify.program_resume_sequence(
            ["lazygit"], cwd="/home/user/src/project",
            token=_SPEC_TOKEN, pid=_SPEC_PID, now=_SPEC_TIME,
        )
        == "\x1b]7577;pid=74877:time=1795543200:argv=bGF6eWdpdA==:"
           "cwd=L2hvbWUvdXNlci9zcmMvcHJvamVjdA==:"
           "mac=35cb909805c380336efd3689a125acc774a9dec8422a1d21492ebec06ca758e1\x1b\\"
    )


def test_program_resume_validation_rejects_malformed_inputs(monkeypatch):
    monkeypatch.delenv("OSC7577_TOKEN", raising=False)
    assert terminal_notify.osc7577_token() is None

    for bad_token in ("", "short", "a" * 65, f"{_SPEC_TOKEN}\n", f" {_SPEC_TOKEN}", "bad$token_123456789012"):
        assert terminal_notify.osc7577_token({"OSC7577_TOKEN": bad_token}) is None
        with pytest.raises(ValueError):
            terminal_notify.program_resume_sequence(["hermes"], token=bad_token, pid=1, now=1)

    for bad_argv in ([], ["./hermes"], ["bin/hermes"], ["-hermes"], [""], ["hermes", "ok\n"], ["hermes", "x\x85"], ["a"] * 65):
        with pytest.raises(ValueError):
            terminal_notify.program_resume_sequence(bad_argv, token=_SPEC_TOKEN, pid=1, now=1)

    with pytest.raises(ValueError):
        terminal_notify.program_resume_sequence(["hermes"], cwd="relative/dir", token=_SPEC_TOKEN, pid=1, now=1)

    with pytest.raises(ValueError):
        terminal_notify.program_resume_sequence(["hermes", "--resume", "s" * 2000], token=_SPEC_TOKEN, pid=1, now=1)


def test_build_hermes_resume_argv_flags(monkeypatch):
    import sys
    from types import SimpleNamespace
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.relaunch as relaunch_mod

    monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/local/bin/hermes")
    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(explicit_cli_profile=lambda: None))
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    assert terminal_notify.build_hermes_resume_argv(surface="cli") == ["/usr/local/bin/hermes", "--cli"]
    assert terminal_notify.build_hermes_resume_argv("sess-1", surface="cli") == [
        "/usr/local/bin/hermes", "--cli", "--resume", "sess-1"
    ]

    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "work")
    assert terminal_notify.build_hermes_resume_argv("sess-2", surface="tui", tui_native=True, tui_dev=True) == [
        "/usr/local/bin/hermes", "-p", "work", "--tui", "--tui-native", "--dev", "--resume", "sess-2"
    ]

    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(explicit_cli_profile=lambda: "default"))
    monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "./relative-hermes")
    assert terminal_notify.build_hermes_resume_argv("sess-3", surface="cli") == [
        sys.executable, "-m", "hermes_cli.main", "-p", "default", "--cli", "--resume", "sess-3"
    ]


def _decode_osc7577_argv(seq: str) -> list[str]:
    import base64

    assert seq.startswith("\x1b]7577;") and seq.endswith("\x1b\\")
    body = seq[len("\x1b]7577;") : -2]
    for part in body.split(":"):
        if part.startswith("argv="):
            return [base64.b64decode(item).decode("utf-8") for item in part[len("argv=") :].split(",")]
    return []


def test_program_resume_cli_lifecycle_and_clear_semantics(monkeypatch):
    import sys
    from types import SimpleNamespace
    import cli as cli_module
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.relaunch as relaunch_mod

    written = []
    monkeypatch.setenv("OSC7577_TOKEN", _SPEC_TOKEN)
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/local/bin/hermes")
    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(explicit_cli_profile=lambda: None))
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    cli = HermesCLI.__new__(HermesCLI)
    cli.program_resume_enabled = True
    cli.session_id = "sess-empty"

    # Non-interactive path stays silent until _tui_process_loop activates.
    cli._sync_program_resume()
    assert written == []

    # Interactive empty session omits --resume.
    cli._program_resume_interactive = True
    cli._sync_program_resume()
    assert len(written) == 1
    assert _decode_osc7577_argv(written[0]) == ["/usr/local/bin/hermes", "--cli"]
    cli._sync_program_resume()
    assert len(written) == 1  # unchanged deduplicated

    # Starting a turn upgrades to --resume <session_id>.
    cli._agent_running = True
    cli._sync_program_resume()
    assert _decode_osc7577_argv(written[-1]) == ["/usr/local/bin/hermes", "--cli", "--resume", "sess-empty"]

    # Bang shell return forces a re-report even when argv is unchanged.
    count_before_bang = len(written)
    cli._sync_program_resume(force=True)
    assert len(written) == count_before_bang + 1

    # /new drops durable state back to bare --cli.
    cli._agent_running = False
    cli.conversation_history = []
    cli._resumed = False
    cli._session_title_persisted = False
    cli.session_id = "sess-next"
    cli._sync_program_resume()
    assert _decode_osc7577_argv(written[-1]) == ["/usr/local/bin/hermes", "--cli"]

    # Oversized/invalid session_id after a reported command clears the stale record.
    cli._resumed = True
    cli.session_id = "bad\nsession"
    cli._sync_program_resume()
    assert ":clear=1:mac=" in written[-1]

    def _prep_for_shutdown(instance):
        instance._viewer_disconnected = False
        instance._terminal_io_broken = False
        instance._program_status_closed = True
        instance._session_db = None
        instance._voice_recorder = None
        instance.agent = None
        instance._pet_stop_anim = lambda: None
        instance._persist_active_session_before_close = lambda: None
        instance._print_exit_summary = lambda: None
        instance._release_active_session = lambda: None

    # Signal exit (SIGHUP/SIGTERM watchdog armed) MUST NOT send clear=1.
    cli._program_resume_closed = False
    cli.session_id = "sess-live"
    cli._sync_program_resume()
    before_sig = len(written)
    monkeypatch.setattr(cli_module, "_signal_watchdog_armed", True)
    monkeypatch.setattr(cli_module, "_invoke_interrupted_session_end", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_run_cleanup", lambda: None)
    _prep_for_shutdown(cli)
    cli._tui_shutdown()
    assert len(written) == before_sig  # no clear=1 on signal shutdown

    # Normal /quit clears once and blocks further writes.
    monkeypatch.setattr(cli_module, "_signal_watchdog_armed", False)
    cli2 = HermesCLI.__new__(HermesCLI)
    cli2.program_resume_enabled = True
    cli2._program_resume_interactive = True
    cli2.session_id = "sess-quit"
    cli2._resumed = True
    _prep_for_shutdown(cli2)
    cli2._sync_program_resume()
    before_quit = len(written)
    cli2._tui_shutdown()
    assert len(written) == before_quit + 1
    assert ":clear=1:mac=" in written[-1]
    cli2._clear_program_resume()
    assert len(written) == before_quit + 1


def test_gateway_chat_view_osc7577_respects_finite_and_quit(monkeypatch):
    import sys
    from types import SimpleNamespace
    from hermes_cli.gateway_chat_view import GatewayChatView
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.relaunch as relaunch_mod

    written = []
    monkeypatch.setenv("OSC7577_TOKEN", _SPEC_TOKEN)
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/local/bin/hermes")
    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(explicit_cli_profile=lambda: None))
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    class _DummyClient:
        pass

    view = GatewayChatView(_DummyClient(), {"stored_session_id": "gw-1", "info": {"lazy": True}})
    view._sync_program_resume()
    assert _decode_osc7577_argv(written[-1]) == ["/usr/local/bin/hermes", "--cli"]

    view._session_durable = True
    view._sync_program_resume()
    assert _decode_osc7577_argv(written[-1]) == ["/usr/local/bin/hermes", "--cli", "--resume", "gw-1"]

    view._clear_program_resume()
    assert ":clear=1:mac=" in written[-1]

    # One-shot / finite view never writes OSC 7577.
    before_finite = len(written)
    finite_view = GatewayChatView(_DummyClient(), {"stored_session_id": "gw-2", "info": {"lazy": False}})
    finite_view.finite = True
    finite_view._sync_program_resume()
    assert len(written) == before_finite
