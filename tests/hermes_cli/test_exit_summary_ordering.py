"""Call-order contract for the interactive exit summary: summary BEFORE cleanup.

``_run_cleanup()`` arms the exit watchdog, which force-exits the process with
``os._exit(0)`` when a cleanup step (memory-provider ``on_session_end``, MCP
server shutdown) wedges past its leash. Any code that runs *after*
``_run_cleanup()`` is therefore silently discarded -- including the cost report
and ``--resume`` hint that ``_print_exit_summary()`` emits.

These tests drive the two interactive exit sites -- ``HermesCLI.run()``'s
stdin-unavailable bail and the main TUI exit ``_tui_shutdown`` -- with
``_print_exit_summary`` / ``_run_cleanup`` / ``_release_active_session`` spied,
and assert the summary is emitted BEFORE cleanup at each.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def _reset_cleanup_flag(monkeypatch):
    monkeypatch.setattr(cli, "_cleanup_done", False)


def _install_spies(monkeypatch, calls):
    """Spy on the exit-path collaborators, leaving the ordering logic itself real."""
    monkeypatch.setattr(cli, "_run_cleanup", lambda *a, **k: calls.append("cleanup"))
    monkeypatch.setattr(
        cli.HermesCLI,
        "_print_exit_summary",
        lambda self, *a, **k: calls.append("summary"),
    )
    monkeypatch.setattr(
        cli.HermesCLI,
        "_release_active_session",
        lambda self, *a, **k: calls.append("release"),
    )


class _AfterRender:
    """Stands in for prompt_toolkit's ``Application.after_render``; swallows ``+=``."""

    def __iadd__(self, _other):
        return self

    def __add__(self, _other):
        return self


def _lighten_run(monkeypatch):
    """Neutralize the heavy/global side effects ``run()`` performs before the
    stdin probe (prompt_toolkit monkeypatching, atexit registration, metrics)."""
    monkeypatch.setattr(cli, "_disable_prompt_toolkit_cpr_warning", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_apply_bracketed_paste_timeout_patch", lambda *a, **k: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_startup.cli_prompt_ready_handler",
        lambda *a, **k: (lambda: None),
    )
    import prompt_toolkit.renderer as _ptr

    monkeypatch.setattr(_ptr, "_hermes_osd_patched", True, raising=False)


def _bare_cli():
    """A real ``HermesCLI`` with ``__init__`` skipped, so class-level spies resolve.

    ``object.__new__`` avoids the heavy constructor (session DB, config) while
    keeping attribute lookup going through the real class -- required for
    ``self._finish_interactive_exit`` / ``self._print_exit_summary`` to be found.
    """
    return object.__new__(cli.HermesCLI)


def test_run_stdin_unavailable_prints_summary_before_cleanup(monkeypatch):
    """``run()``'s stdin-unavailable bail must print the summary before cleanup."""
    calls = []
    _install_spies(monkeypatch, calls)
    _lighten_run(monkeypatch)

    noop = lambda *a, **k: None
    fake = _bare_cli()
    fake._claim_active_session = lambda *a, **k: True
    fake._tui_print_startup = noop
    fake._tui_init_run_state = noop
    fake._tui_build_key_bindings = lambda: None
    fake._tui_build_layout = lambda kb: (None, None)
    fake._tui_build_application = lambda *a, **k: SimpleNamespace(after_render=_AfterRender())
    fake._pet_flush_kitty_frame = noop
    fake._install_resize_recovery = noop
    fake._tui_spinner_loop = noop
    fake._tui_process_loop = noop
    fake._tui_wake_startup = noop
    fake._tui_install_signal_handlers = noop
    fake._tui_stdin_usable = lambda: False
    fake._tui_shutdown = noop

    cli.HermesCLI.run(fake)

    assert calls == ["summary", "cleanup"], (
        "the exit summary must print before cleanup; got order: %r" % (calls,)
    )


def test_tui_shutdown_prints_summary_before_cleanup(monkeypatch):
    """The main interactive TUI exit (``_tui_shutdown``) must not clean up first."""
    calls = []
    _install_spies(monkeypatch, calls)

    fake = _bare_cli()
    fake._should_exit = False
    fake._pet_stop_anim = lambda: None
    fake.agent = None
    fake._agent_running = False
    fake._voice_recorder = None
    fake._persist_active_session_before_close = lambda: None
    fake._session_db = None
    fake._delete_session_on_exit = False

    cli.HermesCLI._tui_shutdown(fake)

    assert calls == ["summary", "cleanup", "release"], (
        "summary must print first, then cleanup, then the lease release; got: %r"
        % (calls,)
    )


def test_finish_interactive_exit_helper_prints_summary_first(monkeypatch):
    """The shared helper both exit sites call orders summary -> cleanup -> release."""
    calls = []
    _install_spies(monkeypatch, calls)

    fake = _bare_cli()
    cli.HermesCLI._finish_interactive_exit(fake, release_session=True)
    assert calls == ["summary", "cleanup", "release"]

    # Without a release request, the helper must not touch the lease.
    calls.clear()
    cli.HermesCLI._finish_interactive_exit(fake)
    assert calls == ["summary", "cleanup"]
