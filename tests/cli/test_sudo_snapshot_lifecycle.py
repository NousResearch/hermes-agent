"""An interrupted prompt must restore its own draft before a successor captures it."""

import threading
from types import SimpleNamespace

from tests.cli.test_sudo_cancellation import _cli, _wait


def test_interrupt_snapshot_restore_cannot_consume_successor():
    cli = _cli()
    results = []
    restoring, release, old_done = threading.Event(), threading.Event(), threading.Event()
    original_restore = cli._restore_modal_input_snapshot

    def run_old():
        results.append(cli._sudo_password_callback())
        old_done.set()

    old = threading.Thread(target=run_old, daemon=True)
    old.start()
    _wait(lambda: cli._sudo_state is not None)

    def restore():
        if threading.current_thread() is cancel:
            restoring.set()
            assert release.wait(3)
        original_restore()

    cli._restore_modal_input_snapshot = restore
    cancel = threading.Thread(target=cli._clear_active_overlays_for_interrupt, daemon=True)
    fresh = threading.Thread(target=lambda: results.append(cli._sudo_password_callback()), daemon=True)
    cancel.start()
    assert restoring.wait(3)
    try:
        worker_finished = old_done.wait(0.1)
        fresh.start()
        if worker_finished:
            _wait(lambda: cli._sudo_state is not None)  # reproduced old cleanup can reach the new draft
        release.set()
        cancel.join(3)
        _wait(lambda: cli._sudo_state is not None)
        cli._app.current_buffer.text = "secret"
        cli._tui_enter_overlay(SimpleNamespace(app=cli._app))
        old.join(3)
        fresh.join(3)
        assert results == [None, "secret"]
        assert cli._app.current_buffer.text == "draft"
        assert cli._sudo_state is None
    finally:
        release.set()
        if cli._sudo_state:
            cli._sudo_state["response_queue"].put(None)
        for thread in (old, cancel, fresh):
            if thread.ident is not None:
                thread.join(3)
