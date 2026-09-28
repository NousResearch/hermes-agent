"""The startup pending-writes notice: a non-empty staging queue must be visible.

An unattended review fork stages its consolidation proposals in the pending store,
and the only hint about them lands in the fork's tool result — a session nobody
reads. The queue then grows unbounded and invisible. ``HermesCLI._show_pending_writes_notice``
is the one surface that reports the count a user actually sees.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_pwn_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli._app = None
    return cli


def _printed(cli):
    return " ".join(str(c.args[0]) for c in cli.console.print.call_args_list)


def test_startup_notice_lists_each_non_empty_queue(hermes_home):
    from tools import write_approval as wa

    wa.stage_write(
        "memory",
        {"action": "add", "target": "user", "content": "one"},
        summary="one",
        origin="background_review",
    )
    wa.stage_write(
        "skills",
        {"action": "create", "name": "demo"},
        summary="two",
        origin="background_review",
    )

    cli = _cli()
    cli._show_pending_writes_notice()

    out = _printed(cli)
    assert "1 /memory write(s)" in out
    assert "1 /skills write(s)" in out
    assert "/memory pending" in out and "/skills pending" in out


def test_startup_notice_silent_when_queue_empty(hermes_home):
    cli = _cli()
    cli._show_pending_writes_notice()
    cli.console.print.assert_not_called()


def test_startup_notice_never_blocks_startup(hermes_home, monkeypatch):
    """A broken pending store must not kill the session before it starts."""
    from tools import write_approval as wa

    def boom(subsystem):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(wa, "pending_count", boom)
    cli = _cli()
    cli._show_pending_writes_notice()  # no raise
    cli.console.print.assert_not_called()
