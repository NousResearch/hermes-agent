"""The classic CLI's session title follows a compression rotation.

A gateway-backed CLI keeps the conversation's creation id for its whole life, while every
compression moves the live segment, and the title with it, to a new row. Reading the creation
row's title made the status bar badge (and ``/title``) go blank after the first compression.

The CLI's store is its ``GatewaySessionStore`` on a connection to a real session authority in this
process; the rows are seeded and read back through the owner's storage interface handle.
"""
import time
from datetime import datetime

import pytest

from cli import HermesCLI
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.gateway_owner import GatewayOwner
from tests.fixtures.storage_seed import seed_messages


@pytest.fixture
def owner(tmp_path, monkeypatch):
    owner = GatewayOwner(tmp_path, monkeypatch)
    yield owner
    owner.close()


def _rotated(rows, root):
    rows.sessions.set_title(root, "weekly digest")
    seed_messages(rows, root, {"role": "user", "content": "hi"})
    rows.lineage.publish_compression_child(
        root, CompressionChild(session_id="tip", source="cli"),
        [NewMessage.from_live({"role": "user", "content": "hi"})], require_compression_lease=False)
    # The rotation moves the name onto the live segment (the creation row's copy is cleared).
    rows.sessions.set_title("tip", "weekly digest")


def _cli(store):
    """The CLI on its gateway store, holding the session the owner created for it."""
    cli = HermesCLI.__new__(HermesCLI)
    cli._session_db, cli._pending_title = store, None
    cli.model, cli.agent, cli.session_start, cli.conversation_history = "m", None, datetime.now(), []
    cli.session_id = store.open_session(cli)
    return cli


def test_status_bar_and_title_command_read_the_compression_tip(owner, capsys):
    rows = owner.persistence
    cli = _cli(owner.client())
    root = cli.session_id
    _rotated(rows, root)
    assert rows.sessions.get_title(root) is None  # the precondition the CLI used to miss
    # The first frame never waits for the read-only reader to open: the title is blank until the
    # warm-up thread has opened it, then the next repaint reads the compression tip's title.
    assert cli._get_status_bar_session_title() == ""
    deadline = time.monotonic() + 5
    while getattr(cli, "_status_bar_title_warming", False) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cli._get_status_bar_session_title() == "weekly digest"
    cli._cmd_title("/title")
    assert "weekly digest" in capsys.readouterr().out
    cli._cmd_title("/title renamed")
    assert (rows.sessions.get_title("tip"), rows.sessions.get_title(root)) == ("renamed", None)
