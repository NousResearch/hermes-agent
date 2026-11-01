"""``/queue move`` on the owner's FIFO: displayed indices map to the owner's reorderable places."""
import pytest

from hermes_cli import gateway_queue
from hermes_cli.cli_loops_mixin import CLILoopsMixin


class _Cli(CLILoopsMixin):
    session_id = 's'

    def __init__(self, items):
        self.items = items

    def _queue_items(self):
        return self.items

    def _gateway_queue_store(self):
        return object()


@pytest.fixture
def owner(monkeypatch):
    import cli
    moves, printed = [], []
    monkeypatch.setattr(cli, '_cprint', printed.append)
    monkeypatch.setattr(gateway_queue, 'reorderable', lambda c, store: ['a', 'b', 'c'], raising=False)
    monkeypatch.setattr(gateway_queue, 'move', lambda c, store, admission, position: moves.append(
        (admission, position)) or [admission])
    # A notification row the owner never lets move, listed first, then three movable prompts.
    return _Cli([('note', 'notification'), ('a', 'A'), ('b', 'B'), ('c', 'C')]), moves, printed


def test_move_targets_the_destination_rows_reorderable_place(owner):
    cli, moves, _ = owner
    cli._queue_move('3 2')  # B to where A is listed: A's place among the movable rows is 0
    assert moves == [('b', 0)]


def test_move_never_reads_index_zero_as_the_last_row(owner):
    cli, moves, printed = owner
    cli._queue_move('0 1')
    cli._queue_move('2 1')  # onto the notification row: it never moves, so nothing can take its place
    assert moves == [], moves
    assert len(printed) == 2
