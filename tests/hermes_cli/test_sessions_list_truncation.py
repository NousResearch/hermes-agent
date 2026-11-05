"""``hermes sessions list`` tells the user when ``--limit`` cut the listing (#111989).

The cap is applied inside the SQL query, so the lister probes one row past it; the footer
must appear only when that probe row exists.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def db(tmp_path):
    """The seeded store's file."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for i in range(6):
            seed_session(store, f"sess_{i}", "cli")
            seed_messages(store, f"sess_{i}", {"role": "user", "content": f"hello {i}"})
        yield tmp_path / "state.db"


def _list(db, limit, capsys):
    from persistence_assembly import open_persistence_reader
    with open_persistence_reader(db) as store:
        sessions_cmd._cmd_list(Namespace(limit=limit, source=None, workspace=None), persistence=store)
    return capsys.readouterr().out


def test_footer_when_more_sessions_than_limit(db, capsys):
    out = _list(db, 4, capsys)
    ids = [line.split()[-1] for line in out.splitlines() if line.startswith("hello")]
    assert len(ids) == 4  # the probe row is never rendered
    assert "--limit 8" in out


def test_no_footer_when_listing_fits(db, capsys):
    assert "more" not in _list(db, 6, capsys)  # exactly the limit
    assert "more" not in _list(db, 20, capsys)  # fewer than the limit
