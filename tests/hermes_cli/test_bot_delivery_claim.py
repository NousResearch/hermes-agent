"""Bot-authored quiet one-shots identify delivery ownership before consuming the author."""

from types import SimpleNamespace

import pytest

import cli
from agent.turn_author import turn_author_env


@pytest.mark.parametrize(("quiet", "author", "expected"), [
    (True, {"id": "bot:peer", "name": "Peer", "is_bot": True}, "bot_delivery"),
    (False, {"id": "bot:peer", "name": "Peer", "is_bot": True}, "cli"),
    (True, {"id": "person", "name": "Person", "is_bot": False}, "cli"),
    (True, None, "cli"),
])
def test_single_query_claim_surface(monkeypatch, quiet, author, expected):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_TURN_AUTHOR", raising=False)
    if author is not None:
        for key, value in turn_author_env(author).items():
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **kw: False)
    seen = []
    class Claimed(Exception):
        pass
    def claim(surface, **kwargs):
        seen.append(surface)
        raise Claimed
    stub = SimpleNamespace(_claim_active_session=claim)
    with pytest.raises(Claimed):
        cli._run_single_query_mode(stub, "task", None, quiet, True)
    assert seen == [expected]
