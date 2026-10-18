"""A long-lived pool must not revert the order an operator set with ``hermes auth priority``.

Every pool flush rewrites each row it holds, ``priority`` included. A process that loaded the
pool before the operator reordered it would otherwise put its stale order back on its next
cooldown write, and the operator's order would silently vanish.
"""

import json
import time

import pytest

IDS = ("alpha", "beta", "gamma")
OPERATOR_ORDER = [("gamma", 0), ("alpha", 1), ("beta", 2)]


def _row(cid: str, priority: int, **fields) -> dict:
    return {"id": cid, "label": cid, "priority": priority, **fields}


def _seed(home, rows: list[dict]) -> None:
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": rows}}))


def _rows(home) -> list[dict]:
    return json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"]


def _disk_order(home) -> list[tuple[str, int]]:
    return sorted(((row["id"], row["priority"]) for row in _rows(home)), key=lambda pair: pair[1])


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A root home holding three Anthropic API keys, and a named profile that borrows them."""
    monkeypatch.setenv("HOME", str(tmp_path))  # keep host ~/.claude credentials out of the pool
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    root = tmp_path / "hermes"
    (root / "profiles" / "forge").mkdir(parents=True)
    for home in (root, root / "profiles" / "forge"):
        (home / "config.yaml").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(root))
    _seed(root, [_row(cid, index, auth_type="api_key", source="manual", access_token=f"key-{cid}")
                 for index, cid in enumerate(IDS)])
    return root


@pytest.mark.parametrize("stale_home", ["", "profiles/forge"], ids=["owned", "borrowed-root"])
def test_stale_writer_keeps_the_operators_new_order(root, monkeypatch, stale_home):
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HERMES_HOME", str(root / stale_home))
    stale = load_pool("anthropic")  # a long-running process, loaded before the reorder
    monkeypatch.setenv("HERMES_HOME", str(root))
    load_pool("anthropic").move_entry("gamma", 0)  # `hermes auth priority anthropic gamma 0`
    monkeypatch.setenv("HERMES_HOME", str(root / stale_home))

    next_entry = stale.mark_exhausted_and_rotate(status_code=429, credential_id="alpha")

    assert _disk_order(root) == OPERATOR_ORDER
    assert next(row for row in _rows(root) if row["id"] == "alpha")["last_status"] == "exhausted"
    # The stale process rotates in the operator's order from this write on, not in its own.
    assert next_entry.id == "gamma"


def test_a_writer_that_reorders_the_pool_itself_writes_its_whole_order(root):
    from agent.credential_pool import load_pool

    first = load_pool("anthropic")
    load_pool("anthropic").move_entry("gamma", 0)

    first.move_entry("beta", 0)  # the later deliberate reorder wins, as one consistent order

    assert _disk_order(root) == [("beta", 0), ("alpha", 1), ("gamma", 2)]


def test_adopting_a_peers_token_rotation_is_not_a_reorder(root):
    """The refresh path adopts a peer's whole row; the row's newer priority is not this writer's change."""
    from agent.credential_pool import load_pool

    hour_ms = 3_600_000
    now_ms = int(time.time() * 1000)
    _seed(root, [
        _row(cid, index, auth_type="oauth", source="manual:hermes_pkce",
             access_token=f"at-{cid}", refresh_token=f"rt-{cid}",
             expires_at_ms=now_ms - hour_ms if cid == "alpha" else now_ms + hour_ms)
        for index, cid in enumerate(IDS)
    ])
    stale = load_pool("anthropic")
    load_pool("anthropic").move_entry("gamma", 0)
    rows = _rows(root)  # a peer process then refreshes alpha
    next(row for row in rows if row["id"] == "alpha").update(
        access_token="at-alpha-2", refresh_token="rt-alpha-2", expires_at_ms=now_ms + hour_ms)
    _seed(root, rows)

    assert stale.select().access_token == "at-alpha-2"  # stale's expired alpha adopts the peer's row
    stale.mark_exhausted_and_rotate(status_code=429, credential_id="beta")

    assert _disk_order(root) == OPERATOR_ORDER
