"""``kanban.max_in_progress_per_endpoint``: one concurrency lane per provider route.

A local model server (both GPUs behind ``localhost:11434``) or one cloud account
(``openrouter``) is a single quota lane: N workers pointed at it saturate the
same quota, whatever profiles they belong to. The dispatcher resolves each
candidate's endpoint, counts the running workers per lane from the DB every
tick, and defers a candidate whose lane is full.

Regression guard for the first test: with the key unset the dispatcher must be
byte-identical to before — no new skips and no new event payload fields.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Spelled by parts so no address-rewriting layer in the toolchain can mangle it:
# this is the IPv4 loopback the alias-collapse tests must share with ``localhost``.
LOOPBACK_V4 = ".".join(("127", "0", "0", "1"))


@pytest.fixture()
def endpoint_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a kanban DB and room for profile dirs."""
    home = tmp_path / "endpoint-home"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # A module purge keeps this file robust when run beside the per-profile cap
    # tests (they replace hermes_cli modules mid-session).
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    # The host's live memory pressure must not cap this tick (the module purge
    # above discards the conftest's neutralizer).
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    kb.init_db()
    return home


def _profile(home: Path, name: str, *, provider: str | None = None, base_url: str | None = None) -> str:
    """Create a live profile (identity marker = config.yaml) with a model route."""
    directory = home / "profiles" / name
    directory.mkdir(parents=True, exist_ok=True)
    if provider is None and base_url is None:
        (directory / "config.yaml").write_text("{}\n", encoding="utf-8")
        return name
    lines = ["model:"]
    if provider:
        lines.append(f"  provider: {provider}")
    if base_url:
        lines.append(f'  base_url: "{base_url}"')
    (directory / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def _fake_spawn(*_args, **_kwargs):
    return 4242


def _spawned_payload(conn, task_id: str) -> dict:
    from hermes_cli import kanban_db as kb

    events = [e for e in kb.list_events(conn, task_id) if e.kind == "spawned"]
    assert events, f"no spawned event for {task_id}"
    return events[-1].payload or {}


# ---------------------------------------------------------------------------
# 1. Key unset: no new skips, no new event fields
# ---------------------------------------------------------------------------


def test_unset_key_leaves_dispatch_and_events_unchanged(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", provider="openrouter")
    beta = _profile(endpoint_home, "beta", provider="openrouter")

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title="a", assignee=alpha),
               kb.create_task(conn, title="b", assignee=beta)]

    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

    assert sorted(tid for tid, *_ in res.spawned) == sorted(ids)
    assert res.skipped_endpoint_capped == []
    assert res.endpoint_by_task == {}
    with kbc.connect() as conn:
        for tid in ids:
            assert "endpoint" not in _spawned_payload(conn, tid)


# ---------------------------------------------------------------------------
# 2. Two cloud cards on one account: the second is deferred
# ---------------------------------------------------------------------------


def test_second_card_on_one_cloud_account_is_endpoint_capped(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", provider="openrouter")
    beta = _profile(endpoint_home, "beta", provider="openrouter")

    with kbc.connect() as conn:
        first = kb.create_task(conn, title="first", assignee=alpha)
        second = kb.create_task(conn, title="second", assignee=beta)

    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                max_in_progress_per_endpoint={"openrouter": 1})

    assert [tid for tid, *_ in res.spawned] == [first]
    assert res.skipped_endpoint_capped == [(second, "openrouter", 1)]
    assert res.endpoint_by_task == {first: "openrouter", second: "openrouter"}
    with kbc.connect() as conn:
        assert _spawned_payload(conn, first)["endpoint"] == "openrouter"


# ---------------------------------------------------------------------------
# 3. Loopback aliases share one counter
# ---------------------------------------------------------------------------


def test_loopback_aliases_share_one_lane(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", base_url="http://localhost:11434/v1")
    beta = _profile(endpoint_home, "beta", base_url=f"http://{LOOPBACK_V4}:11434/v1")
    gamma = _profile(endpoint_home, "gamma", base_url="http://[::1]:11434/v1")

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title=name, assignee=name)
               for name in (alpha, beta, gamma)]

    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                max_in_progress_per_endpoint={"localhost:11434": 1})

    assert [tid for tid, *_ in res.spawned] == [ids[0]]
    assert res.skipped_endpoint_capped == [(ids[1], "localhost:11434", 1),
                                           (ids[2], "localhost:11434", 1)]
    assert set(res.endpoint_by_task.values()) == {"localhost:11434"}


def test_normalise_endpoint_key_collapses_spellings():
    from hermes_cli.kanban_endpoint_caps import normalise_endpoint_key as key

    spellings = [
        "http://localhost:11434/v1",
        f"http://{LOOPBACK_V4}:11434/v1",
        "http://[::1]:11434/v1",
        "localhost:11434",
        "[::1]:11434",
        "http://LOCALHOST:11434",
    ]
    assert {key(s) for s in spellings} == {"localhost:11434"}
    assert key("https://api.openrouter.ai/v1") == "api.openrouter.ai"
    assert key("openrouter") == "openrouter"
    assert key("") is None
    assert key(None) is None


# ---------------------------------------------------------------------------
# 4. The two caps bite independently
# ---------------------------------------------------------------------------


def test_profile_cap_still_bites_when_endpoint_has_room(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", provider="openrouter")
    beta = _profile(endpoint_home, "beta", provider="openrouter")

    with kbc.connect() as conn:
        alpha_one = kb.create_task(conn, title="a1", assignee=alpha)
        alpha_two = kb.create_task(conn, title="a2", assignee=alpha)
        beta_one = kb.create_task(conn, title="b1", assignee=beta)

    with kbc.connect() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn,
            max_in_progress_per_profile=1,
            max_in_progress_per_endpoint={"openrouter": 5},
        )

    assert [tid for tid, *_ in res.spawned] == [alpha_one, beta_one]
    assert res.skipped_per_profile_capped == [(alpha_two, alpha, 1)]
    assert res.skipped_endpoint_capped == []


def test_endpoint_cap_still_bites_when_profile_has_room(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", provider="openrouter")

    with kbc.connect() as conn:
        first = kb.create_task(conn, title="a1", assignee=alpha)
        second = kb.create_task(conn, title="a2", assignee=alpha)

    with kbc.connect() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn,
            max_in_progress_per_profile=5,
            max_in_progress_per_endpoint={"openrouter": 1},
        )

    assert [tid for tid, *_ in res.spawned] == [first]
    assert res.skipped_per_profile_capped == []
    assert res.skipped_endpoint_capped == [(second, "openrouter", 1)]


# ---------------------------------------------------------------------------
# 5. Unresolvable route falls to "*" and records that key
# ---------------------------------------------------------------------------


def test_unresolvable_endpoint_uses_the_wildcard_lane(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    gamma = _profile(endpoint_home, "gamma")
    delta = _profile(endpoint_home, "delta")

    with kbc.connect() as conn:
        first = kb.create_task(conn, title="first", assignee=gamma)
        second = kb.create_task(conn, title="second", assignee=delta)

    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                max_in_progress_per_endpoint={"*": 1})

    assert [tid for tid, *_ in res.spawned] == [first]
    assert res.skipped_endpoint_capped == [(second, "*", 1)]
    assert res.endpoint_by_task == {first: "*", second: "*"}


def test_unlisted_endpoint_falls_to_the_wildcard_cap():
    from hermes_cli.kanban_endpoint_caps import cap_for

    caps = {"*": 4}
    assert cap_for(caps, "somewhere:8080") == 4
    assert cap_for({"somewhere:8080": 2, "*": 4}, "somewhere:8080") == 2
    assert cap_for({"openrouter": 8}, "somewhere:8080") is None


# ---------------------------------------------------------------------------
# 6. A card-level override pins its own lane
# ---------------------------------------------------------------------------


def test_provider_override_keys_on_its_own_provider(endpoint_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", base_url="http://localhost:11434/v1")

    with kbc.connect() as conn:
        overridden = kb.create_task(
            conn, title="cloud", assignee=alpha,
            model_override="some-cloud-model", provider_override="openrouter",
        )
        local = kb.create_task(conn, title="local", assignee=alpha)

    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                max_in_progress_per_endpoint={"openrouter": 1, "*": 1})

    assert res.endpoint_by_task[overridden] == "openrouter"
    assert res.endpoint_by_task[local] == "localhost:11434"
    assert len(res.spawned) == 2


# ---------------------------------------------------------------------------
# 7. Config parsing: normalise keys, drop junk, unset means no cap
# ---------------------------------------------------------------------------


def test_parse_caps_unset_is_no_cap():
    from hermes_cli.kanban_endpoint_caps import parse_caps

    assert parse_caps(None) is None
    assert parse_caps({}) is None


def test_parse_caps_normalises_keys_and_drops_junk():
    from hermes_cli.kanban_endpoint_caps import parse_caps

    caps = parse_caps({
        f"{LOOPBACK_V4}:11434": 2,
        "OPENROUTER": "3",
        "*": 1,
        "nonsense": 0,
        "bad key": "not-a-number",
    })
    assert caps == {"localhost:11434": 2, "openrouter": 3, "*": 1}


def test_counts_are_recomputed_from_the_db_each_tick(endpoint_home):
    """A finished worker frees its lane on the next tick, with no cached state."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    alpha = _profile(endpoint_home, "alpha", provider="openrouter")

    with kbc.connect() as conn:
        first = kb.create_task(conn, title="a1", assignee=alpha)
        second = kb.create_task(conn, title="a2", assignee=alpha)

    with kbc.connect() as conn:
        res1 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                 max_in_progress_per_endpoint={"openrouter": 1})
    assert [tid for tid, *_ in res1.spawned] == [first]
    assert [tid for tid, *_ in res1.skipped_endpoint_capped] == [second]

    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (first,))

    with kbc.connect() as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn,
                                 max_in_progress_per_endpoint={"openrouter": 1})
    assert [tid for tid, *_ in res2.spawned] == [second]


# ---------------------------------------------------------------------------
# 8. The loader surface: config.yaml -> parsed caps
# ---------------------------------------------------------------------------


def test_configured_caps_are_read_through_the_config_loader(endpoint_home):
    """The daemon/CLI seam reads and normalises the key from ``config.yaml``."""
    from hermes_cli import kanban_db_dispatch as kbd

    (endpoint_home / "config.yaml").write_text(
        "kanban:\n"
        "  max_in_progress_per_endpoint:\n"
        f'    "{LOOPBACK_V4}:11434": 2\n'
        '    "openrouter": 8\n'
        '    "*": 4\n',
        encoding="utf-8",
    )
    assert kbd.configured_max_in_progress_per_endpoint() == {
        "localhost:11434": 2, "openrouter": 8, "*": 4,
    }

    (endpoint_home / "config.yaml").write_text("kanban:\n  review_dispatch: false\n",
                                               encoding="utf-8")
    assert kbd.configured_max_in_progress_per_endpoint() is None
