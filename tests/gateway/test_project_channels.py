"""Behavior contracts for gateway/project_channels.py (Discord project mirroring).

Real imports against a temp HERMES_HOME; Discord REST is stubbed at the
``_request`` seam (the module's single HTTP chokepoint) so every test exercises
the real provisioning / relay / resolution logic without network.

Contracts, not snapshots: tests assert how pieces of data must relate (a
provisioned channel id round-trips through projects.db; a relayed message is
never relayed twice; a disabled feature performs zero HTTP) rather than
freezing current values.
"""

import json
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def pc(temp_home):
    from gateway import project_channels

    # Module-level state persists across tests within one pytest process:
    # _MIRRORED_SESSIONS is an in-process dedupe set keyed by session id, and
    # test fixtures reuse ids like 's1'. The JSON state files are already
    # isolated per temp HERMES_HOME; this set is the only cross-test leak.
    project_channels._MIRRORED_SESSIONS.clear()
    project_channels._LAST_RETRY_AFTER.clear()
    return project_channels


@pytest.fixture()
def pdb(temp_home):
    from hermes_cli import projects_db

    return projects_db


def _enabled_cfg(**over):
    cfg = {
        "discord": {
            "project_channels": {
                "enabled": True,
                "guild_id": "g1",
                **over,
            }
        }
    }
    return cfg


class _FakeDiscord:
    """Route project_channels._request calls to an in-memory guild."""

    def __init__(self):
        self.channels = []  # [{id,name,type,parent_id}]
        self.messages = {}  # channel_id -> [content]
        self.calls = []  # (method, path)
        self._next_id = 1000
        self.fail_with_429 = 0  # fail the next N POSTs with a rate limit

    def install(self, pc_module, monkeypatch):
        monkeypatch.setattr(pc_module, "_request", self._request)
        monkeypatch.setattr(pc_module, "_bot_token", lambda: "test-token")
        # No real sleeping in tests.
        monkeypatch.setattr(pc_module.time, "sleep", lambda _s: None)

    def _request(self, method, path, token, payload=None):
        self.calls.append((method, path))
        if method == "GET" and path.endswith("/channels"):
            return list(self.channels)
        if method == "POST" and path.endswith("/channels"):
            ch = {
                "id": str(self._next_id),
                "name": payload["name"],
                "type": payload.get("type", 0),
                "parent_id": payload.get("parent_id"),
            }
            self._next_id += 1
            self.channels.append(ch)
            return ch
        if method == "POST" and "/threads" in path:
            ch = {
                "id": str(self._next_id),
                "name": payload["name"],
                "type": 11,
                "parent_id": path.split("/")[2],
            }
            self._next_id += 1
            self.channels.append(ch)
            return ch
        if method == "POST" and path.endswith("/messages"):
            if self.fail_with_429 > 0:
                self.fail_with_429 -= 1
                return None  # _request returns None on 429
            cid = path.split("/")[2]
            self.messages.setdefault(cid, []).append(payload["content"])
            return {"id": str(self._next_id)}
        if method == "PATCH" and path.startswith("/channels/"):
            # Rename: mutate the stored channel so a later GET reflects it,
            # exactly as Discord would. Without this the double cannot catch a
            # reconciler that renames forever because the name never sticks.
            cid = path.split("/")[2]
            for ch in self.channels:
                if str(ch["id"]) == cid:
                    if "name" in (payload or {}):
                        ch["name"] = payload["name"]
                    return dict(ch)
            return {}
        return {}


@pytest.fixture()
def fake_discord(pc, monkeypatch):
    fake = _FakeDiscord()
    fake.install(pc, monkeypatch)
    return fake


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


def test_disabled_by_default_and_performs_no_http(pc, pdb, fake_discord, tmp_path):
    """The feature must be inert without explicit opt-in: no provisioning, no
    HTTP, project creation unaffected."""
    assert pc.is_enabled({}) is False
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(tmp_path)])
    assert pc.provision_project(pid, config={}) is None
    assert fake_discord.calls == []


def test_enabled_requires_guild_id(pc):
    cfg = {"discord": {"project_channels": {"enabled": True}}}
    assert pc.is_enabled(cfg) is False


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def test_provision_creates_channel_and_persists_binding(pc, pdb, fake_discord, tmp_path):
    """Channel id returned by Discord must round-trip through projects.db and
    resolve back via project_for_channel."""
    cfg = _enabled_cfg()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="My Game", folders=[str(tmp_path)])
    cid = pc.provision_project(pid, config=cfg)
    assert cid is not None
    with pdb.connect_closing() as conn:
        proj = pdb.get_project(conn, pid)
        assert proj.discord_channel_id == cid
        assert pdb.project_for_channel(conn, cid).id == pid
    # The channel actually exists guild-side, under the category.
    names = {c["name"] for c in fake_discord.channels}
    assert "proj-my-game" in names
    assert "Projects" in names  # category auto-created


def test_provision_is_idempotent(pc, pdb, fake_discord, tmp_path):
    """Re-provisioning must return the same channel and create nothing new."""
    cfg = _enabled_cfg()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(tmp_path)])
    first = pc.provision_project(pid, config=cfg)
    channel_count = len(fake_discord.channels)
    second = pc.provision_project(pid, config=cfg)
    assert first == second
    assert len(fake_discord.channels) == channel_count


def test_provision_adopts_existing_same_named_channel(pc, pdb, fake_discord, tmp_path):
    """A pre-existing channel with the expected name is adopted, not duplicated."""
    cfg = _enabled_cfg()
    fake_discord.channels.append(
        {"id": "777", "name": "proj-p", "type": 0, "parent_id": None}
    )
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(tmp_path)])
    cid = pc.provision_project(pid, config=cfg)
    assert cid == "777"
    assert sum(1 for c in fake_discord.channels if c["name"] == "proj-p") == 1


# ---------------------------------------------------------------------------
# Channel <-> cwd resolution
# ---------------------------------------------------------------------------


def test_cwd_resolution_is_bidirectional_and_prefix_based(
    pc, pdb, fake_discord, tmp_path
):
    """channel_for_cwd and cwd_for_channel must agree, including for nested
    paths (longest-prefix rule shared with project_for_path)."""
    cfg = _enabled_cfg()
    root = tmp_path / "proj"
    (root / "nested").mkdir(parents=True)
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(root)])
    cid = pc.provision_project(pid, config=cfg)

    assert pc.cwd_for_channel(cid, config=cfg) == str(root)
    # A thread inherits via parent_id.
    assert pc.cwd_for_channel("somethread", parent_id=cid, config=cfg) == str(root)
    # Nested cwd maps back to the channel.
    assert pc.channel_for_cwd(str(root / "nested"), config=cfg) == cid
    # Unrelated path maps to nothing.
    assert pc.channel_for_cwd(str(tmp_path / "elsewhere"), config=cfg) is None


def test_cwd_for_missing_directory_is_refused(pc, pdb, fake_discord, tmp_path):
    """A bound project whose folder vanished must yield no cwd (the agent must
    not chdir into a nonexistent directory)."""
    cfg = _enabled_cfg()
    root = tmp_path / "gone"
    root.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(root)])
    cid = pc.provision_project(pid, config=cfg)
    root.rmdir()
    assert pc.cwd_for_channel(cid, config=cfg) is None


# ---------------------------------------------------------------------------
# Create-from-Discord
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["../evil", "a/b", "a\\b", "C:x", "..", 'x"y', "x|y", ""]
)
def test_create_from_discord_rejects_unsafe_names(pc, fake_discord, tmp_path, bad):
    root = tmp_path / "projects_root"
    root.mkdir()
    cfg = _enabled_cfg(projects_root=str(root))
    res = pc.create_project_from_discord(bad, config=cfg)
    assert res["ok"] is False
    # Nothing may be created under the projects root for a rejected name.
    assert list(root.iterdir()) == []


def test_create_from_discord_makes_folder_row_and_channel(
    pc, pdb, fake_discord, tmp_path
):
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    res = pc.create_project_from_discord("Cool Thing", config=cfg)
    assert res["ok"] is True
    folder = Path(res["folder"])
    assert folder.is_dir() and folder.parent == tmp_path
    with pdb.connect_closing() as conn:
        proj = pdb.project_for_path(conn, str(folder))
        assert proj is not None
        assert proj.discord_channel_id == res["channel_id"]


def test_create_from_discord_is_idempotent_on_existing_project(
    pc, pdb, fake_discord, tmp_path
):
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    first = pc.create_project_from_discord("Twice", config=cfg)
    second = pc.create_project_from_discord("Twice", config=cfg)
    assert second["ok"] is True and second["already_existed"] is True
    assert second["project_id"] == first["project_id"]


# ---------------------------------------------------------------------------
# Adoption: manual Discord channel -> project (reverse of sync_all_projects)
# ---------------------------------------------------------------------------


def _seed_category(fake, name="Projects", cat_id="cat1"):
    fake.channels.append({"id": cat_id, "name": name, "type": 4, "parent_id": None})
    return cat_id


def _seed_channel(fake, name, parent_id, chan_id):
    fake.channels.append(
        {"id": chan_id, "name": name, "type": 0, "parent_id": parent_id}
    )
    return chan_id


def test_adopt_creates_project_folder_and_binding(pc, pdb, fake_discord, tmp_path):
    """A hand-made proj-* channel becomes a real project bound to that channel."""
    cat = _seed_category(fake_discord)
    _seed_channel(fake_discord, "proj-manual-thing", cat, "c99")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    results = pc.adopt_orphan_channels(config=cfg)

    assert [n for n, pid in results if pid] == ["proj-manual-thing"]
    folder = tmp_path / "manual-thing"
    assert folder.is_dir()
    with pdb.connect_closing() as conn:
        proj = pdb.project_for_channel(conn, "c99")
        assert proj is not None
        assert proj.primary_path == str(folder)


def test_adopt_is_idempotent(pc, pdb, fake_discord, tmp_path):
    """Re-running adoption must not fork a second project for the same channel."""
    cat = _seed_category(fake_discord)
    _seed_channel(fake_discord, "proj-once", cat, "c1")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    first = pc.adopt_orphan_channels(config=cfg)
    second = pc.adopt_orphan_channels(config=cfg)

    assert [n for n, pid in first if pid] == ["proj-once"]
    # Second pass sees it bound and skips it entirely.
    assert second == []
    with pdb.connect_closing() as conn:
        assert len(pdb.list_projects(conn)) == 1


def test_adopt_ignores_channels_outside_the_category(pc, pdb, fake_discord, tmp_path):
    """Only channels in the configured category are adoptable.

    A guild's unrelated channels must never become projects — adopting one
    would bind the agent's cwd to a folder the user never asked for.
    """
    cat = _seed_category(fake_discord)
    _seed_channel(fake_discord, "proj-inside", cat, "c1")
    _seed_channel(fake_discord, "proj-outside", "other-cat", "c2")
    _seed_channel(fake_discord, "proj-orphan-no-parent", None, "c3")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    results = pc.adopt_orphan_channels(config=cfg)

    assert [n for n, pid in results if pid] == ["proj-inside"]
    with pdb.connect_closing() as conn:
        assert pdb.project_for_channel(conn, "c2") is None
        assert pdb.project_for_channel(conn, "c3") is None


def test_adopt_ignores_channels_without_the_prefix(pc, pdb, fake_discord, tmp_path):
    cat = _seed_category(fake_discord)
    _seed_channel(fake_discord, "general", cat, "c1")
    _seed_channel(fake_discord, "proj-real", cat, "c2")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    results = pc.adopt_orphan_channels(config=cfg)

    assert [n for n, pid in results if pid] == ["proj-real"]
    with pdb.connect_closing() as conn:
        assert pdb.project_for_channel(conn, "c1") is None


def test_adopt_binds_existing_project_instead_of_duplicating(
    pc, pdb, fake_discord, tmp_path
):
    """Channel deleted then hand-recreated: rebind, don't create a second project.

    The folder already belongs to a project, so adoption must attach the new
    channel id to it rather than creating a duplicate row for the same path.
    """
    folder = tmp_path / "existing"
    folder.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(
            conn, name="existing", folders=[str(folder)], primary_path=str(folder)
        )
    cat = _seed_category(fake_discord)
    _seed_channel(fake_discord, "proj-existing", cat, "c-new")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    results = pc.adopt_orphan_channels(config=cfg)

    assert [n for n, p in results if p] == ["proj-existing"]
    with pdb.connect_closing() as conn:
        assert len(pdb.list_projects(conn)) == 1
        assert pdb.get_project(conn, pid).discord_channel_id == "c-new"


def test_adopt_does_nothing_when_disabled(pc, fake_discord, tmp_path):
    """A disabled feature performs zero HTTP."""
    cfg = {"discord": {"project_channels": {"enabled": False, "guild_id": "g1"}}}
    assert pc.adopt_orphan_channels(config=cfg) == []
    assert fake_discord.calls == []


def test_adopt_without_category_present_is_a_noop(pc, fake_discord, tmp_path):
    """A missing category must not be created as a side effect of a scan."""
    _seed_channel(fake_discord, "proj-lonely", "nope", "c1")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    assert pc.adopt_orphan_channels(config=cfg) == []
    # Read-only: the GET happened, but no channel/category was ever POSTed.
    assert not [c for c in fake_discord.calls if c[0] == "POST"]


# ---------------------------------------------------------------------------
# Rename reconciliation (project <-> channel, session -> thread)
# ---------------------------------------------------------------------------


def _bound_project(pdb, fake, tmp_path, name="Alpha", chan="proj-alpha", cid="c1"):
    """Create a project bound to a seeded channel, and agree their names."""
    folder = tmp_path / name.lower()
    folder.mkdir(exist_ok=True)
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(
            conn, name=name, folders=[str(folder)], primary_path=str(folder)
        )
        pdb.update_project(conn, pid, discord_channel_id=cid)
    _seed_category(fake)
    _seed_channel(fake, chan, "cat1", cid)
    return pid


def test_reconcile_first_pass_only_records_never_renames(
    pc, pdb, fake_discord, tmp_path
):
    """Enabling reconciliation must not mass-rename existing channels."""
    _bound_project(pdb, fake_discord, tmp_path)
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    assert pc.reconcile_project_names(config=cfg) == []
    assert not [c for c in fake_discord.calls if c[0] == "PATCH"]
    # But it did record the agreed pair, so a later change is detectable.
    assert pc.name_state_path().exists()


def test_discord_rename_propagates_to_hermes(pc, pdb, fake_discord, tmp_path):
    pid = _bound_project(pdb, fake_discord, tmp_path)
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    pc.reconcile_project_names(config=cfg)  # agree

    # User renames the channel in the Discord client.
    fake_discord.channels[-1]["name"] = "proj-renamed-in-discord"
    actions = pc.reconcile_project_names(config=cfg)

    assert len(actions) == 1
    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, pid).name == "renamed-in-discord"


def test_hermes_rename_propagates_to_discord(pc, pdb, fake_discord, tmp_path):
    pid = _bound_project(pdb, fake_discord, tmp_path)
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    pc.reconcile_project_names(config=cfg)  # agree

    with pdb.connect_closing() as conn:
        pdb.update_project(conn, pid, name="Beta Project")
    actions = pc.reconcile_project_names(config=cfg)

    assert len(actions) == 1
    assert fake_discord.channels[-1]["name"] == "proj-beta-project"


def test_reconcile_is_stable_after_syncing(pc, pdb, fake_discord, tmp_path):
    """A reconciled pair must not keep renaming on every later pass."""
    pid = _bound_project(pdb, fake_discord, tmp_path)
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    pc.reconcile_project_names(config=cfg)
    with pdb.connect_closing() as conn:
        pdb.update_project(conn, pid, name="Gamma")
    pc.reconcile_project_names(config=cfg)

    before = len(fake_discord.calls)
    assert pc.reconcile_project_names(config=cfg) == []
    # Only the channel LIST call, no further PATCH.
    assert not [
        c for c in fake_discord.calls[before:] if c[0] == "PATCH"
    ]


def test_both_sides_renamed_hermes_wins(pc, pdb, fake_discord, tmp_path):
    """A simultaneous rename resolves to Hermes, without flip-flopping.

    projects.db is the system of record (it owns the folder binding), so the
    Discord edit is discarded and the channel is renamed back to match.
    """
    pid = _bound_project(pdb, fake_discord, tmp_path)
    cfg = _enabled_cfg(projects_root=str(tmp_path))
    pc.reconcile_project_names(config=cfg)

    fake_discord.channels[-1]["name"] = "proj-from-discord"
    with pdb.connect_closing() as conn:
        pdb.update_project(conn, pid, name="From Hermes")

    pc.reconcile_project_names(config=cfg)
    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, pid).name == "From Hermes"
    assert fake_discord.channels[-1]["name"] == "proj-from-hermes"

    # And it settles: a further pass changes nothing.
    before = len(fake_discord.calls)
    assert pc.reconcile_project_names(config=cfg) == []
    assert not [c for c in fake_discord.calls[before:] if c[0] == "PATCH"]


def test_reconcile_ignores_unbound_projects(pc, pdb, fake_discord, tmp_path):
    """A project with no channel (or a deleted one) is skipped, not crashed on."""
    folder = tmp_path / "lonely"
    folder.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(
            conn, name="Lonely", folders=[str(folder)], primary_path=str(folder)
        )
        pdb.update_project(conn, pid, discord_channel_id="gone-999")
    cfg = _enabled_cfg(projects_root=str(tmp_path))

    assert pc.reconcile_project_names(config=cfg) == []


# ---------------------------------------------------------------------------
# Session -> thread mirroring + relay
# ---------------------------------------------------------------------------


class _FakeSessionDB:
    def __init__(self, sessions, messages):
        self._sessions = sessions
        self._messages = messages

    def list_sessions_rich(self, **_kw):
        return list(self._sessions)

    def get_messages(self, session_id):
        return list(self._messages.get(session_id, []))


def _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg):
    root = tmp_path / "proj"
    root.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="P", folders=[str(root)])
    cid = pc.provision_project(pid, config=cfg)
    return root, cid


def test_mirror_skips_discord_native_sessions(pc, pdb, fake_discord, tmp_path):
    """A Discord-born session already has a thread; mirroring it would
    duplicate. Contract: source='discord' rows never appear in the pending set,
    even with a project cwd."""
    cfg = _enabled_cfg()
    root, _cid = _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg)
    db = _FakeSessionDB(
        [
            {"id": "s_disc", "source": "discord", "cwd": str(root), "title": "D"},
            {"id": "s_desk", "source": "desktop", "cwd": str(root), "title": "K"},
        ],
        {},
    )
    pending = pc.sessions_needing_threads(db)
    ids = {p["session_id"] for p in pending}
    assert ids == {"s_desk"}


def test_mirror_creates_thread_once_and_binds_reverse_lookup(
    pc, pdb, fake_discord, tmp_path
):
    cfg = _enabled_cfg()
    root, cid = _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg)
    db = _FakeSessionDB(
        [{"id": "s1", "source": "tui", "cwd": str(root), "title": "T"}],
        {"s1": [{"id": 1, "role": "user", "content": "hello"}]},
    )
    results = pc.mirror_sessions_to_threads(db, config=cfg)
    assert len(results) == 1 and results[0][1] is not None
    tid = results[0][1]
    # Reverse lookup binds the thread to its session.
    assert pc.mirrored_session_for_thread(tid) == "s1"
    # Second pass creates nothing.
    assert pc.mirror_sessions_to_threads(db, config=cfg) == []


def test_relay_backfills_then_never_duplicates(pc, pdb, fake_discord, tmp_path):
    """First sight posts the tail of the transcript (bounded); subsequent
    passes relay only rows above the high-water mark; a message is copied at
    most once, ever."""
    cfg = _enabled_cfg(backfill_limit=2)
    root, cid = _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg)
    msgs = [
        {"id": 1, "role": "user", "content": "one"},
        {"id": 2, "role": "assistant", "content": "two"},
        {"id": 3, "role": "user", "content": "three"},
    ]
    db = _FakeSessionDB(
        [{"id": "s1", "source": "tui", "cwd": str(root), "title": "T"}],
        {"s1": msgs},
    )
    pc.mirror_sessions_to_threads(db, config=cfg)
    n = pc.relay_new_messages(db, config=cfg)
    # backfill_limit=2 → the last two turns, plus the truncation notice.
    assert n == 2
    # Nothing new → nothing relayed.
    assert pc.relay_new_messages(db, config=cfg) == 0
    # A new message relays exactly once.
    msgs.append({"id": 4, "role": "assistant", "content": "four"})
    assert pc.relay_new_messages(db, config=cfg) == 1
    assert pc.relay_new_messages(db, config=cfg) == 0


def test_relay_never_recopies_marked_text(pc, pdb, fake_discord, tmp_path):
    """Echo-loop guard: content carrying RELAY_MARKER is a copy we made and
    must never be relayed again."""
    cfg = _enabled_cfg(backfill_limit=0)
    root, cid = _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg)
    msgs = [{"id": 1, "role": "user", "content": "seed"}]
    db = _FakeSessionDB(
        [{"id": "s1", "source": "tui", "cwd": str(root), "title": "T"}],
        {"s1": msgs},
    )
    pc.mirror_sessions_to_threads(db, config=cfg)
    pc.relay_new_messages(db, config=cfg)  # seeds high-water mark
    msgs.append(
        {"id": 2, "role": "assistant", "content": f"{pc.RELAY_MARKER} copied text"}
    )
    assert pc.relay_new_messages(db, config=cfg) == 0


def test_rate_limited_backfill_does_not_lose_messages(
    pc, pdb, fake_discord, tmp_path
):
    """A 429 on the first attempt must not advance past an unposted message:
    post_message retries once after the cooldown. Contract: with one transient
    429 per message, every backfilled message still lands."""
    cfg = _enabled_cfg(backfill_limit=3)
    root, cid = _mk_session_env(pc, pdb, fake_discord, tmp_path, cfg)
    msgs = [
        {"id": i, "role": "user", "content": f"m{i}"} for i in range(1, 4)
    ]
    db = _FakeSessionDB(
        [{"id": "s1", "source": "tui", "cwd": str(root), "title": "T"}],
        {"s1": msgs},
    )
    results = pc.mirror_sessions_to_threads(db, config=cfg)
    tid = results[0][1]
    fake_discord.fail_with_429 = 1  # first POST fails, retry must recover it
    n = pc.relay_new_messages(db, config=cfg)
    assert n == 3
    bodies = "\n".join(fake_discord.messages.get(tid, []))
    for i in range(1, 4):
        assert f"m{i}" in bodies


# ---------------------------------------------------------------------------
# projects_db column contract
# ---------------------------------------------------------------------------


def test_discord_channel_id_upgrades_legacy_db_in_place(pdb, tmp_path):
    """Opening a DB created without the column must add it (additive
    migration) and existing rows must read back with None."""
    import sqlite3

    db_path = tmp_path / "projects.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE projects (
            id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
            description TEXT, icon TEXT, color TEXT, board_slug TEXT,
            primary_path TEXT, created_at INTEGER NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE project_folders (
            project_id TEXT NOT NULL, path TEXT NOT NULL, label TEXT,
            is_primary INTEGER NOT NULL DEFAULT 0, added_at INTEGER NOT NULL,
            PRIMARY KEY (project_id, path)
        );
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO projects (id, slug, name, created_at)
            VALUES ('p1', 'legacy', 'Legacy', 0);
        """
    )
    conn.commit()
    conn.close()

    c = pdb.connect(db_path=db_path)
    try:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(projects)")}
        assert "discord_channel_id" in cols
        assert pdb.get_project(c, "p1").discord_channel_id is None
    finally:
        c.close()
