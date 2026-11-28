"""A routing row whose platform's plugin is not registered yet (Yuanbao here: a Labs plugin platform installed
only in a secondary profile or after boot) survives a full routing save instead of being deleted, and joins the
live index once the plugin registers."""

import json

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_a_route_for_an_unregistered_platform_is_written_back_on_save(tmp_path):
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope = store._routing_scope()
        saved = json.loads(json.dumps(live.to_dict()))
        saved.update(session_key="agent:main:yuanbao:dm:U1", platform="yuanbao", session_id="sid-yb",
                     origin={"platform": "yuanbao", "chat_id": "U1", "chat_type": "dm"})
        db.routing.upsert_routing_entry("agent:main:yuanbao:dm:U1", json.dumps(saved), routing_scope=scope)

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        with restarted._lock:
            restarted._save()
        rows = db.routing.load_routing_entries(routing_scope=scope)
        assert "agent:main:yuanbao:dm:U1" in rows
        assert json.loads(rows["agent:main:yuanbao:dm:U1"])["session_id"] == "sid-yb"


def test_an_older_sessions_json_copy_does_not_replace_the_held_db_route(tmp_path):
    """state.db is primary: the legacy sessions.json mirror never overwrites a held unparsed row."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope, key = store._routing_scope(), "agent:main:yuanbao:dm:U1"
        newer = {**live.to_dict(), "session_key": key, "platform": "yuanbao", "session_id": "sid-new",
                 "origin": {"platform": "yuanbao", "chat_id": "U1", "chat_type": "dm"}}
        db.routing.upsert_routing_entry(key, json.dumps(newer), routing_scope=scope)
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = {**newer, "session_id": "sid-old"}
        mirror.write_text(json.dumps(data), encoding="utf-8")

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))
        with restarted._lock:
            restarted._save()
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["session_id"] == "sid-new"


def _unregister(value):
    member = Platform._value2member_map_.pop(value, None)
    if member is not None:
        Platform._member_map_.pop(member._name_, None)


def _yb_source():
    return SessionSource(platform=Platform("yuanbao"), chat_id="U1", chat_type="dm")


def test_a_held_route_s_stop_still_applies_once_its_platform_registers(tmp_path):
    """A multiplexed profile's plugin registers after the shared index loaded: the held row joins the live
    index then, so /stop's suspension still starts a new session after the restart (#7536) instead of the stopped session being recovered from state.db."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()):
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        Platform._add_pseudo_member("yuanbao")  # the previous run had the plugin
        try:
            stuck = store.get_or_create_session(_yb_source())
            assert store.suspend_session(stuck.session_key)
            _unregister("yuanbao")
            restarted = SessionStore(store.sessions_dir, GatewayConfig())
            with restarted._lock:
                restarted._ensure_loaded_locked()  # boot: not registered yet, the row is held
            Platform._add_pseudo_member("yuanbao")  # the profile's plugin registers
            assert restarted.get_or_create_session(_yb_source()).session_id != stuck.session_id
        finally:
            _unregister("yuanbao")


def test_a_held_route_s_crash_left_turn_resumes_once_its_platform_registers(tmp_path):
    """The boot's turn-marker recovery runs before a multiplexed profile's plugin registers: the held row gets
    that verdict when it joins, so the interrupted turn is ``resume_pending``."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()):
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        Platform._add_pseudo_member("yuanbao")
        try:
            running = store.get_or_create_session(_yb_source())
            assert store.mark_turn_active(running.session_key)  # the previous run died mid-turn
            _unregister("yuanbao")
            restarted = SessionStore(store.sessions_dir, GatewayConfig())
            restarted.recover_interrupted_turns()  # boot: the row is held, the verdict is not lost
            Platform._add_pseudo_member("yuanbao")
            entry = restarted.get_or_create_session(_yb_source())
            assert (entry.session_id, entry.resume_pending, entry.active_turn_token) == (
                running.session_id, True, None)
        finally:
            _unregister("yuanbao")


def _held_profile_route(tmp_path, db, key):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
    row = {**live.to_dict(), "session_key": key, "platform": "yuanbao", "session_id": "sid-yb",
           "origin": {"platform": "yuanbao", "chat_id": "U1", "chat_type": "dm", "profile": "b"}}
    db.routing.upsert_routing_entry(key, json.dumps(row), routing_scope=store._routing_scope())
    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    with restarted._lock:
        restarted._ensure_loaded_locked()
    return restarted


def test_deleting_a_profile_drops_its_held_routes_too(tmp_path):
    """The live gateway's profile purge (#111926) must not be undone by the next save writing a held row back."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = _held_profile_route(tmp_path, db, "agent:b:yuanbao:dm:U1")
        store._routing_persistence.profile_identity.purge_profile("b")
        store.purge_profile_routing("b")
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        assert "agent:b:yuanbao:dm:U1" not in db.routing.load_routing_entries(routing_scope=store._routing_scope())


def test_renaming_a_profile_moves_its_held_routes_too(tmp_path):
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = _held_profile_route(tmp_path, db, "agent:b:yuanbao:dm:U1")
        store._routing_persistence.profile_identity.rekey_profile("b", "c")
        store.rekey_profile_routing("b", "c")
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        rows = db.routing.load_routing_entries(routing_scope=store._routing_scope())
        assert "agent:b:yuanbao:dm:U1" not in rows
        assert json.loads(rows["agent:c:yuanbao:dm:U1"])["origin"]["profile"] == "c"


def test_a_route_broken_another_way_is_not_held_and_the_mirror_copy_recovers_it(tmp_path):
    """Only an unregistered platform makes a row held: a DB row with a corrupt ``updated_at`` stays skipped, so
    the valid sessions.json copy loads and the next save writes the valid one."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope, key = store._routing_scope(), "agent:main:telegram:dm:t9"
        valid = {**live.to_dict(), "session_key": key, "session_id": "sid-ok",
                 "origin": {"platform": "telegram", "chat_id": "t9", "chat_type": "dm"}}
        db.routing.upsert_routing_entry(key, json.dumps({**valid, "updated_at": "not-a-date"}), routing_scope=scope)
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = valid
        mirror.write_text(json.dumps(data), encoding="utf-8")

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        with restarted._lock:
            restarted._ensure_loaded_locked()
            assert restarted._entries[key].session_id == "sid-ok"
            restarted._save()
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["updated_at"] != "not-a-date"


def _seed_held(tmp_path, db, *, turn_active=False):
    """A saved yuanbao route (as a run with the plugin wrote it), plus a live telegram one, in a fresh store."""
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    Platform._add_pseudo_member("yuanbao")
    try:
        entry = store.get_or_create_session(_yb_source())
        if turn_active:
            assert store.mark_turn_active(entry.session_key)
    finally:
        _unregister("yuanbao")
    store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
    return store, entry.session_key


def test_a_fallback_full_write_keeps_the_held_route(tmp_path, monkeypatch):
    """A turn-marker write whose single-row upsert fails falls back to a full rewrite: held rows go with it."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, key = _seed_held(tmp_path, db)
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        live = restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))

        def _fail(*_a, **_k):
            raise RuntimeError("disk I/O error")
        monkeypatch.setattr(type(db.routing), "upsert_routing_entry", _fail)
        assert restarted.mark_turn_active(live.session_key)
        assert key in db.routing.load_routing_entries(routing_scope=restarted._routing_scope())


def test_a_route_held_after_a_recovered_db_load_survives_the_next_save(tmp_path, monkeypatch):
    """state.db cannot be read at startup and recovers later: its unregistered-platform rows are held then too,
    and that newer copy wins over the mirror's older one."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, key = _seed_held(tmp_path, db)
        scope = store._routing_scope()
        newer = json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = {**newer, "session_id": "sid-old"}  # an older mirror copy
        mirror.write_text(json.dumps(data), encoding="utf-8")
        real_load, calls = type(db.routing).load_routing_entries, []

        def _first_fails(self, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            return real_load(self, *a, **k)
        monkeypatch.setattr(type(db.routing), "load_routing_entries", _first_fails)
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        monkeypatch.setattr(type(db.routing), "load_routing_entries", real_load)
        row = db.routing.load_routing_entries(routing_scope=scope)[key]
        assert json.loads(row)["session_id"] == newer["session_id"]


def test_a_clean_startup_clears_a_held_route_s_turn_marker_on_disk(tmp_path):
    """The clean-shutdown receipt is consumed right after the discard: a held row's stale marker must be gone
    from state.db then, or a crash before the plugin registers resumes a turn that had finished."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, key = _seed_held(tmp_path, db, turn_active=True)
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        assert restarted.discard_active_turn_markers() >= 1
        row = json.loads(db.routing.load_routing_entries(routing_scope=restarted._routing_scope())[key])
        assert (row["origin"]["platform"], row["active_turn_token"]) == ("yuanbao", None)


def test_a_recovered_db_load_keeps_the_boot_s_marker_discard_on_a_held_mirror_route(tmp_path, monkeypatch):
    """state.db unreadable at a clean startup: the mirror's held route gets the marker discard, and the DB
    retry inside that save must not put the old marker back."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, key = _seed_held(tmp_path, db, turn_active=True)
        scope = store._routing_scope()
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])
        mirror.write_text(json.dumps(data), encoding="utf-8")
        real_load, calls = type(db.routing).load_routing_entries, []

        def _first_fails(self, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            return real_load(self, *a, **k)
        monkeypatch.setattr(type(db.routing), "load_routing_entries", _first_fails)
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.discard_active_turn_markers()
        monkeypatch.setattr(type(db.routing), "load_routing_entries", real_load)
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["active_turn_token"] is None


def test_a_plugin_registering_before_the_db_recovers_keeps_the_newer_db_route(tmp_path, monkeypatch):
    """Mirror copy held at a failed DB load, the plugin registers, then the DB recovers: the newer DB row wins."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, key = _seed_held(tmp_path, db)
        scope = store._routing_scope()
        older = json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])
        newer = {**older, "resume_pending": True, "resume_reason": "restart_interrupted"}
        db.routing.upsert_routing_entry(key, json.dumps(newer), routing_scope=scope)
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = older
        mirror.write_text(json.dumps(data), encoding="utf-8")
        real_load, calls = type(db.routing).load_routing_entries, []

        def _first_fails(self, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            return real_load(self, *a, **k)
        monkeypatch.setattr(type(db.routing), "load_routing_entries", _first_fails)
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        with restarted._lock:
            restarted._ensure_loaded_locked()  # DB unreadable: the mirror's older copy is held
        Platform._add_pseudo_member("yuanbao")  # the plugin registers before the DB recovers
        try:
            with restarted._lock:
                restarted._ensure_loaded_locked()
                assert restarted._entries[key].resume_pending is True
                restarted._save()
        finally:
            _unregister("yuanbao")
        monkeypatch.setattr(type(db.routing), "load_routing_entries", real_load)
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["resume_pending"] is True


def _durable_keys(store, db):
    """Route keys in both durable copies: state.db's routing table and the sessions.json mirror."""
    mirror = json.loads((store.sessions_dir / "sessions.json").read_text(encoding="utf-8"))
    return set(db.routing.load_routing_entries(routing_scope=store._routing_scope())), set(mirror) - {"_README"}


def _seed_two_held(tmp_path):
    """Two saved yuanbao routes (a run with the plugin wrote them) and a live telegram one; the restarted
    store has no yuanbao plugin, so both yuanbao routes are held."""
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    Platform._add_pseudo_member("yuanbao")
    try:
        doomed = store.get_or_create_session(_yb_source())
        kept = store.get_or_create_session(SessionSource(platform=Platform("yuanbao"), chat_id="U2", chat_type="dm"))
    finally:
        _unregister("yuanbao")
    live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    with restarted._lock:
        restarted._ensure_loaded_locked()
    return restarted, doomed, kept, live


def test_hard_deleting_a_session_drops_its_held_route_from_both_durable_copies(tmp_path):
    """The delete endpoint's removal (#42422) covers a held route: gone at once and after an unrelated save,
    other routes stay, a repeat is a no-op, and the plugin registering later starts a fresh session."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        from gateway.platforms.api_server_store import delete_session
        store, doomed, kept, live = _seed_two_held(tmp_path)
        assert delete_session(db, doomed.session_id, store.sessions_dir)  # the endpoint's order: rows, then routes
        assert store.remove_by_session_id(doomed.session_id) == 1
        for keys in _durable_keys(store, db):
            assert doomed.session_key not in keys and {kept.session_key, live.session_key} <= keys
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # unrelated full save
        for keys in _durable_keys(store, db):
            assert doomed.session_key not in keys and {kept.session_key, live.session_key} <= keys
        assert store.remove_by_session_id(doomed.session_id) == 0
        Platform._add_pseudo_member("yuanbao")
        try:
            assert store.get_or_create_session(_yb_source()).session_id != doomed.session_id
            assert store.get_or_create_session(
                SessionSource(platform=Platform("yuanbao"), chat_id="U2", chat_type="dm")).session_id == kept.session_id
        finally:
            _unregister("yuanbao")


def _rewrite_held_row(store, db, key, **fields):
    """Change a held route's saved row, then restart so the new store holds that copy."""
    scope = store._routing_scope()
    row = json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])
    db.routing.upsert_routing_entry(key, json.dumps({**row, **fields}), routing_scope=scope)
    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    with restarted._lock:
        restarted._ensure_loaded_locked()
    return restarted


def test_hard_deleting_a_session_drops_its_live_and_held_routes_together(tmp_path):
    """Session S has a live telegram route and a held yuanbao route: both go in one delete, counted once each."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, held, _kept, live = _seed_two_held(tmp_path)
        store = _rewrite_held_row(store, db, held.session_key, session_id=live.session_id)
        assert store.remove_by_session_id(live.session_id) == 2
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))
        for keys in _durable_keys(store, db):
            assert not {held.session_key, live.session_key} & keys


def test_expiring_an_idle_route_drops_an_idle_held_route_too(tmp_path):
    """Routing expiry covers a held route: an idle one is pruned like a live one, a fresh one stays."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store, idle, fresh, _live = _seed_two_held(tmp_path)
        store = _rewrite_held_row(store, db, idle.session_key, updated_at="2001-01-01T00:00:00")
        assert store.prune_old_entries(30) == 1
        for keys in _durable_keys(store, db):
            assert idle.session_key not in keys and fresh.session_key in keys
