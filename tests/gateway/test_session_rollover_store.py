"""Durable gateway-session rollover receipt and cutover behavior."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from threading import Event, Thread

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB
from hermes_state_errors import SessionCompressionInProgressError, SessionTurnLeaseLostError


def _store(tmp_path, db_path=None):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    db = SessionDB(db_path=db_path or tmp_path / "state.db")
    store._db = db
    return store, db


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="rollover-chat", user_id="rollover-user",
    )


def _shared_thread_source(user_id, *, scope_id="shared-scope"):
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="shared-chat", chat_type="group",
        thread_id="shared-thread", scope_id=scope_id, user_id=user_id,
    )


def _binding(tmp_path):
    packet = (tmp_path / "handoff.md").resolve()
    restart_message = "Read the handoff packet and continue the authorized task."
    wrapper = "Continue with no new authority. First read the bound packet."
    return {
        "profile_home": str(tmp_path.resolve()),
        "packet_path": str(packet),
        "packet_sha256": hashlib.sha256(b"immutable packet").hexdigest(),
        "restart_message": restart_message,
        "bootstrap_prompt": f"{wrapper}\n\n{restart_message}",
        "reason": "context_pressure",
    }


def _prepare(store, entry, tmp_path, generation=7):
    binding = _binding(tmp_path)
    authoring = store.begin_rollover(
        entry.session_key, entry.session_id, generation,
        profile_home=binding["profile_home"], packet_path=binding["packet_path"],
    )
    assert authoring["phase"] == "authoring"
    assert authoring["owner_committed"] is False
    prepared = store.prepare_rollover(
        entry.session_key, entry.session_id, generation, binding,
        expected_new_session_id=authoring["new_session_id"],
    )
    assert prepared["new_session_id"] == authoring["new_session_id"]
    return prepared


def test_precommit_failure_keeps_old_owner(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        receipt = _prepare(store, old, tmp_path)
        db._conn.execute(
            "CREATE TRIGGER fail_rollover_insert BEFORE INSERT ON sessions "
            f"WHEN NEW.id = '{receipt['new_session_id']}' "
            "BEGIN SELECT RAISE(ABORT, 'rollover insert failure'); END"
        )

        with pytest.raises(sqlite3.IntegrityError, match="rollover insert failure"):
            store.commit_rollover(
                old.session_key, old.session_id, 7,
                expected_new_session_id=receipt["new_session_id"],
            )

        assert store.peek_session_id(old.session_key) == old.session_id
        assert db.get_session(old.session_id)["end_reason"] is None
        assert db.get_session(receipt["new_session_id"]) is None
        assert store.get_rollover(old.session_key)["phase"] == "prepared"
    finally:
        db.close()


def test_committed_successor_recovers_after_route_publication_failure(tmp_path, monkeypatch):
    store, db = _store(tmp_path)
    recovered_db = None
    try:
        source = _source()
        old = store.get_or_create_session(source)
        receipt = _prepare(store, old, tmp_path)

        publication_failures = []
        routing_db_method = store._routing_db_method

        def fail_routing_index(name):
            if name == "replace_gateway_routing_entries":
                def fail(*_args, **_kwargs):
                    publication_failures.append("routing-db")
                    raise OSError("routing database unavailable")
                return fail
            return routing_db_method(name)

        def fail_sessions_json(*_args, **_kwargs):
            publication_failures.append("sessions-json")
            raise OSError("sessions.json unavailable")

        monkeypatch.setattr(store, "_routing_db_method", fail_routing_index)
        monkeypatch.setattr(store, "_save_sessions_json", fail_sessions_json)
        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        assert publication_failures == ["routing-db", "sessions-json"]
        assert successor.session_id == receipt["new_session_id"]
        assert store.get_rollover(old.session_key)["phase"] == "committed"

        retry = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        assert retry.session_id == successor.session_id
        assert db.get_session(old.session_id)["end_reason"] == "session_reset"

        recovered_store, recovered_db = _store(tmp_path, db_path=db.db_path)
        recovered = recovered_store.get_or_create_session(source)
        assert recovered.session_id == receipt["new_session_id"]
        assert recovered.is_fresh_reset is True
        assert recovered.prev_session_id == old.session_id
        assert recovered_store.get_rollover(old.session_key)["new_session_id"] == recovered.session_id
    finally:
        if recovered_db is not None:
            recovered_db.close()
        db.close()


def test_cutover_persists_reset_lineage_and_route_identity(tmp_path):
    store, db = _store(tmp_path)
    try:
        source = _source()
        old = store.get_or_create_session(source)
        store.set_model_override(
            old.session_key, {"model": "test-model", "provider": "test-provider"}
        )
        pin = {
            "version": 1, "context_key": "telegram:rollover-chat", "context_prompt": "channel context",
            "redact_pii": False, "channel_prompt": None, "parent_chat_id": None,
        }
        assert store.set_prompt_pin(old.session_key, pin, expected_session_id=old.session_id)
        receipt = _prepare(store, old, tmp_path)

        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        parent_row = db.get_session(old.session_id)
        successor_row = db.get_session(successor.session_id)

        assert successor.session_id == receipt["new_session_id"]
        assert successor.is_fresh_reset is True
        assert successor.prev_session_id == old.session_id
        assert successor.model_override == {"model": "test-model", "provider": "test-provider"}
        assert successor.prompt_pin == pin
        assert parent_row["end_reason"] == "session_reset"
        assert successor_row["parent_session_id"] == old.session_id
        assert successor_row["_reset_from"] == old.session_id
        assert successor_row["source"] == source.platform.value
        assert successor_row["session_key"] == old.session_key
        assert successor_row["user_id"] == source.user_id
        assert successor_row["chat_id"] == source.chat_id
        assert successor_row["system_prompt"] is None
        assert store.load_transcript(successor.session_id) == []
    finally:
        db.close()


def test_interrupted_authoring_blocks_with_old_owner(tmp_path):
    store, db = _store(tmp_path)
    restarted_db = None
    try:
        old = store.get_or_create_session(_source())
        binding = _binding(tmp_path)
        receipt = store.begin_rollover(
            old.session_key, old.session_id, 19,
            profile_home=binding["profile_home"], packet_path=binding["packet_path"],
        )

        restarted, restarted_db = _store(tmp_path, db_path=db.db_path)
        pending = restarted.list_pending_rollovers(recover_interrupted=True)

        assert len(pending) == 1
        assert pending[0]["phase"] == "blocked"
        assert pending[0]["owner_committed"] is False
        assert pending[0]["new_session_id"] == receipt["new_session_id"]
        assert restarted.peek_session_id(old.session_key) == old.session_id
        assert restarted_db.get_session(old.session_id)["end_reason"] is None
    finally:
        if restarted_db is not None:
            restarted_db.close()
        db.close()

def test_cutover_waits_for_native_turn_and_compression_leases(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        receipt = _prepare(store, old, tmp_path)
        turn_holder = "rollover-test-turn"
        assert db.try_acquire_session_turn_lease(old.session_id, turn_holder)

        with pytest.raises(SessionTurnLeaseLostError, match="active turn lease"):
            store.commit_rollover(
                old.session_key, old.session_id, 7,
                expected_new_session_id=receipt["new_session_id"],
            )
        assert db.get_session(old.session_id)["end_reason"] is None
        db.release_session_turn_lease(old.session_id, turn_holder)

        compression_holder = "rollover-test-compressor"
        assert db.try_acquire_compression_lock(old.session_id, compression_holder)
        with pytest.raises(SessionCompressionInProgressError, match="being compressed"):
            store.commit_rollover(
                old.session_key, old.session_id, 7,
                expected_new_session_id=receipt["new_session_id"],
            )
        assert db.get_session(old.session_id)["end_reason"] is None
        db.release_compression_lock(old.session_id, compression_holder)

        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        assert store.get_rollover(old.session_key)["new_session_id"] == successor.session_id
    finally:
        db.close()


@pytest.mark.parametrize("stage", ("begin", "prepare", "commit"))
def test_unrelated_route_progress_during_rollover_database_wait(tmp_path, monkeypatch, stage):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        binding = _binding(tmp_path)
        if stage == "begin":
            method_name = "begin_gateway_rollover"
            operation = lambda: store.begin_rollover(
                old.session_key, old.session_id, 7,
                profile_home=binding["profile_home"], packet_path=binding["packet_path"],
            )
        elif stage == "prepare":
            receipt = store.begin_rollover(
                old.session_key, old.session_id, 7,
                profile_home=binding["profile_home"], packet_path=binding["packet_path"],
            )
            method_name = "prepare_gateway_rollover"
            operation = lambda: store.prepare_rollover(
                old.session_key, old.session_id, 7, binding,
                expected_new_session_id=receipt["new_session_id"],
            )
        else:
            receipt = _prepare(store, old, tmp_path)
            method_name = "commit_gateway_rollover"
            operation = lambda: store.commit_rollover(
                old.session_key, old.session_id, 7,
                expected_new_session_id=receipt["new_session_id"],
            )

        database_waiting = Event()
        release_database = Event()
        unrelated_done = Event()
        result = []
        unrelated = []
        failures = []
        original = getattr(db, method_name)

        def wait_for_database(*args, **kwargs):
            database_waiting.set()
            if not release_database.wait(timeout=5):
                raise TimeoutError("test did not release rollover database wait")
            return original(*args, **kwargs)

        def rollover():
            try:
                result.append(operation())
            except Exception as exc:
                failures.append(exc)

        def create_unrelated_route():
            try:
                unrelated.append(store.get_or_create_session(SessionSource(
                    platform=Platform.TELEGRAM, chat_id="unrelated-chat", user_id="other-user",
                )))
            except Exception as exc:
                failures.append(exc)
            finally:
                unrelated_done.set()

        monkeypatch.setattr(db, method_name, wait_for_database)
        rollover_thread = Thread(target=rollover)
        unrelated_thread = Thread(target=create_unrelated_route)
        rollover_thread.start()
        try:
            assert database_waiting.wait(timeout=2)
            unrelated_thread.start()
            assert unrelated_done.wait(timeout=2)
        finally:
            release_database.set()
            rollover_thread.join(timeout=5)
            if unrelated_thread.ident is not None:
                unrelated_thread.join(timeout=5)

        assert not rollover_thread.is_alive()
        assert not unrelated_thread.is_alive()
        assert not failures
        assert unrelated[0].session_key != old.session_key
        if stage == "commit":
            assert result[0].session_id == receipt["new_session_id"]
        else:
            assert result[0]["phase"] == {"begin": "authoring", "prepare": "prepared"}[stage]
    finally:
        db.close()


def test_newer_manual_reset_and_receipt_win_during_rollover_database_wait(tmp_path, monkeypatch):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        stale_receipt = _prepare(store, old, tmp_path)
        commit = db.commit_gateway_rollover
        commit_waiting = Event()
        release_commit = Event()
        reset_done = Event()
        failures = []
        reset_result = []

        def wait_for_commit(*args, **kwargs):
            commit_waiting.set()
            if not release_commit.wait(timeout=5):
                raise TimeoutError("test did not release rollover database wait")
            return commit(*args, **kwargs)

        def commit_rollover():
            try:
                store.commit_rollover(
                    old.session_key, old.session_id, 7,
                    expected_new_session_id=stale_receipt["new_session_id"],
                )
            except Exception as exc:
                failures.append(exc)

        def reset_and_prepare_new_attempt():
            try:
                deliberate = store.reset_session(old.session_key)
                if deliberate is None:
                    raise AssertionError("manual reset did not retain the route")
                reset_result.append(deliberate)
                reset_result.append(_prepare(store, deliberate, tmp_path, generation=8))
            except Exception as exc:
                failures.append(exc)
            finally:
                reset_done.set()

        monkeypatch.setattr(db, "commit_gateway_rollover", wait_for_commit)
        rollover_thread = Thread(target=commit_rollover)
        reset_thread = Thread(target=reset_and_prepare_new_attempt)
        rollover_thread.start()
        try:
            assert commit_waiting.wait(timeout=2)
            reset_thread.start()
            assert reset_done.wait(timeout=2)
        finally:
            release_commit.set()
            rollover_thread.join(timeout=5)
            if reset_thread.ident is not None:
                reset_thread.join(timeout=5)

        assert not rollover_thread.is_alive()
        assert not reset_thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        deliberate, newer_receipt = reset_result
        assert store.peek_session_id(old.session_key) == deliberate.session_id
        current_receipt = store.get_rollover(old.session_key)
        assert current_receipt["new_session_id"] == newer_receipt["new_session_id"]
        assert current_receipt["phase"] == "prepared"
    finally:
        db.close()


def test_interrupted_admission_recovers_same_successor_for_shared_thread_sender(
    tmp_path, monkeypatch,
):
    store, db = _store(tmp_path)
    restarted_db = None
    try:
        source = _shared_thread_source("first-user")
        other_sender = _shared_thread_source("second-user")
        old = store.get_or_create_session(source)
        receipt = _prepare(store, old, tmp_path)

        def fail_route_persistence(*_args, **_kwargs):
            raise OSError("route publication unavailable")

        monkeypatch.setattr(store, "_persist_routing_data", fail_route_persistence)
        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        db.set_gateway_rollover_phase(
            old.session_key, receipt["new_session_id"], "committed", "bootstrapping",
        )
        db.set_gateway_rollover_phase(
            old.session_key, receipt["new_session_id"], "bootstrapping", "admitting",
        )

        restarted, restarted_db = _store(tmp_path, db_path=db.db_path)
        pending = restarted.list_pending_rollovers(recover_interrupted=True)

        assert len(pending) == 1
        assert pending[0]["phase"] == "blocked"
        assert pending[0]["new_session_id"] == successor.session_id == receipt["new_session_id"]
        assert restarted.peek_session_id(old.session_key) == successor.session_id
        assert restarted.get_or_create_session(other_sender).session_id == successor.session_id

        isolated_sources = [
            SessionSource(
                platform=Platform.TELEGRAM, chat_id="shared-chat", chat_type="group",
                user_id=user_id,
            )
            for user_id in ("first-user", "second-user")
        ]
        isolated_entries = [restarted.get_or_create_session(source) for source in isolated_sources]
        assert isolated_entries[0].session_key != isolated_entries[1].session_key
        assert isolated_entries[0].session_id != isolated_entries[1].session_id
        with pytest.raises(RuntimeError, match="bound route"):
            restarted.get_or_create_session(_shared_thread_source("second-user", scope_id="other-scope"))

        with pytest.raises(ValueError, match="blocked -> bootstrapping"):
            restarted.set_rollover_phase(
                old.session_key, receipt["new_session_id"], "blocked", "bootstrapping",
            )
    finally:
        if restarted_db is not None:
            restarted_db.close()
        db.close()


def test_prepared_receipt_rejects_stale_generation_and_phase_cas(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        receipt = _prepare(store, old, tmp_path, generation=11)

        with pytest.raises(ValueError, match="generation"):
            store.commit_rollover(
                old.session_key, old.session_id, 12,
                expected_new_session_id=receipt["new_session_id"],
            )

        with pytest.raises(ValueError, match="attempt changed"):
            store.prepare_rollover(
                old.session_key, old.session_id, 11, _binding(tmp_path),
                expected_new_session_id="stale-attempt",
            )
        assert store.get_rollover(old.session_key)["new_session_id"] == receipt["new_session_id"]

        with pytest.raises(ValueError, match="phase"):
            store.set_rollover_phase(
                old.session_key, receipt["new_session_id"], "committed", "bootstrapping",
            )
        assert store.peek_session_id(old.session_key) == old.session_id
        assert db.get_session(old.session_id)["end_reason"] is None
    finally:
        db.close()


def test_profile_rollover_receipt_uses_the_lane_database(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig(multiplex_profiles=True))
    try:
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="profile-rollover", user_id="owner", profile="work",
        )
        old = store.get_or_create_session(source)
        profile_db = store._db_for_key(old.session_key)
        receipt = _prepare(store, old, profile_home)
        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )

        assert receipt["profile_home"] == str(profile_home.resolve())
        assert profile_db.get_gateway_rollover(old.session_key)["new_session_id"] == successor.session_id
        assert store._routing_db.get_gateway_rollover(old.session_key) is None
        assert profile_db.get_session(old.session_id)["end_reason"] == "session_reset"
        assert profile_db.get_session(successor.session_id)["parent_session_id"] == old.session_id
    finally:
        store.close_all_db_handles()


def test_stale_receipt_cannot_override_an_explicit_reset(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        receipt = _prepare(store, old, tmp_path)

        deliberate = store.reset_session(old.session_key)
        assert deliberate is not None
        assert deliberate.session_id != old.session_id
        with pytest.raises(ValueError, match="route changed"):
            store.commit_rollover(
                old.session_key, old.session_id, 7,
                expected_new_session_id=receipt["new_session_id"],
            )

        assert store.get_or_create_session(_source()).session_id == deliberate.session_id
        assert store.get_rollover(old.session_key)["phase"] == "invalidated"
        assert db.get_session(old.session_id)["end_reason"] == "session_reset"
    finally:
        db.close()


def test_reconnect_listing_never_marks_live_authoring_as_interrupted(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        binding = _binding(tmp_path)
        receipt = store.begin_rollover(
            old.session_key, old.session_id, 9,
            profile_home=binding["profile_home"], packet_path=binding["packet_path"],
        )
        assert store.list_pending_rollovers()[0]["phase"] == "authoring"
        store.set_rollover_phase(old.session_key, receipt["new_session_id"], "authoring", "invalidated")
        assert store.list_pending_rollovers(recover_interrupted=True) == []
    finally:
        db.close()


def test_pending_receipt_prevents_automatic_route_replacement(tmp_path):
    store, db = _store(tmp_path)
    try:
        old = store.get_or_create_session(_source())
        receipt = _prepare(store, old, tmp_path)
        old.suspended = True
        assert store.get_or_create_session(_source()).session_id == old.session_id
        assert db.get_session(receipt["new_session_id"]) is None
        assert db.get_session(old.session_id)["end_reason"] is None
    finally:
        db.close()


def test_startup_recovery_does_not_mutate_unserved_profile_receipts(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    homes = {name: home / "profiles" / name for name in ("served", "unserved")}
    for directory in homes.values():
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig(multiplex_profiles=True))
    try:
        entries = {}
        for name, directory in homes.items():
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id=name, user_id="owner", profile=name,
            ))
            entries[name] = entry
            _prepare(store, entry, directory)
        pending = store.list_pending_rollovers(
            profile_homes=[homes["served"]], recover_interrupted=True,
        )
        assert [receipt["session_key"] for receipt in pending] == [entries["served"].session_key]
        assert pending[0]["phase"] == "blocked"
        assert store.get_rollover(entries["unserved"].session_key)["phase"] == "prepared"
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_slack_downtime_delta_survives_rollover_and_route_recovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    store, db = _store(tmp_path)
    recovered_db = None
    try:
        adapter = SlackAdapter(PlatformConfig(enabled=True))
        adapter.set_session_store(store)
        source = adapter._thread_session_source("C123", "123.000", "U_USER", "T_TEAM", "group")
        old = store.get_or_create_session(source)
        store.set_session_metadata(old.session_key, "slack_thread_watermark:C123:123.000", "123.100")
        receipt = _prepare(store, old, tmp_path)

        def unavailable_index(*_args):
            raise OSError("routing publication unavailable")

        monkeypatch.setattr(store, "_persist_routing_data", unavailable_index)
        successor = store.commit_rollover(
            old.session_key, old.session_id, 7,
            expected_new_session_id=receipt["new_session_id"],
        )
        restarted, recovered_db = _store(tmp_path, db_path=db.db_path)
        assert restarted.get_or_create_session(source).session_id == successor.session_id
        adapter = SlackAdapter(PlatformConfig(enabled=True))
        adapter.set_session_store(restarted)
        adapter._bot_user_id = "U_BOT"
        adapter._team_bot_user_ids = {"T_TEAM": "U_BOT"}
        adapter._user_name_cache = {("T_TEAM", "U_USER"): "User", ("T_TEAM", "U_OTHER"): "Other"}
        adapter._app = SimpleNamespace(client=SimpleNamespace(conversations_replies=AsyncMock(
            return_value={"messages": [
                {"ts": "123.100", "user": "U_USER", "text": "Already observed"},
                {"ts": "123.200", "user": "U_OTHER", "text": "Missed during downtime"},
                {"ts": "123.300", "user": "U_USER", "text": "Current reply"},
            ]},
        )))
        context, _, _ = await adapter._hydrate_thread_context(
            channel_id="C123", event_thread_ts="123.000", ts="123.300",
            user_id="U_USER", team_id="T_TEAM", is_thread_reply=True, is_mentioned=False, is_dm=False,
        )
        assert "Missed during downtime" in (context or "")
        assert "Already observed" not in context
        assert "Current reply" not in context
    finally:
        if recovered_db is not None:
            recovered_db.close()
        db.close()
