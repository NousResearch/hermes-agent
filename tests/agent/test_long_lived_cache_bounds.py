"""Module-level caches in a long-lived gateway shed only entries nobody can use any more.

Each test pins the memory contract (rotated credentials, abandoned scopes, closed loops and long
idle sessions do not stay resident) and the capability contract next to it (whatever is live or
recently used keeps being served from the cache).
"""

from __future__ import annotations

import asyncio

import pytest

import agent.agent_runtime_helpers as arh
import agent.auxiliary_client as aux
import agent.model_metadata as mm
import agent.skill_commands as sc


def test_endpoint_catalog_memo_drops_expired_credentials_on_write(monkeypatch):
    monkeypatch.setattr(mm, "_endpoint_model_metadata_cache", {})
    monkeypatch.setattr(mm, "_endpoint_model_metadata_cache_time", {})
    now = 1_000_000.0
    monkeypatch.setattr(mm.time, "time", lambda: now)
    rotated, live = ("https://inference.example/v1", "fp-old"), ("https://inference.example/v1", "fp-live")
    mm._endpoint_model_metadata_cache.update({rotated: {"m": {}}, live: {"m": {}}})
    mm._endpoint_model_metadata_cache_time.update({rotated: now - mm._ENDPOINT_MODEL_CACHE_TTL - 1, live: now - 10})

    fresh = ("https://inference.example/v1", "fp-new")
    assert mm._remember_endpoint_models(fresh, {"m": {"context_length": 1}}) == {"m": {"context_length": 1}}

    assert set(mm._endpoint_model_metadata_cache) == {live, fresh}
    assert set(mm._endpoint_model_metadata_cache_time) == {live, fresh}


def test_codex_context_memos_keep_the_newest_128_tokens(monkeypatch):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {f"tok{i}": ({"gpt": 1}, 1000.0 + i) for i in range(128)})
    monkeypatch.setattr(mm, "_codex_oauth_max_context_cache", {f"tok{i}": {"gpt": 2} for i in range(128)})

    mm._remember_no_codex_catalog("tok-new")

    assert len(mm._codex_oauth_context_cache) == 128
    assert "tok0" not in mm._codex_oauth_context_cache and "tok0" not in mm._codex_oauth_max_context_cache
    assert "tok-new" in mm._codex_oauth_context_cache and "tok127" in mm._codex_oauth_max_context_cache
    mm._remember_no_codex_catalog("tok127")  # re-storing a known token never evicts another
    assert len(mm._codex_oauth_context_cache) == 128 and "tok1" in mm._codex_oauth_context_cache


def test_skill_command_scopes_are_lru_bounded_and_recent_scope_stays_warm(monkeypatch):
    import tools.skills_tool as st
    project = {"value": "/repo/0"}
    monkeypatch.setattr(sc, "_skill_commands_by_key", {})
    monkeypatch.setattr(sc, "_skill_commands_sig_by_key", {})
    monkeypatch.setattr(sc, "_last_interactive_skill_commands_by_key", {})
    monkeypatch.setattr(sc, "_resolve_skill_commands_platform", lambda: "local")
    monkeypatch.setattr(sc, "_resolve_skill_commands_home", lambda: "/home/profile")
    monkeypatch.setattr(sc, "_resolve_skill_commands_project", lambda: project["value"])
    monkeypatch.setattr(sc, "_skill_dirs_signature", lambda: ("sig",))
    monkeypatch.setattr(sc, "get_plugin_skill_commands", dict)
    scans = []
    monkeypatch.setattr(st, "_skill_catalog", lambda: scans.append(project["value"]) or [])

    for i in range(200):
        project["value"] = "/repo/0"  # the scope a live session keeps using
        sc.get_interactive_skill_commands()
        project["value"] = f"/repo/{i + 1}"
        sc.get_interactive_skill_commands()

    for table in (sc._skill_commands_by_key, sc._skill_commands_sig_by_key, sc._last_interactive_skill_commands_by_key):
        assert len(table) <= 2 * sc._MAX_WATCHED_SCOPES
        assert ("local", "/home/profile", "/repo/0") in table
    assert scans.count("/repo/0") == 1  # never evicted, never rescanned

    project["value"] = "/repo/1"  # long evicted: rescans on its next use, same map as before
    assert sc.get_skill_commands() == {} and scans.count("/repo/1") == 2


def test_async_aux_semaphores_do_not_outlive_their_event_loop(monkeypatch):
    aux._reset_aux_semaphores()
    monkeypatch.setattr(aux, "_get_task_max_concurrency", lambda task: 1)

    async def contend():
        # Two callers on a limit of 1: the second waits, which binds the semaphore to this loop.
        sem = aux._acquire_async_aux_semaphore("compression")

        async def hold():
            async with sem:
                await asyncio.sleep(0)

        await asyncio.gather(hold(), hold())
        return sem

    sems = []
    for _ in range(5):
        loop = asyncio.new_event_loop()
        sems.append(loop.run_until_complete(contend()))
        loop.close()
        del loop

    live = asyncio.new_event_loop()
    try:
        sems.append(live.run_until_complete(contend()))  # a fresh loop never gets a closed loop's semaphore
        assert list(aux._aux_async_semaphores) == [live]
    finally:
        live.close()
    assert len({id(s) for s in sems}) == len(sems)
    aux._reset_aux_semaphores()


@pytest.fixture
def heal_state():
    for table in (arh._empty_heal_log_state, arh._empty_heal_pending_notice, arh._empty_heal_user_notified):
        table.clear()
    yield
    for table in (arh._empty_heal_log_state, arh._empty_heal_pending_notice, arh._empty_heal_user_notified):
        table.clear()


def test_heal_counters_are_lru_bounded_and_no_session_is_ever_renotified(monkeypatch, heal_state):
    session = {"id": "busy"}
    monkeypatch.setattr(arh, "_session_id_for_heal_log", lambda: session["id"])
    monkeypatch.setattr(arh, "_heal_escalation_threshold", lambda: 1)

    for sid in ("busy", "quiet"):
        session["id"] = sid
        arh._log_empty_non_final_heal(1)
        assert arh.consume_pending_sanitizer_heal_notice()
    for i in range(3000):
        session["id"] = f"s{i}"
        arh._log_empty_non_final_heal(1)
        arh.consume_pending_sanitizer_heal_notice()
        if i % 100 == 0:
            session["id"] = "busy"
            arh._log_empty_non_final_heal(1)

    assert len(arh._empty_heal_log_state) <= 1024
    assert "busy" in arh._empty_heal_log_state and "quiet" not in arh._empty_heal_log_state
    for sid in ("busy", "quiet"):  # the long-forgotten session heals again: still no second notice
        session["id"] = sid
        arh._log_empty_non_final_heal(1)
        assert arh.consume_pending_sanitizer_heal_notice() is None
    assert arh.get_sanitizer_heal_stats()["busy"]["heal_events"] == 32
