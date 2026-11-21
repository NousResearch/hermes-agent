"""``GET /api/skills/hub/updates`` must not hold the process-wide skills lock while it fetches.

The route runs the hub update check, which downloads every installed hub skill from its recorded
source — minutes on a slow link — and used to do that inside ``_profile_scope``: the process-wide
``_SKILLS_PROFILE_LOCK`` that every config/skills/tools request takes first. One slow check then
stalled ``GET /api/skills`` and ``GET /api/tools/toolsets`` until the desktop's 30 s timeout
(#155069).

The check itself resolves its paths through the ``HERMES_HOME`` contextvar (``tools/skills_hub``'s
per-call path resolvers) and touches neither ``tools.skills_tool.SKILLS_DIR`` nor
``tools.skill_manager_tool``'s globals, so the lock-free ``config_scoped_to_thread`` scope carries
everything it reads.

Second half: a client that gives up cannot cancel an ``asyncio.to_thread`` worker, so a request
arriving while a check runs must JOIN that check instead of queueing another full fetch, and an
abandoned request must not cancel it for everyone else.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("fastapi")


def _client():
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


def _updates_route():
    from hermes_cli.web_routers.skills import list_skill_hub_updates

    return list_skill_hub_updates


def test_a_slow_update_check_does_not_block_a_skills_request(monkeypatch, _isolate_hermes_home):
    client = _client()
    in_fetch, release = threading.Event(), threading.Event()

    def _slow_check():
        in_fetch.set()
        release.wait(timeout=30)
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _slow_check)
    results: dict[str, int] = {}

    def _get(name: str, path: str) -> None:
        results[name] = client.get(path).status_code

    updates = threading.Thread(target=_get, args=("updates", "/api/skills/hub/updates"), daemon=True)
    updates.start()
    try:
        assert in_fetch.wait(timeout=10), "the update route never reached the hub fetch"
        skills = threading.Thread(target=_get, args=("skills", "/api/skills"), daemon=True)
        skills.start()
        skills.join(timeout=10)
        assert not skills.is_alive(), "a skills request waited for the hub update fetch"
        assert results["skills"] == 200
    finally:
        release.set()
        updates.join(timeout=10)
    assert results["updates"] == 200


def test_a_request_arriving_mid_check_joins_it(monkeypatch, _isolate_hermes_home):
    gate, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def _check():
        calls.append(1)
        gate.set()
        release.wait(timeout=30)
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _check)
    route = _updates_route()

    async def _drive():
        first = asyncio.create_task(route(None))
        assert await asyncio.to_thread(gate.wait, 10), "the first request never reached the fetch"
        second = asyncio.create_task(route(None))
        for _ in range(10):
            await asyncio.sleep(0)  # let the second request reach the in-flight check
        assert len(calls) == 1, "a second request started its own full hub fetch"
        release.set()
        assert await first == {"updates": []}
        assert await second == {"updates": []}

    asyncio.run(_drive())
    assert len(calls) == 1


def test_an_abandoned_request_does_not_cancel_the_running_check(monkeypatch, _isolate_hermes_home):
    gate, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def _check():
        calls.append(1)
        gate.set()
        release.wait(timeout=30)
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _check)
    route = _updates_route()

    async def _drive():
        first = asyncio.create_task(route(None))
        assert await asyncio.to_thread(gate.wait, 10), "the first request never reached the fetch"
        abandoned = asyncio.create_task(route(None))
        for _ in range(10):
            await asyncio.sleep(0)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        release.set()
        assert await first == {"updates": []}

    asyncio.run(_drive())
    assert len(calls) == 1, "the abandoned request started (or restarted) a hub fetch"


def test_a_named_profile_check_reads_that_profiles_home(monkeypatch, _isolate_hermes_home):
    """The lock-free scope carries the profile's home into the worker: it decides which homes
    the hub lock file and skill directories resolve to, so getting it wrong would report another
    profile's updates."""
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    seen: list[str] = []

    def _check():
        seen.append(str(get_hermes_home()))
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _check)

    assert asyncio.run(_updates_route()("worker_alpha")) == {"updates": []}
    assert seen == [str(worker_home)]


def test_a_failed_check_does_not_wedge_the_slot(monkeypatch, _isolate_hermes_home):
    """A raising check must clear the in-flight slot, or every later page load answers 502."""
    from fastapi import HTTPException

    calls: list[int] = []

    def _check():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("hub index unreachable")
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _check)
    route = _updates_route()

    with pytest.raises(HTTPException) as failed:
        asyncio.run(route(None))
    assert failed.value.status_code == 502
    assert asyncio.run(route(None)) == {"updates": []}
    assert len(calls) == 2


def test_a_later_request_checks_again(monkeypatch, _isolate_hermes_home):
    calls: list[int] = []

    def _check():
        calls.append(1)
        return []

    monkeypatch.setattr("tools.skills_hub_install.check_for_skill_updates", _check)
    route = _updates_route()

    assert asyncio.run(route(None)) == {"updates": []}
    assert asyncio.run(route(None)) == {"updates": []}
    assert len(calls) == 2, "the finished check was never released for the next request"
