"""Remote config: reads come from the plane, the poller, outages and the multi-group plane shape."""
from __future__ import annotations

import json

import pytest

from hermes_cli.config_backend import (
    ConfigLockedError,
    get_config_backend,
    write_config_key,
)
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from utils import fast_safe_load

from .conftest import _gets
from .stub_plane import INSTANCE, TOKEN


def test_reads_come_from_plane_not_local_file(plane):
    from hermes_cli.config import load_config
    plane.upper = {"model": {"default": "hermes-4", "provider": "nous"}, "terminal": {"timeout": 180}}
    local = plane.home / "config.yaml"
    local.write_text(json.dumps({"model": {"default": "LOCAL"}, "terminal": {"timeout": 1}}))

    cfg = load_config()

    assert cfg["model"]["default"] == "hermes-4"
    assert cfg["terminal"]["timeout"] == 180
    assert "display" in cfg  # DEFAULT_CONFIG stays under the remote user layer
    assert fast_safe_load(local.read_text())["model"]["default"] == "LOCAL"  # untouched
    gets = _gets(plane)
    assert gets and all(r["auth"] == f"Bearer {TOKEN}" and r["instance"] == INSTANCE for r in gets)
    assert gets[0]["profile"] == "default"


def test_poll_picks_up_changes_and_invalidates_caches(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    assert load_config()["display"]["personality"] == "concise"
    backend = remote_pkg.get_remote_backend()
    v1 = backend.version(plane.home)

    backend.poll_all()  # unchanged → 304
    assert plane.requests[-1]["if_none_match"] and backend.version(plane.home) == v1

    plane.upper = {"display": {"personality": "pirate"}}
    backend.poll_all()
    assert backend.version(plane.home) != v1
    assert load_config()["display"]["personality"] == "pirate"


def test_outage_while_running_keeps_in_memory_doc(plane):
    from hermes_cli.config import load_config
    plane.upper = {"display": {"personality": "concise"}}
    load_config()
    backend = remote_pkg.get_remote_backend()
    plane.fail_status = 503

    backend.poll_all()  # never raises, never exits

    assert load_config()["display"]["personality"] == "concise"
    assert "503" in backend.describe(plane.home)
    plane.fail_status = None
    backend.poll_all()
    assert "last poll failed" not in backend.describe(plane.home)


def test_poll_refusal_logs_error_but_keeps_running(plane, caplog):
    from hermes_cli.config import load_config
    load_config()
    plane.fail_status = 403
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["ERROR"]
    plane.fail_status = 503
    caplog.clear()
    with caplog.at_level("WARNING"):
        remote_pkg.get_remote_backend().poll_all()
    assert [r.levelname for r in caplog.records if "poll for profile" in r.getMessage()] == ["WARNING"]


def test_forked_child_rearms_its_poller(plane):
    import os

    from hermes_cli.config import load_config
    load_config()
    backend = remote_pkg.get_remote_backend()
    first = backend._poller
    backend._poller_pid = -1  # as in a child after fork: the parent's thread did not survive
    load_config()
    assert backend._poller_pid == os.getpid() and backend._poller is not first
    assert backend._poller is not None and backend._poller.is_alive()


def test_poll_interval_floor(monkeypatch):
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "1")
    assert backend_mod.poll_interval() == backend_mod.MIN_POLL_SECONDS
    monkeypatch.setenv("HERMES_CONFIG_REMOTE_POLL_SECONDS", "600")
    assert backend_mod.poll_interval() == 600


def test_named_profile_fetches_its_own_layer(plane):
    from hermes_cli.config import load_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    work = plane.home / "profiles" / "work"
    work.mkdir(parents=True)
    plane.profile("work")["values"] = {"display": {"personality": "formal"}}
    token = set_hermes_home_override(work)
    try:
        assert load_config()["display"]["personality"] == "formal"
    finally:
        reset_hermes_home_override(token)
    assert {r["profile"] for r in _gets(plane)} == {"work"}


def _hold_gets(monkeypatch):
    """Pause the FIRST GET after its response arrived and before the backend installs it, until
    ``release`` is set; every later request passes straight through."""
    import threading

    from plugins.config_backends.remote import client
    fetched, release = threading.Event(), threading.Event()
    original = client.request

    def held(method, *args, **kwargs):
        resp = original(method, *args, **kwargs)
        if method == "GET" and not fetched.is_set():
            fetched.set()
            assert release.wait(10)
        return resp

    monkeypatch.setattr(client, "request", held)
    return fetched, release


def test_poll_in_flight_does_not_overwrite_an_acknowledged_write(plane, monkeypatch):
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "upper-change"
    fetched, release = _hold_gets(monkeypatch)

    poll = threading.Thread(target=backend.poll_one, args=(st,))
    poll.start()
    assert fetched.wait(10)  # the poll's GET (profile v0) has its response, not yet installed
    write_config_key(plane.home / "config.yaml", "display.personality", "my-write")
    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    release.set()
    poll.join(10)
    assert not poll.is_alive()

    assert (st.profile_version, st.doc["display"]["personality"]) == (1, "my-write")
    assert plane.profile("default")["values"] == {"display": {"personality": "my-write"}}


def test_stale_fetch_with_same_profile_version_is_dropped(plane, monkeypatch):
    """Only an upper level changed, so both responses carry profileVersion 0: the guard must be
    'which fetch was installed last', not a profileVersion comparison."""
    import threading
    plane.upper = {"display": {"personality": "old"}}
    backend = get_config_backend()
    st = backend._state(plane.home)
    plane.upper["display"]["personality"] = "v1"
    fetched, release = _hold_gets(monkeypatch)

    slow = threading.Thread(target=backend.poll_one, args=(st,))
    slow.start()
    assert fetched.wait(10)  # holds an upper=v1 response
    plane.upper["display"]["personality"] = "v2"
    assert backend.poll_one(st) is True  # a newer fetch, started later, lands first
    assert st.doc["display"]["personality"] == "v2"
    release.set()
    slow.join(10)
    assert not slow.is_alive()

    assert st.profile_version == 0
    assert st.doc["display"]["personality"] == "v2"


def _multi_group_plane(plane):
    plane.upper = {"terminal": {"backend": "docker"}, "model": {"default": "hermes-4"}}
    plane.upper_locks = [{"path": "terminal.backend", "level": "group", "groupId": "g-sec"}]
    plane.groups = [{"groupId": "g-sec", "priority": 10, "version": 3},
                    {"groupId": "g-eng", "priority": 5, "version": 1}]
    plane.group_provenance = {"terminal": {"backend": "g-sec"}}


def test_multi_group_response_fields_are_ignored_and_group_lock_refuses_client_side(plane):
    """portal-groups §7.5: several group levels, locks[].groupId and groupProvenance are ignorable;
    a group lock is just an upper lock (lockedBy "group"), refused before anything is sent."""
    from hermes_cli.config import read_raw_config
    _multi_group_plane(plane)

    doc = read_raw_config()

    assert doc["terminal"]["backend"] == "docker" and doc["model"]["default"] == "hermes-4"
    assert get_config_backend().locked(plane.home, "terminal.backend") == "group"
    with pytest.raises(ConfigLockedError) as exc:
        write_config_key(plane.home / "config.yaml", "terminal.backend", "local")
    assert (exc.value.path, exc.value.locked_by) == ("terminal.backend", "group")
    assert plane.patches() == []
    write_config_key(plane.home / "config.yaml", "model.default", "hermes-5")  # unlocked: written
    (p,) = plane.patches()
    assert p["body"]["set"] == {"model.default": "hermes-5"}


def test_etag_is_opaque_and_echoed_verbatim(plane, monkeypatch):
    """The plane's ETag (hermes-config-etag/2 since portal-groups §6.8) is never parsed: whatever
    bytes it sends come back unchanged in If-None-Match."""
    from hermes_cli.config import read_raw_config
    _multi_group_plane(plane)
    real_effective = plane.effective
    opaque = 'W/"v2:any-bytes/at all"'

    def effective(name):
        body = real_effective(name)
        body["etag"] = opaque
        return body

    monkeypatch.setattr(plane, "effective", effective)
    read_raw_config()
    backend = get_config_backend()
    st = backend._state(plane.home)
    assert backend.poll_one(st) is False and not st.last_error  # 304: nothing changed
    assert _gets(plane)[-1]["if_none_match"] == opaque


def test_stub_plane_deep_merge_follows_contract_6_2():
    """The test double resolves like the plane: null over a mapping is ignored (contract §6.2)."""
    from .stub_plane import deep_merge
    upper = {"display": {"personality": "concise"}, "model": "a", "tags": [1, 2]}
    assert deep_merge(upper, {"display": None}) == upper
    assert deep_merge(upper, {"model": None})["model"] is None
    assert deep_merge(upper, {"tags": [3]})["tags"] == [3]


def test_profile_added_while_the_poller_iterates_does_not_kill_it(plane):
    """F3: a first read of another profile inserts its state while the poll loop walks the
    roster; the only poller thread must survive it."""
    import dataclasses
    import threading

    backend = get_config_backend()
    st = backend._state(plane.home)
    work = plane.home / "profiles" / "work"
    work.mkdir(parents=True)
    inserted = threading.Event()

    class AddsAProfileMidPass(backend_mod._ProfileState):
        reads = 0

        @property
        def next_poll(self):
            AddsAProfileMidPass.reads += 1
            if AddsAProfileMidPass.reads == 2 and not inserted.is_set():  # mid-pass: the deadline scan
                backend._state(work)
                inserted.set()
            return time.monotonic() + 1000

        @next_poll.setter
        def next_poll(self, value):
            pass

    import time
    key = backend._key(plane.home)
    backend._states[key] = AddsAProfileMidPass(**{
        f.name: getattr(st, f.name) for f in dataclasses.fields(st) if f.name != "next_poll"})
    loop = threading.Thread(target=backend._poll_loop, daemon=True)
    loop.start()
    assert inserted.wait(10)
    loop.join(2)
    assert loop.is_alive()
    assert backend._key(work) in backend._states
    backend._stop.set()


def test_a_dead_poller_is_rearmed_by_a_cached_read(plane):
    """F3: a poller that died must not leave cached profiles without updates."""
    import threading

    from hermes_cli.config import load_config
    load_config()
    backend = remote_pkg.get_remote_backend()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    backend._poller = dead
    load_config()
    assert backend._poller is not dead and backend._poller.is_alive()
