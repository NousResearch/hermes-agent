"""The assembly's doors around hosted mode: an explicitly configured provider name is never defaulted, the process-wide
shutdown is the unbound host's, the boot check runs whether or not a hosted scope is already bound, and a repository
kept from a handle stays bound to the hosted scope the handle served."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import persistence_assembly as pa
from persistence import ScopePinRefused, SessionId
from persistence.conversation.sessions import NewSession
from tests.persistence.hosted.test_provider_discovery import add_bundled, configured_home, plugin_source
from tests.persistence.hosted.test_turn_store_resolved_once import hosted


@pytest.mark.parametrize("configured", ["''", "'   '", "", "3", "[sqlite]", "{name: sqlite}", "true"])
def test_an_explicit_provider_that_names_nothing_is_refused(tmp_path, bundled_root, entry_points, configured):
    """``persistence.provider`` present but empty, blank or not a string raises ValueError: only absence means sqlite."""
    home = configured_home(tmp_path, configured)
    with pytest.raises(ValueError, match="names no provider"):
        pa._provider_for_home(home)


def test_a_persistence_section_that_is_not_a_mapping_is_refused(tmp_path, bundled_root, entry_points):
    home = tmp_path / "target"
    home.mkdir()
    (home / "config.yaml").write_text("persistence: sqlite\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no provider"):
        pa._provider_for_home(home)


@pytest.mark.parametrize("config_text", [None, "", "model: x\n", "persistence:\n", "persistence: {}\n",
                                         "persistence:\n  other: 1\n"])
def test_an_absent_provider_means_sqlite(tmp_path, bundled_root, entry_points, config_text):
    home = tmp_path / "target"
    home.mkdir()
    if config_text is not None:
        (home / "config.yaml").write_text(config_text, encoding="utf-8")
    assert pa._provider_for_home(home) == "sqlite"


def test_an_explicit_empty_provider_in_the_process_config_is_refused(monkeypatch):
    from hermes_cli import config
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"persistence": {"provider": ""}})
    with pytest.raises(ValueError, match="names no provider"):
        pa._provider_for_home(get_hermes_home())


def test_close_all_stores_refuses_while_a_hosted_scope_is_bound(provider):
    """A hosted activity cannot close every pin's backend and every local store; the unbound host still can."""
    with hosted("tenant-a"):
        handle = pa.open_persistence(pa.profile_scope())
        handle.sessions.create(NewSession(session_id=SessionId("a"), source="cli"))
    try:
        with hosted("tenant-b"), pytest.raises(ScopePinRefused):
            pa.close_all_stores()
        with hosted("tenant-a"), pytest.raises(ScopePinRefused):
            pa.close_all_stores()
        with hosted("tenant-a"):
            assert handle.sessions.get(SessionId("a")) is not None
        assert pa.close_all_stores() >= 0
        assert pa._pinned_backends == {}
    finally:
        handle.close()


@pytest.mark.parametrize("order", ["bound-first", "boot-first"])
def test_the_boot_check_runs_with_or_without_a_bound_hosted_scope(bundled_root, entry_points, order):
    from hermes_constants import get_hermes_home

    add_bundled(bundled_root, "isolating", plugin_source(pinning=True, isolating="pass"))
    (get_hermes_home() / "config.yaml").write_text("persistence:\n  provider: isolating\n", encoding="utf-8")
    if order == "boot-first":
        assert pa.require_hosted_persistence() is None
        with hosted("tenant-a"):
            assert pa.require_hosted_persistence() is None
    else:
        with hosted("tenant-a"):
            assert pa.require_hosted_persistence() is None
        assert pa.require_hosted_persistence() is None


def test_a_kept_repository_is_refused_under_another_bound_pin(provider):
    with hosted("tenant-a"):
        handle = pa.open_persistence(pa.profile_scope())
        sessions, aio = handle.sessions, handle.aio
        sessions.create(NewSession(session_id=SessionId("private-a"), source="cli"))
    try:
        with hosted("tenant-b"):
            with pytest.raises(ScopePinRefused):
                sessions.get(SessionId("private-a"))
            with pytest.raises(ScopePinRefused):
                asyncio.run(aio.sessions.get(SessionId("private-a")))
        with hosted("tenant-a"):
            assert sessions.get(SessionId("private-a")) is not None
    finally:
        handle.close()


def test_a_kept_local_repository_cannot_touch_local_state_while_hosted(tmp_path):
    """The real SQLite store: a repository kept from a local handle reads and writes nothing while a hosted scope is
    bound, and serves the local install again once it is not."""
    handle = pa.open_persistence(pa.profile_scope(home=tmp_path), eager=True)
    try:
        sessions, aio = handle.sessions, handle.aio
        sessions.create(NewSession(session_id=SessionId("local-secret"), source="cli"))
        with hosted("tenant-b"):
            with pytest.raises(ScopePinRefused):
                sessions.get(SessionId("local-secret"))
            with pytest.raises(ScopePinRefused):
                sessions.create(NewSession(session_id=SessionId("hosted-row"), source="cli"))
            with pytest.raises(ScopePinRefused):
                asyncio.run(aio.sessions.get(SessionId("local-secret")))
        assert sessions.get(SessionId("local-secret")) is not None
        assert sessions.get(SessionId("hosted-row")) is None
    finally:
        handle.close()


@pytest.mark.parametrize("door", ["store_location", "storage_health", "store_exists"])
def test_the_file_level_helpers_read_no_configuration(monkeypatch, door):
    """The file-level helpers answer for the current home (on disk) without loading the process config, as they always
    have: a config backend that cannot load does not reach them."""
    from hermes_cli import config
    from hermes_constants import get_hermes_home

    def unavailable(*_a, **_k):
        raise RuntimeError("the config backend is unavailable")

    monkeypatch.setattr(config, "load_config_readonly", unavailable)
    monkeypatch.setattr(config, "load_config", unavailable)
    scope = pa.profile_scope(home=get_hermes_home())
    assert scope.profile_dir.is_dir()
    result = getattr(pa, door)(scope)
    if door == "store_location":
        assert result == Path(scope.hermes_home_key) / "state.db"
    elif door == "store_exists":
        assert result in (True, False)
    else:
        assert result.probe() is not None


def test_the_file_level_helpers_answer_from_a_held_backend(tmp_path, bundled_root, entry_points):
    """While the process holds another provider's backend for a store, the file-level helpers answer for it: no file,
    no SQLite health. Once released they answer from the file as for any home, reading no configuration."""
    from persistence import CapabilityUnsupported

    add_bundled(bundled_root, "fakedir", plugin_source())
    scope = pa.profile_scope(home=configured_home(tmp_path, "fakedir"))
    handle = pa.open_persistence(scope, eager=True)
    try:
        assert pa.store_location(scope) is None
        with pytest.raises(CapabilityUnsupported):
            pa.storage_health(scope)
        with pytest.raises(CapabilityUnsupported):
            pa.store_exists(scope)
    finally:
        handle.close()
    assert pa.store_location(scope) == Path(scope.hermes_home_key) / "state.db"
    local = pa.profile_scope(home=configured_home(tmp_path, "sqlite", name="local"))
    held = pa.open_persistence(local, eager=True)
    try:
        assert pa.store_location(local) == Path(local.hermes_home_key) / "state.db"
        assert pa.store_exists(local) is True
    finally:
        held.close()
