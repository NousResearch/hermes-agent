"""Bounded catalog dispatch through real scope/config readers and worker threads."""

import contextvars
import hashlib
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home, hermes_home_key
from hermes_cli.config import atomic_config_write, load_config, load_config_readonly
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.skill_catalog.identity import Descriptor, source_id
from hermes_cli.skill_catalog.provider import SkillCatalogProvider
from hermes_cli.skill_catalog.registry import Entry
from hermes_cli.skill_catalog.validation import CatalogError


class Catalog(SkillCatalogProvider):
    name = "team"
    label = "Team"
    implementation_revision = "1"


def make_entry(home, provider=None, key="fixture-catalog"):
    home.mkdir(parents=True, exist_ok=True)
    with _plugin_home_scope(home):
        config = load_config()
    section = config.setdefault("plugins", {})
    section["enabled"] = list(set(section.get("enabled") or []) | {key})
    section.setdefault("entries", {})[key] = {
        "enabled": True, "granted_capabilities": ["skills.catalog"],
        "settings": {"marker": home.name}}
    atomic_config_write(home / "config.yaml", config)
    return Entry(source_id(key, "team"), key, Descriptor(1, "team", "Team", "1"),
                 provider or Catalog(), hermes_home_key(home),
                 PluginManifest(name=key, version="1", capabilities=["skills.catalog"]), ready=True)


def test_call_binds_entry_scope_inside_worker_and_preserves_caller_context(tmp_path):
    from hermes_cli.skill_catalog import worker

    seen = []
    sentinel = contextvars.ContextVar("catalog_test_sentinel")
    sentinel.set("copied-context")
    caller_thread = threading.get_ident()

    class Reader(Catalog):
        def availability(self, request):
            settings = load_config_readonly()["plugins"]["entries"]["fixture-catalog"]["settings"]
            seen.append((get_hermes_home(), settings["marker"], threading.get_ident(),
                         sentinel.get(), request))
            return {"ok": True, "data": {"state": "ready", "message": "", "ignored": "drop"}}

    entry = make_entry(tmp_path / "selected", Reader())
    outer = tmp_path / "caller"
    with _plugin_home_scope(outer):
        assert worker.call(entry, "availability", {}) == {"state": "ready", "message": ""}
        assert get_hermes_home() == outer
    home, marker, thread_id, copied, request = seen[0]
    assert home == Path(entry.scope) and marker == "selected"
    assert thread_id != caller_thread and copied == "copied-context"
    context = request["context"]
    assert set(context) == {"profile_key", "plugin_key", "generation", "request_id", "deadline_ms"}
    assert context["profile_key"] == hashlib.sha256(entry.scope.encode("utf-8")).hexdigest()
    assert context["plugin_key"] == entry.plugin_key and context["generation"] == entry.generation
    assert re.fullmatch("[0-9a-f]{32}", context["request_id"])
    assert type(context["deadline_ms"]) is int
    assert int(time.time() * 1000) - 3000 <= context["deadline_ms"] <= int(time.time() * 1000) + 3000


def test_invalid_request_never_resolves_provider_method(tmp_path):
    from hermes_cli.skill_catalog import worker

    class Tripwire:
        def __getattribute__(self, name):
            raise AssertionError("provider attribute evaluated")

    entry = make_entry(tmp_path, Tripwire())
    with pytest.raises(CatalogError) as failure:
        worker.call(entry, "availability", {"context": {"plugin_key": "forged"}})
    assert failure.value.code == "invalid_request"


def test_provider_exception_is_safe_and_response_is_validated(tmp_path):
    from hermes_cli.skill_catalog import worker

    class Broken(Catalog):
        def availability(self, request):
            raise RuntimeError("private credential and token")

    entry = make_entry(tmp_path, Broken())
    with pytest.raises(CatalogError) as failure:
        worker.call(entry, "availability", {})
    assert failure.value.code == "provider_error"
    assert "private" not in failure.value.message
    entry.provider = type("Malformed", (Catalog,), {
        "availability": lambda self, request: {"ok": True, "data": {"state": "ready"}}})()
    with pytest.raises(CatalogError) as failure:
        worker.call(entry, "availability", {})
    assert failure.value.code == "invalid_response"


class Pending:
    def __init__(self, entry, method, payload, **kwargs):
        from hermes_cli.skill_catalog import worker

        self.returned = threading.Event()
        self.value = None

        def run():
            try:
                self.value = worker.call(entry, method, payload, **kwargs)
            except BaseException as exc:
                self.value = exc
            finally:
                self.returned.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finish(self):
        self.thread.join(4)
        assert not self.thread.is_alive(), "catalog caller did not return within its bound"
        return self.value


def assert_error(code, function, *args, **kwargs):
    with pytest.raises(CatalogError) as failure:
        function(*args, **kwargs)
    assert failure.value.code == code
    return failure.value


def operation(entry):
    return {"ok": True, "data": {
        "operation_id": "operation_0123456789", "identifier": entry.source_id + ":item",
        "action": "install", "state": "completed", "outcome": "installed",
        "managed_state": "installed", "message": "", "installation_revision": "1"}}


CONFIRM = {"action": "install", "id": "item", "review_token": "review_0123456789",
           "acknowledgements": [], "admission": None}


@pytest.mark.parametrize("method,payload,budget", [
    ("availability", {}, 3), ("installed", {}, 3),
    ("reviewed", {"review_token": "review_0123456789"}, 3),
    ("operation", {"operation_id": "operation_0123456789"}, 3),
    ("search", {}, 5), ("inspect", {"id": "item"}, 15),
    ("check", {"ids": []}, 15), ("scan", {"id": "item"}, 30),
    ("review", {"action": "install", "id": "item"}, 60), ("confirm", CONFIRM, 60),
])
@pytest.mark.parametrize("remaining", [None, 1.25])
def test_method_budget_and_caller_deadline_are_core_stamped(tmp_path, monkeypatch, method, payload,
                                                          budget, remaining):
    from types import SimpleNamespace
    from hermes_cli.skill_catalog import worker

    requests = []

    def record(self, request):
        requests.append(request)
        return {"ok": False, "error": {"code": "unsupported", "message": "fixture"}}

    provider = type("Recorder", (Catalog,), {method: record})()
    entry = make_entry(tmp_path, provider)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 1000.0))
    deadline = None if remaining is None else 100.0 + remaining
    assert_error("unsupported", worker.call, entry, method, payload, deadline=deadline)
    assert requests[0]["context"]["deadline_ms"] == int((1000 + min(budget, remaining or budget)) * 1000)
    assert set(requests[0]) == set(payload) | {"context"} | {
        "search": {"query", "limit", "cursor"}, "installed": {"limit", "cursor"},
        "inspect": {"version"}, "scan": {"version"}, "review": {"version"},
        "check": {"cursor"}}.get(method, set())


@pytest.mark.parametrize("state", ["not_ready", "stopped", "revoked", "expired"])
def test_worker_rechecks_entry_after_blocked_scope_entry(tmp_path, monkeypatch, state):
    from contextlib import contextmanager
    from hermes_cli.skill_catalog import worker

    entered, release = threading.Event(), threading.Event()
    invoked = []
    threads = []
    spawn = worker.spawn_context_thread

    def track(*args, **kwargs):
        thread = spawn(*args, **kwargs)
        threads.append(thread)
        return thread

    class Recorder(Catalog):
        def availability(self, request):
            invoked.append(True)
            return super().availability(request)

    entry = make_entry(tmp_path, Recorder())

    @contextmanager
    def delayed_scope(home):
        with _plugin_home_scope(home):
            entered.set()
            assert release.wait(8)
            yield

    monkeypatch.setattr(worker, "_plugin_home_scope", delayed_scope)
    monkeypatch.setattr(worker, "spawn_context_thread", track)
    pending = Pending(entry, "availability", {}, deadline=time.monotonic() + (2 if state == "expired" else 6))
    try:
        assert entered.wait(4)
        if state == "not_ready":
            entry.ready = False
        elif state == "stopped":
            entry.stopped = True
        elif state == "revoked":
            with _plugin_home_scope(Path(entry.scope)):
                config = load_config()
            config["plugins"]["entries"][entry.plugin_key]["granted_capabilities"] = []
            atomic_config_write(Path(entry.scope) / "config.yaml", config)
        else:
            assert pending.returned.wait(4)
        release.set()
        value = pending.finish()
        for thread in threads:
            thread.join(4)
            assert not thread.is_alive()
        assert isinstance(value, CatalogError)
        assert value.code == {"revoked": "consent_required", "expired": "timeout"}.get(
            state, "provider_unavailable")
        assert invoked == []
    finally:
        release.set()
        pending.thread.join(4)
        for thread in threads:
            thread.join(4)


def test_expired_request_never_resolves_provider_attribute(tmp_path):
    from hermes_cli.skill_catalog import worker

    class Tripwire:
        def __getattribute__(self, name):
            raise AssertionError("expired dispatch")

    entry = make_entry(tmp_path, Tripwire())
    assert_error("timeout", worker.call, entry, "availability", {}, deadline=time.monotonic() - 1)


def test_timed_out_read_retains_slot_until_real_worker_cleanup_exits(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli.skill_catalog import worker

    cleanup, release = threading.Event(), threading.Event()
    threads = []
    spawn = worker.spawn_context_thread

    def track(*args, **kwargs):
        thread = spawn(*args, **kwargs)
        threads.append(thread)
        return thread

    @contextmanager
    def blocked_cleanup(home):
        with _plugin_home_scope(home):
            try:
                yield
            finally:
                cleanup.set()
                assert release.wait(8)

    monkeypatch.setattr(worker, "spawn_context_thread", track)
    monkeypatch.setattr(worker, "_plugin_home_scope", blocked_cleanup)
    entry = make_entry(tmp_path)
    pending = Pending(entry, "availability", {}, deadline=time.monotonic() + 2)
    try:
        assert cleanup.wait(4)
        value = pending.finish()
        assert isinstance(value, CatalogError) and value.code == "timeout"
        assert_error("busy", worker.call, entry, "search", {})
        assert len(threads) == 1, "busy must refuse without queuing or spawning"
    finally:
        release.set()
        pending.thread.join(4)
        for thread in threads:
            thread.join(4)
    assert worker.call(entry, "availability", {}) == {"state": "ready", "message": ""}


def test_confirm_slot_is_retained_but_operation_reads_are_independent(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import worker

    entered, release = threading.Event(), threading.Event()
    threads = []
    spawn = worker.spawn_context_thread

    def track(*args, **kwargs):
        thread = spawn(*args, **kwargs)
        threads.append(thread)
        return thread

    class Effect(Catalog):
        def confirm(self, request):
            if not entered.is_set():
                entered.set()
                assert release.wait(8)
            return operation(entry)

        def operation(self, request):
            return operation(entry)

    monkeypatch.setattr(worker, "spawn_context_thread", track)
    entry = make_entry(tmp_path, Effect())
    pending = Pending(entry, "confirm", CONFIRM, deadline=time.monotonic() + 2)
    try:
        assert entered.wait(4)
        assert worker.call(entry, "operation", {"operation_id": "operation_0123456789"})["state"] == "completed"
        value = pending.finish()
        assert isinstance(value, CatalogError) and value.code == "outcome_unknown"
        assert_error("busy", worker.call, entry, "confirm", CONFIRM)
        assert worker.call(entry, "operation", {"operation_id": "operation_0123456789"})["state"] == "completed"
    finally:
        release.set()
        pending.thread.join(4)
        for thread in threads:
            thread.join(4)
    assert worker.call(entry, "confirm", CONFIRM)["state"] == "completed"


def test_profile_capacity_is_four_across_entries_and_timeouts(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import worker

    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    threads = []
    spawn = worker.spawn_context_thread

    def track(*args, **kwargs):
        thread = spawn(*args, **kwargs)
        threads.append(thread)
        return thread

    def blocker(event):
        class Blocked(Catalog):
            def availability(self, request):
                event.set()
                assert release.wait(8)
                return super().availability(request)
        return Blocked()

    monkeypatch.setattr(worker, "spawn_context_thread", track)
    entries = [make_entry(tmp_path / "shared", blocker(event), key=f"catalog-{index}")
               for index, event in enumerate(entered)]
    spare = make_entry(tmp_path / "shared", key="spare")
    pending = []
    try:
        for entry, event in zip(entries, entered):
            pending.append(Pending(entry, "availability", {}, deadline=time.monotonic() + 2))
            assert event.wait(4)
        assert_error("busy", worker.call, spare, "availability", {})
        other = make_entry(tmp_path / "other")
        assert worker.call(other, "availability", {})["state"] == "ready"
        for item in pending:
            value = item.finish()
            assert isinstance(value, CatalogError) and value.code == "timeout"
        assert_error("busy", worker.call, spare, "availability", {})
        assert len(threads) == 5
    finally:
        release.set()
        for item in pending:
            item.thread.join(4)
        for thread in threads:
            thread.join(4)
    assert worker.call(spare, "availability", {})["state"] == "ready"


def test_shutdown_fences_old_generation_and_reports_active_effect_without_cancellation(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import worker

    entered, release = threading.Event(), threading.Event()
    threads = []
    spawn = worker.spawn_context_thread

    def track(*args, **kwargs):
        thread = spawn(*args, **kwargs)
        threads.append(thread)
        return thread

    class Effect(Catalog):
        def confirm(self, request):
            entered.set()
            assert release.wait(8)
            return operation(entry)

    monkeypatch.setattr(worker, "spawn_context_thread", track)
    entry = make_entry(tmp_path, Effect())
    pending = Pending(entry, "confirm", CONFIRM)
    try:
        assert entered.wait(4)
        report = worker.shutdown(entry.scope, timeout=0)
        assert len(report) == 1 and report[0]["code"] == "outcome_unknown"
        assert report[0]["registration_id"] == entry.registration_id
        assert threads[0].is_alive() and not release.is_set()
        assert_error("provider_unavailable", worker.call, entry, "availability", {})
        same_generation = replace(entry, registration_id="another_registration_0123456789")
        assert_error("provider_unavailable", worker.call, same_generation, "availability", {})
        renewed = replace(entry, generation="new_generation_0123456789", provider=Catalog())
        assert worker.call(renewed, "availability", {})["state"] == "ready"
        assert_error("provider_unavailable", worker.call, entry, "availability", {})
    finally:
        release.set()
        pending.thread.join(4)
        for thread in threads:
            thread.join(4)
    assert pending.finish()["state"] == "completed"
    assert worker.shutdown(entry.scope, timeout=0) == ()
    assert_error("provider_unavailable", worker.call, renewed, "availability", {})


def test_shutdown_cannot_be_reset_by_mutating_old_entry_generation(tmp_path):
    from hermes_cli.skill_catalog import worker

    entry = make_entry(tmp_path)
    assert worker.call(entry, "availability", {})["state"] == "ready"
    assert worker.shutdown(entry.scope, timeout=0) == ()
    entry.stopped = False
    entry.generation = "changed_old_generation_0123456789"
    assert_error("provider_unavailable", worker.call, entry, "availability", {})
    renewed = replace(entry, generation="explicit_new_generation_0123456789")
    assert worker.call(renewed, "availability", {})["state"] == "ready"


def test_same_registration_entry_copy_cannot_bypass_read_capacity(tmp_path):
    from hermes_cli.skill_catalog import worker

    entered, release = threading.Event(), threading.Event()

    class Blocked(Catalog):
        def availability(self, request):
            entered.set()
            assert release.wait(8)
            return super().availability(request)

    entry = make_entry(tmp_path, Blocked())
    pending = Pending(entry, "availability", {})
    try:
        assert entered.wait(4)
        assert_error("busy", worker.call, replace(entry, provider=Catalog()), "availability", {})
    finally:
        release.set()
        pending.thread.join(4)


def test_provider_attribute_and_authorization_run_only_in_owned_worker(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import worker

    caller_thread = threading.get_ident()
    seen = []
    entry = make_entry(tmp_path)
    original_authorized = entry.authorized

    def authorized():
        seen.append(("authorization", threading.get_ident(), get_hermes_home()))
        return original_authorized()

    class Dynamic:
        def __getattribute__(self, name):
            seen.append(("method", threading.get_ident(), get_hermes_home()))
            assert name == "availability"
            return lambda request: {"ok": True, "data": {"state": "ready", "message": ""}}

    entry.provider = Dynamic()
    monkeypatch.setattr(entry, "authorized", authorized)
    assert worker.call(entry, "availability", {})["state"] == "ready"
    assert [label for label, _, _ in seen] == ["authorization", "method"]
    assert all(thread_id != caller_thread and home == Path(entry.scope) for _, thread_id, home in seen)


def test_shutdown_fences_registered_entry_before_first_call(tmp_path):
    from types import SimpleNamespace
    from hermes_cli.skill_catalog import registry, worker

    entry = make_entry(tmp_path)
    context = SimpleNamespace(manifest=entry.manifest)
    registry.register_catalog(context, entry.provider, scope=entry.scope)
    registry.publish(entry.source_id, scope=entry.scope)
    try:
        registered, = registry.snapshot(scope=entry.scope)
        assert worker.shutdown(entry.scope, timeout=0) == ()
        assert_error("provider_unavailable", worker.call, registered, "availability", {})
    finally:
        registry.restore_registration(entry.source_id, entry.provider, None, scope=entry.scope)


def test_provider_raised_catalog_error_is_not_a_trusted_wire_error(tmp_path):
    from hermes_cli.skill_catalog import worker

    class Broken(Catalog):
        def availability(self, request):
            raise CatalogError("sign_in_required", "private exception detail")

    entry = make_entry(tmp_path, Broken())
    failure = assert_error("provider_error", worker.call, entry, "availability", {})
    assert "private" not in failure.message


def test_shutdown_drains_worker_scope_cleanup_before_returning(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli.skill_catalog import worker

    cleanup, release, draining, finished = (threading.Event() for _ in range(4))
    entry = make_entry(tmp_path)

    @contextmanager
    def blocked_cleanup(home):
        with _plugin_home_scope(home):
            try:
                yield
            finally:
                cleanup.set()
                assert release.wait(8)

    monkeypatch.setattr(worker, "_plugin_home_scope", blocked_cleanup)
    pending = Pending(entry, "availability", {})
    assert cleanup.wait(4)
    real_join = threading.Thread.join

    def observed_join(thread, timeout=None):
        if threading.current_thread().name == "catalog-test-drain":
            draining.set()
        return real_join(thread, timeout)

    def drain():
        try:
            assert worker.shutdown(entry.scope, timeout=5) == ()
        finally:
            finished.set()

    monkeypatch.setattr(threading.Thread, "join", observed_join)
    shutdown_thread = threading.Thread(target=drain, name="catalog-test-drain", daemon=True)
    shutdown_thread.start()
    try:
        assert draining.wait(4)
        assert not finished.is_set()
        assert_error("provider_unavailable", worker.call, entry, "availability", {})
    finally:
        release.set()
        pending.thread.join(4)
        shutdown_thread.join(4)
    assert finished.is_set() and not shutdown_thread.is_alive()


def test_worker_start_failure_releases_unused_capacity(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import worker

    entry = make_entry(tmp_path)
    spawn = worker.spawn_context_thread

    def fail_start(*args, **kwargs):
        raise RuntimeError("private spawn detail")

    monkeypatch.setattr(worker, "spawn_context_thread", fail_start)
    error = assert_error("provider_error", worker.call, entry, "availability", {})
    assert "private" not in error.message
    monkeypatch.setattr(worker, "spawn_context_thread", spawn)
    assert worker.call(entry, "availability", {})["state"] == "ready"


def test_parallel_profile_readers_remain_scoped_while_other_worker_is_blocked(tmp_path):
    from hermes_cli.skill_catalog import worker

    entered, release = threading.Event(), threading.Event()
    seen = []

    class Reader(Catalog):
        def availability(self, request):
            home = get_hermes_home()
            if home.name == "first":
                entered.set()
                assert release.wait(8)
            marker = load_config_readonly()["plugins"]["entries"]["fixture-catalog"]["settings"]["marker"]
            seen.append((home.name, marker, request["context"]["profile_key"]))
            return super().availability(request)

    first = make_entry(tmp_path / "first", Reader())
    second = make_entry(tmp_path / "second", Reader())
    pending = Pending(first, "availability", {})
    try:
        assert entered.wait(4)
        assert worker.call(second, "availability", {})["state"] == "ready"
        assert not pending.returned.is_set()
    finally:
        release.set()
        pending.thread.join(4)
    assert pending.finish()["state"] == "ready"
    assert {(home, marker) for home, marker, _ in seen} == {("first", "first"), ("second", "second")}
    assert len({profile_key for _, _, profile_key in seen}) == 2
