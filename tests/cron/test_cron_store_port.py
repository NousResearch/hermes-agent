"""The cron storage port as a plugin kind, and the local backend's pin and administration gate.

``resolve_cron_storage(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is today's ``jobs.json``/``executions.db``; another name loads a provider through
real discovery (``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a
fallback to the local files. The local backend serves only the local tenant and its own
profile, every job mutation requires the bound scope's administration right, and every job
records the principal who created it — by the same write that persists it.
"""
import contextlib
import json
import textwrap
from datetime import datetime, timedelta

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from cron_store_port import (
    CronAdminRefused, CronPinRefused, CronStorageProvider, CronStore, resolve_cron_storage,
)
from cron_store_port_local import LocalCronStorageProvider
from hermes_scope import LOCAL_OWNER, LOCAL_TENANT, Group, Principal, Scope

_PLUGIN = '''
from cron_store_port import CronStorageProvider


class Probe(CronStorageProvider):
    @property
    def name(self):
        return "probe"

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_cron_storage(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_cron_storage(name)
        assert isinstance(provider, LocalCronStorageProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(CronStorageProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, home=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_cron_storage("probe")
    assert isinstance(provider, CronStorageProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):", "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_cron_storage("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="postgres"):
        resolve_cron_storage("postgres")


@pytest.fixture
def store(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    with bind_profile_scope(home):
        yield LocalCronStorageProvider().open(home), home


def test_the_local_backend_is_todays_store(store):
    """The port is today's files at today's paths, and today's callers see the same rows."""
    backend, home = store
    assert isinstance(backend, CronStore)
    job = backend.create_job("through the port", "2h", name="port job")
    assert backend.create_execution(job["id"], source="test")["status"] == "claimed"

    jobs_file = home / "cron" / "jobs.json"
    assert jobs_file.is_file()
    with open(jobs_file, encoding="utf-8-sig") as handle:
        assert json.load(handle)["jobs"][0]["id"] == job["id"]
    assert (home / "cron" / "executions.db").is_file()
    from cron import executions as audit
    from cron import jobs as cron_jobs
    with bind_profile_scope(home):  # today's callers, unported, read the same store
        assert [j["id"] for j in cron_jobs.list_jobs(include_disabled=True)] == [job["id"]]
        assert audit.latest_execution(job["id"])["status"] == "claimed"


def test_a_job_records_the_creating_principal_from_the_bound_scope(store):
    backend, home = store
    job = backend.create_job("the owner's job", "2h")
    assert job["created_by"] == LOCAL_OWNER.id  # a profile-bound local activity acts for the owner
    with bind_state(Scope(LOCAL_TENANT, str(home), Principal("jane"))):
        shared = backend.create_job("jane's job", "2h")
        assert shared["created_by"] == "jane"
    assert backend.get_job(job["id"])["created_by"] == LOCAL_OWNER.id
    assert backend.get_job(shared["id"])["created_by"] == "jane"


def test_an_update_cannot_repoint_the_creating_principal(store):
    backend, _home = store
    job = backend.create_job("the owner's job", "2h")
    with pytest.raises(ValueError, match="created_by"):
        backend.update_job(job["id"], {"created_by": "someone-else"})
    assert backend.get_job(job["id"])["created_by"] == LOCAL_OWNER.id
    with pytest.raises(ValueError, match="id"):
        backend.update_job(job["id"], {"id": "another-id"})


@pytest.mark.parametrize("pin", [
    # A hosted tenant naming this very path as its agent is still not this local store's.
    lambda home: Scope("acme", str(home), Principal("u"), owner=Group("support"), can_administer=False),
    lambda home: Scope(LOCAL_TENANT, str(home.parent / "elsewhere"), Principal("owner")),
], ids=["another_tenant", "another_profile"])
def test_the_local_backend_refuses_any_pin_but_its_own_profile(store, pin):
    backend, home = store
    backend.create_job("the owner's job", "2h")
    with bind_state(pin(home)):
        with pytest.raises(CronPinRefused):
            backend.get_job("nonsense")
        with pytest.raises(CronPinRefused):
            backend.list_jobs()
        with pytest.raises(CronPinRefused):
            backend.get_due_jobs()
        with pytest.raises(CronPinRefused):
            backend.list_executions()
        with pytest.raises(CronPinRefused):
            backend.create_job("a foreign job", "2h")
        with pytest.raises(CronPinRefused):
            backend.claim_job_for_fire("nonsense")
        with pytest.raises(CronPinRefused):
            backend.create_execution("nonsense", source="test")


def test_with_no_scope_bound_the_local_backend_serves_the_ambient_profile_only(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    backend = LocalCronStorageProvider().open(home)
    monkeypatch.setenv("HERMES_HOME", str(other))
    with pytest.raises(CronPinRefused):
        backend.list_jobs()  # the ambient profile is not this store's
    with bind_profile_scope(home):
        assert backend.list_jobs() == []  # a bound profile names its own store


def test_a_store_whose_home_is_not_an_absolute_profile_path_is_refused():
    with pytest.raises(CronPinRefused):
        LocalCronStorageProvider().open("a/relative/home")


def _non_administering(home):
    """The same profile's scope with the administration right withheld: a principal who may use
    the agent but never administer it."""
    return bind_state(Scope(LOCAL_TENANT, str(home), Principal("jane"), can_administer=False))


def test_every_job_mutation_requires_the_administration_right(store):
    backend, home = store
    job = backend.create_job("the owner's job", "2h", name="admin job")
    before = backend.get_job(job["id"])
    with _non_administering(home):
        for mutation in (
            lambda: backend.create_job("a job without the right", "2h"),
            lambda: backend.update_job(job["id"], {"prompt": "changed"}),
            lambda: backend.pause_job(job["id"]),
            lambda: backend.resume_job(job["id"]),
            lambda: backend.trigger_job(job["id"]),
            lambda: backend.rearm_oneshot(job["id"], "2099-01-01T09:00:00"),
            lambda: backend.remove_job(job["id"]),
            lambda: backend.cancel_job_run(job["id"], reason="test"),
            # The check precedes every lookup: a missing job is refused the same way, not
            # read as nothing-to-do.
            lambda: backend.update_job("absent", {"prompt": "x"}),
            lambda: backend.pause_job("absent"),
            lambda: backend.resume_job("absent"),
            lambda: backend.trigger_job("absent"),
            lambda: backend.rearm_oneshot("absent", "2099-01-01T09:00:00"),
            lambda: backend.remove_job("absent"),
            lambda: backend.cancel_job_run("absent", reason="test"),
        ):
            with pytest.raises(CronAdminRefused):
                mutation()
    # A refusal changed nothing.
    assert backend.get_job(job["id"]) == before


def test_reads_and_the_schedulers_own_path_need_no_administration_right(store):
    backend, home = store
    job = backend.create_job("the owner's job", "2h", name="due job")
    backend.update_job(job["id"], {"next_run_at": (datetime.now().astimezone()
                                                    - timedelta(minutes=5)).isoformat()})
    with _non_administering(home):
        assert backend.get_job(job["id"])["id"] == job["id"]
        assert [row["id"] for row in backend.list_jobs()] == [job["id"]]
        assert [row["id"] for row in backend.get_due_jobs()] == [job["id"]]
        execution = backend.create_execution(job["id"], source="test")
        assert backend.claim_dispatch(job["id"]) is True
        assert backend.claim_job_for_fire(job["id"]) is True
        assert backend.heartbeat_fire_claim(job["id"], expected_owner="whoever") is False
        assert backend.mark_job_run(job["id"], True) is True
        assert backend.mark_execution_running(execution["id"]) is not None
        with backend.fire_claim_fence(job["id"], expected_owner="whoever") as holds:
            assert holds is False
        assert backend.list_executions()[0]["id"] == execution["id"]


def test_a_caller_with_no_scope_bound_administers_as_today(tmp_path, monkeypatch):
    """Today's local caller has no scope bound and administers: its mutations land as always."""
    home = tmp_path / "profile"
    home.mkdir()
    backend = LocalCronStorageProvider().open(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    job = backend.create_job("the ambient caller's job", "2h")
    assert backend.pause_job(job["id"]) is not None
    assert backend.remove_job(job["id"]) is True


def test_the_creating_principal_is_never_an_argument(store):
    backend, _home = store
    with pytest.raises(TypeError):
        backend.create_job("a forged job", "2h", created_by="someone-else")


def test_a_created_job_is_stamped_by_the_same_write_that_persists_it(store, monkeypatch):
    """An interruption at the write leaves either no job or a fully stamped one: stamping the
    creator is never a second, separately interruptible durable write."""
    backend, home = store
    from cron import jobs as cron_jobs

    saves = []
    real_save = cron_jobs.save_jobs

    def interrupting_save(jobs, **kwargs):
        saves.append([job.get("id") for job in jobs])
        if len(saves) >= 2:  # a second durable write: the interruption strikes between them
            raise OSError("interrupted between the writes")
        return real_save(jobs, **kwargs)

    monkeypatch.setattr(cron_jobs, "save_jobs", interrupting_save)
    with contextlib.suppress(OSError):
        backend.create_job("the interrupted job", "2h", name="interrupted")
    jobs_file = home / "cron" / "jobs.json"
    with open(jobs_file, encoding="utf-8-sig") as handle:
        persisted = json.load(handle)["jobs"]
    for record in persisted:
        assert record.get("created_by"), \
            f"a persisted job is missing its creating principal: {record['id']}"
