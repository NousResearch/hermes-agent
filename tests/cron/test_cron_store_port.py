"""The cron storage port as a plugin kind, and the local backend's pin.

``resolve_cron_storage(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is today's ``jobs.json``/``executions.db``; another name loads a provider through
real discovery (``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a
fallback to the local files. The local backend serves only the local tenant and its own
profile, and every job records the principal who created it.
"""
import json
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from cron_store_port import (
    CronPinRefused, CronStorageProvider, CronStore, resolve_cron_storage,
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
