"""The secret store conformance suite against the local backend (today's profile ``.env`` and
``auth.json``).

Agents are profile homes under ``tmp_path``. The local backend holds the agent level only, so
only the pinning part of the suite runs here; the owner-wide and tenant-wide resolution invariants
run against a backend that holds every level (``test_reference.py``). The suite's list of
invariants it does NOT cover applies here too (``hermes_conformance.secrets.suite``).
"""

import json
import os
from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from agent.secret_scope import load_env_file
from hermes_cli.auth import _load_auth_store
from hermes_conformance.secrets import Backend, SecretPinningConformance
from hermes_scope import LOCAL_OWNER, Group, Principal, Scope
from hermes_secret_port_local import LocalSecretStoreProvider

#: The key the managed-key hook pins, as an administrator would (the managed ``.env`` defines
#: it, so the write lock refuses changing it).
PINNED_KEY = "ZZ_ADMINISTRATOR_PINNED_KEY"


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`, and the profile homes it serves."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)
    provider = LocalSecretStoreProvider()

    def bind_member(agent):
        return bind_state(Scope("local", str(homes[agent]), Principal("member"),
                                owner=LOCAL_OWNER, can_administer=False))

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"),
                                owner=Group("g"), can_administer=False))

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    @contextmanager
    def managed_key():
        """While active, :data:`PINNED_KEY` is defined by the managed ``.env`` (the
        administrator's scope the write lock already honors), so writing it is refused with a
        plain ``ValueError`` — not the store's pin refusal."""
        managed = tmp_path / "managed"
        managed.mkdir(exist_ok=True)
        (managed / ".env").write_text(f"{PINNED_KEY}=set-by-the-administrator\n", encoding="utf-8")
        previous = os.environ.get("HERMES_MANAGED_DIR")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        try:
            yield PINNED_KEY
        finally:
            if previous is None:
                monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
            else:
                monkeypatch.setenv("HERMES_MANAGED_DIR", previous)

    @contextmanager
    def fail_writes():
        """While active, the seeded agent's ``.env`` and ``auth.json`` are unreadable
        directories, so today's writers fail part-way through the write — after the caller's
        value (or record) is bound on their frames — with a plain ``OSError``."""
        broken = []
        for filename in (".env", "auth.json"):
            path = homes["a"] / filename
            if path.is_dir():
                path.rmdir()
            elif path.exists():
                path.unlink()
            path.mkdir()
            broken.append(path)
        try:
            yield
        finally:
            for path in broken:
                path.rmdir()

    backend = Backend(open=lambda agent: provider.open(homes[agent]),
                      bind=lambda agent: bind_profile_scope(homes[agent]),
                      bind_member=bind_member, foreign_tenant=foreign_tenant, unbound=unbound,
                      managed_key=managed_key, fail_writes=fail_writes,
                      levels=provider.levels)
    return backend, homes, provider


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    return backend


class TestLocalSecrets(SecretPinningConformance):
    pass


# --- today's semantics, unchanged ---------------------------------------------------------------


def test_a_set_succeeds_through_todays_env_file_and_reads_back(tmp_path, monkeypatch):
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    with backend.bind("a"):
        store.set("ANTHROPIC_API_KEY", "sk-file-value", level="agent")
        assert store.get("ANTHROPIC_API_KEY") == "sk-file-value"
        assert "ANTHROPIC_API_KEY" in store.names()
    # Today's writer and today's tokenizer see the same file the port wrote.
    assert load_env_file(homes["a"] / ".env")["ANTHROPIC_API_KEY"] == "sk-file-value"
    with backend.bind("a"):
        assert store.delete("ANTHROPIC_API_KEY", level="agent") is True
        assert store.get("ANTHROPIC_API_KEY") is None
    assert "ANTHROPIC_API_KEY" not in load_env_file(homes["a"] / ".env")


def test_the_launch_profiles_env_only_credential_succeeds_for_the_process_home_only(tmp_path, monkeypatch):
    """A credential the launcher injected into the environment (no file to rebuild it from)
    resolves for the process's own home, exactly as the bound turn's secret scope does, and for
    no other profile."""
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))  # a is this process's own home
    monkeypatch.setenv("ZZ_LAUNCHER_INJECTED_KEY", "injected-by-the-launcher")
    with backend.bind("a"):
        assert backend.open("a").get("ZZ_LAUNCHER_INJECTED_KEY") == "injected-by-the-launcher"
    with backend.bind("b"):
        assert backend.open("b").get("ZZ_LAUNCHER_INJECTED_KEY") is None


def test_an_auth_record_roundtrips_succeeds_through_todays_auth_json(tmp_path, monkeypatch):
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    record = {"api_key": "sk-auth-value", "type": "api_key"}
    with backend.bind("a"):
        store.set_auth("openai", record)
        assert store.get_auth("openai") == record
    # Today's auth store file, written the way every local writer writes it.
    raw = _load_auth_store(homes["a"] / "auth.json")
    assert raw["providers"]["openai"] == record
    assert not raw.get("active_provider")  # a record write is not a provider choice
    with open(homes["a"] / "auth.json", encoding="utf-8") as handle:
        assert json.load(handle)["providers"]["openai"]["api_key"] == "sk-auth-value"
