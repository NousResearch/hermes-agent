"""Platform backends: routing, fallback, and the exact vendor-CLI argv we build.

The Bitwarden/1Password backends shell out to `bws` / `op`, neither of which
exists on a CI host — so every test here records the argv instead of running it,
and platform selection is exercised through injected fake backends.
"""

import json
import subprocess
from pathlib import Path

import pytest

import agent.credential_backends as cb
from agent.credential_backends import (
    AmbiguousBackendError,
    BackendError,
    BackendStatus,
    CredentialBackend,
    save_credential,
    remove_credential,
)
from agent.credential_store import CredentialStore


class FakeBackend(CredentialBackend):
    """In-memory platform: records create/update/delete, optionally failing."""

    def __init__(self, name="fakeone", display_name="Fake One", *, available=True,
                 reason="test backend"):
        self.name = name
        self.display_name = display_name
        self.available = available
        self.reason = reason
        self.created = []
        self.updated = []
        self.deleted = []
        self.fail_create: str | None = None
        self.fail_delete: str | None = None
        self.values = {}

    def status(self) -> BackendStatus:
        return BackendStatus(self.name, self.display_name, self.available, self.reason)

    def create(self, name, value, *, note=""):
        if self.fail_create:
            raise BackendError(self.fail_create)
        self.created.append((name, value))
        ref = f"{self.name}:{name}"
        self.values[ref] = value
        return ref

    def read(self, ref):
        return self.values[ref]

    def update(self, ref, value):
        self.updated.append((ref, value))
        self.values[ref] = value

    def delete(self, ref):
        if self.fail_delete:
            raise BackendError(self.fail_delete)
        self.deleted.append(ref)


@pytest.fixture()
def store(tmp_path):
    return CredentialStore(tmp_path / "credentials")


def _install(monkeypatch, *backends):
    monkeypatch.setattr(cb, "all_backends", lambda: list(backends))
    return list(backends)


def _recorder(monkeypatch, payload):
    """Replace run_secret_cli with a recorder returning *payload* as JSON."""
    calls = []

    def fake_run_cli(argv, *, allow_env=(), extra_env=None, timeout=60.0):
        calls.append({"argv": list(argv), "allow_env": tuple(allow_env),
                      "extra_env": dict(extra_env or {})})
        return subprocess.CompletedProcess(list(argv), 0, json.dumps(payload), "")

    monkeypatch.setattr("agent.secret_sources.base.run_secret_cli", fake_run_cli)
    return calls


# --- selection & fallback ---------------------------------------------------

def test_auto_saves_locally_when_no_platform_is_available(monkeypatch, store):
    _install(monkeypatch)
    result = save_credential("alpha_token_one", "value1", store=store)
    assert result["success"] and result["backend"] == "local"
    assert result["warnings"] == []
    assert store.get("alpha_token_one") == "value1"


def test_auto_uses_the_single_available_platform(monkeypatch, store):
    fake = FakeBackend()
    _install(monkeypatch, fake)
    result = save_credential("alpha_token_one", "value1", store=store)
    assert result["backend"] == "fakeone"
    assert fake.created == [("alpha_token_one", "value1")]

    entry = store.entry("alpha_token_one")
    assert entry["backend"] == "fakeone" and entry["ref"] == "fakeone:alpha_token_one"
    assert "value" not in entry                      # platform-only: no local copy
    assert store.get("alpha_token_one") == "value1"  # read routes through the backend
    assert [e["backend"] for e in store.list()] == ["fakeone"]


def test_auto_with_two_platforms_is_ambiguous(monkeypatch, store):
    _install(monkeypatch, FakeBackend(name="one"), FakeBackend(name="two"))
    with pytest.raises(AmbiguousBackendError) as excinfo:
        save_credential("alpha_token_one", "value1", store=store)
    assert excinfo.value.names == ["one", "two"]


def test_unavailable_platform_can_be_named_but_not_used(monkeypatch, store):
    down = FakeBackend(name="down", available=False, reason="not installed")
    _install(monkeypatch, down)
    with pytest.raises(BackendError, match="not installed"):
        save_credential("alpha_token_one", "value1", backend="down", store=store)
    assert store.entry("alpha_token_one") is None


def test_auto_platform_failure_falls_back_to_local_with_a_warning(monkeypatch, store):
    flaky = FakeBackend()
    flaky.fail_create = "bws exploded"
    _install(monkeypatch, flaky)
    result = save_credential("alpha_token_one", "value1", store=store)
    assert result["success"] and result["backend"] == "local"
    assert store.get("alpha_token_one") == "value1"
    assert "bws exploded" in result["warnings"][0]


def test_explicit_platform_failure_never_falls_back(monkeypatch, store):
    flaky = FakeBackend()
    flaky.fail_create = "op exploded"
    _install(monkeypatch, flaky)
    with pytest.raises(BackendError, match="op exploded"):
        save_credential("alpha_token_one", "value1", backend="fakeone", store=store)
    assert store.entry("alpha_token_one") is None


def test_explicit_local_skips_an_available_platform(monkeypatch, store):
    fake = FakeBackend()
    _install(monkeypatch, fake)
    result = save_credential("alpha_token_one", "value1", backend="local", store=store)
    assert result["backend"] == "local"
    assert fake.created == []


# --- overwrite / move / remove ---------------------------------------------

def test_overwrite_updates_the_same_platform_secret(monkeypatch, store):
    fake = FakeBackend()
    _install(monkeypatch, fake)
    save_credential("alpha_token_one", "first", store=store)

    refused = save_credential("alpha_token_one", "second", store=store)
    assert refused["success"] is False and "--force" in refused["error"]

    result = save_credential("alpha_token_one", "second", overwrite=True, store=store)
    assert result["success"] and result["backend"] == "fakeone"
    assert fake.updated == [("fakeone:alpha_token_one", "second")]
    assert len(fake.created) == 1                    # updated in place, not recreated


def test_moving_to_another_platform_deletes_the_old_copy_last(monkeypatch, store):
    one = FakeBackend(name="one")
    two = FakeBackend(name="two")
    _install(monkeypatch, one, two)
    save_credential("alpha_token_one", "value1", backend="one", store=store)
    result = save_credential("alpha_token_one", "value1", backend="two",
                             overwrite=True, store=store)
    assert result["backend"] == "two"
    assert one.deleted == ["one:alpha_token_one"]    # only after the new copy exists
    assert store.entry("alpha_token_one")["backend"] == "two"


def test_moving_from_local_drops_the_local_value(monkeypatch, store):
    fake = FakeBackend()
    _install(monkeypatch, fake)
    store.set("alpha_token_one", "value1")
    save_credential("alpha_token_one", "value1", backend="fakeone",
                    overwrite=True, store=store)
    entry = store.entry("alpha_token_one")
    assert entry["backend"] == "fakeone" and "value" not in entry


def test_remove_deletes_from_the_platform_then_the_index(monkeypatch, store):
    fake = FakeBackend()
    _install(monkeypatch, fake)
    save_credential("alpha_token_one", "value1", store=store)

    result = remove_credential("alpha_token_one", store=store)
    assert result["success"] and result["backend"] == "fakeone"
    assert fake.deleted == ["fakeone:alpha_token_one"]
    assert store.entry("alpha_token_one") is None


def test_remove_keeps_the_index_when_the_platform_delete_fails(monkeypatch, store):
    fake = FakeBackend()
    fake.fail_delete = "network down"
    _install(monkeypatch, fake)
    save_credential("alpha_token_one", "value1", store=store)

    result = remove_credential("alpha_token_one", store=store)
    assert result["success"] is False and "network down" in result["error"]
    assert store.entry("alpha_token_one") is not None  # so the user can retry


def test_status_listing_reports_availability_and_reason(monkeypatch, store):
    _install(monkeypatch, FakeBackend(name="up", display_name="Up"),
             FakeBackend(name="down", display_name="Down", available=False,
                         reason="needs setup"))
    text = cb.format_statuses(store=store)
    assert "Up" in text and "available" in text
    assert "Down" in text and "unavailable" in text and "needs setup" in text
    assert str(store.base_dir) in text


# --- Bitwarden argv ---------------------------------------------------------

def _bitwarden(monkeypatch, **cfg):
    backend = cb.BitwardenBackend()
    monkeypatch.setattr(backend, "_cfg",
                        lambda: {"enabled": True, "project_id": "proj-1", **cfg})
    monkeypatch.setattr(backend, "_binary", lambda install=False: Path("/usr/bin/bws"))
    monkeypatch.setattr(cb, "_token_value", lambda env_name: "bws-token")
    return backend


def test_bitwarden_create_argv_and_locator(monkeypatch):
    calls = _recorder(monkeypatch, {"id": "secret-uuid"})
    backend = _bitwarden(monkeypatch)

    ref = backend.create("my_key", "sk-value", note="from hermes")

    assert ref == "secret-uuid"
    assert calls[0]["argv"] == ["/usr/bin/bws", "secret", "create", "my_key",
                                "--note", "from hermes", "--output", "json",
                                "sk-value", "proj-1"]
    assert calls[0]["extra_env"]["BWS_ACCESS_TOKEN"] == "bws-token"
    assert "BWS_ACCESS_TOKEN" not in str(calls[0]["allow_env"])  # env, not a pass-through


def test_bitwarden_create_protects_a_leading_dash_value(monkeypatch):
    calls = _recorder(monkeypatch, {"id": "secret-uuid"})
    backend = _bitwarden(monkeypatch)

    backend.create("my_key", "-not-a-flag")

    argv = calls[0]["argv"]
    assert argv[argv.index("--") + 1] == "-not-a-flag"
    assert argv[-1] == "proj-1"


def test_bitwarden_rejects_a_name_the_read_path_would_drop(monkeypatch):
    _recorder(monkeypatch, {"id": "x"})
    backend = _bitwarden(monkeypatch)
    with pytest.raises(BackendError, match="environment-variable names"):
        backend.create("hyphen-name", "value")


def test_bitwarden_read_update_delete_argv(monkeypatch):
    calls = _recorder(monkeypatch, {"id": "secret-uuid", "value": "sk-value"})
    backend = _bitwarden(monkeypatch)

    assert backend.read("secret-uuid") == "sk-value"
    backend.update("secret-uuid", "new-value")
    backend.delete("secret-uuid")

    assert [call["argv"][1:4] for call in calls] == [
        ["secret", "get", "secret-uuid"],
        ["secret", "edit", "secret-uuid"],
        ["secret", "delete", "secret-uuid"],
    ]
    assert calls[1]["argv"][calls[1]["argv"].index("--value") + 1] == "new-value"


def test_bitwarden_status_reasons(monkeypatch):
    backend = cb.BitwardenBackend()
    monkeypatch.setattr(backend, "_cfg", lambda: {"enabled": False})
    assert backend.status().available is False and "hermes secrets bitwarden setup" in backend.status().reason

    monkeypatch.setattr(backend, "_cfg", lambda: {"enabled": True, "project_id": ""})
    assert "project_id" in backend.status().reason

    monkeypatch.setattr(backend, "_cfg", lambda: {"enabled": True, "project_id": "p"})
    monkeypatch.setattr(cb, "_token_value", lambda env_name: "")
    assert "BWS_ACCESS_TOKEN" in backend.status().reason

    monkeypatch.setattr(cb, "_token_value", lambda env_name: "tok")
    monkeypatch.setattr(backend, "_binary", lambda install=False: None)
    assert "bws binary" in backend.status().reason


# --- 1Password argv ---------------------------------------------------------

def _onepassword(monkeypatch, **cfg):
    backend = cb.OnePasswordBackend()
    monkeypatch.setattr(backend, "_cfg",
                        lambda: {"enabled": True, "vault": "Hermes", **cfg})
    monkeypatch.setattr(backend, "_binary", lambda: Path("/usr/bin/op"))
    monkeypatch.setattr(cb, "_token_value", lambda env_name: "op-token")
    return backend


def test_onepassword_create_builds_the_reference(monkeypatch):
    calls = _recorder(monkeypatch, {"id": "item-id"})
    backend = _onepassword(monkeypatch, account="work")

    ref = backend.create("my_key", "sk-value")

    assert ref == "op://Hermes/item-id/password"
    argv = calls[0]["argv"]
    assert argv[0] == "/usr/bin/op" and argv[1:3] == ["--account", "work"]
    assert argv[3:6] == ["item", "create", "--category"]
    assert "--vault" in argv and argv[argv.index("--vault") + 1] == "Hermes"
    assert "password=sk-value" in argv          # value never touches our own argv twice
    assert calls[0]["extra_env"]["OP_SERVICE_ACCOUNT_TOKEN"] == "op-token"


def test_onepassword_update_and_delete_name_the_vault(monkeypatch):
    calls = _recorder(monkeypatch, {"id": "item-id"})
    backend = _onepassword(monkeypatch)

    backend.update("op://Hermes/item-id/password", "new-value")
    backend.delete("op://Hermes/item-id/password")

    edit = calls[0]["argv"]
    assert edit[1:3] == ["item", "edit"] and "item-id" in edit
    assert edit[edit.index("--vault") + 1] == "Hermes"
    assert "password=new-value" in edit
    delete = calls[1]["argv"]
    assert delete[1:3] == ["item", "delete"] and "item-id" in delete


def test_onepassword_vault_falls_back_to_an_existing_reference(monkeypatch):
    backend = cb.OnePasswordBackend()
    monkeypatch.setattr(backend, "_cfg",
                        lambda: {"enabled": True, "env": {"OPENAI_API_KEY":
                                                          "op://My Vault/Item/password"}})
    assert backend._vault() == "My Vault"
    monkeypatch.setattr(backend, "_cfg", lambda: {"enabled": True, "env": {}})
    assert backend.status().available is False and "vault" in backend.status().reason


def test_onepassword_read_uses_the_public_fetch_without_cache(monkeypatch):
    backend = _onepassword(monkeypatch)
    seen = {}

    def fake_fetch(**kwargs):
        seen.update(kwargs)
        return {"value": "sk-value"}, []

    monkeypatch.setattr("agent.secret_sources.onepassword.fetch_onepassword_secrets", fake_fetch)
    assert backend.read("op://Hermes/item-id/password") == "sk-value"
    assert seen["references"] == {"value": "op://Hermes/item-id/password"}
    assert seen["use_cache"] is False            # a just-updated value must not read stale