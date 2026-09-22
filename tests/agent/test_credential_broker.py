from __future__ import annotations

import base64
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_broker import CredentialBroker
from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.bitwarden import BitwardenLoginBackend
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultItemMeta


def test_local_broker_creates_then_updates_same_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    broker = CredentialBroker([LocalLoginBackend()], {"write_backend": "local"})

    created = broker.save_login(label="Example", origin="https://example.com/login",
                                identifier_type="email", identifier="jane@example.com",
                                password="first secret", otp_secret="JBSWY3DPEHPK3PXP")
    updated = broker.save_login(label="Example", origin="https://example.com/account",
                                identifier_type="email", identifier="jane@example.com",
                                password="second secret")

    assert created.action == "created"
    assert updated.action == "updated"
    assert updated.meta.id == created.meta.id
    assert len(broker.write_backend().list_items()) == 1
    assert broker.write_backend().resolve_password(created.meta.id) == "second secret"
    assert broker.write_backend().resolve_secret(created.meta.id)["otp_secret"] == "JBSWY3DPEHPK3PXP"
    assert broker.remove_item(created.meta.id) is True
    assert broker.remove_item(created.meta.id) is False


def test_local_migration_is_dry_run_safe_idempotent_and_preserves_totp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    source = LocalLoginBackend()
    source.create_login(label="Existing", origin="https://existing.example", identifier_type="email",
                        identifier="same@example.com", password="existing local")
    source.create_login(label="Missing", origin="https://missing.example", identifier_type="username",
                        identifier="jane", password="migration secret", otp_secret="JBSWY3DPEHPK3PXP")

    class FakeTarget:
        name = "bitwarden"
        prefix = "bw:"
        display_name = "Bitwarden"
        needs_unlock = False

        def __init__(self):
            self.items = [VaultItemMeta("bw:existing", "login", "Existing", "https://existing.example",
                                        "2026-09-17", "email", "same@example.com")]
            self.secrets = {}

        def capabilities(self):
            return frozenset({"list", "resolve", "create_login", "remove"})

        def is_unlocked(self):
            return True

        def list_items(self):
            return list(self.items)

        def create_login(self, **kwargs):
            handle = f"bw:new-{len(self.items)}"
            meta = VaultItemMeta(handle, "login", kwargs["label"], kwargs["origin"], "2026-09-17",
                                 kwargs["identifier_type"], kwargs["identifier"])
            self.items.append(meta)
            self.secrets[handle] = {"password": kwargs["password"], "otp_secret": kwargs["otp_secret"]}
            return meta

        def resolve_password(self, handle):
            return self.secrets[handle]["password"]

        def remove_item(self, handle):
            self.items = [item for item in self.items if item.id != handle]
            self.secrets.pop(handle, None)
            return True

    target = FakeTarget()
    broker = CredentialBroker([source, target], {"write_backend": "bitwarden"})

    dry_run = broker.migrate_local_logins()
    assert [item.status for item in dry_run.items] == ["skipped_existing", "would_import"]
    assert len(target.items) == 1

    migrated = broker.migrate_local_logins(execute=True)
    assert migrated.imported == 1 and migrated.skipped == 1 and migrated.failed == 0
    assert target.secrets["bw:new-1"] == {
        "password": "migration secret",
        "otp_secret": "JBSWY3DPEHPK3PXP",
    }

    repeated = broker.migrate_local_logins(execute=True)
    assert repeated.imported == 0 and repeated.skipped == 2
    assert len(target.items) == 2
    assert "migration secret" not in repr(migrated)


def test_local_migration_scrubs_target_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    source = LocalLoginBackend()
    source.create_login(label="Failure", origin="https://failure.example", identifier_type="email",
                        identifier="failure@example.com", password="never echo this")

    class FailingTarget:
        name = "bitwarden"
        prefix = "bw:"
        display_name = "Bitwarden"
        needs_unlock = False

        def capabilities(self):
            return frozenset({"list", "resolve", "create_login"})

        def is_unlocked(self):
            return True

        def list_items(self):
            return []

        def create_login(self, **kwargs):
            raise RuntimeError(f"rejected {kwargs['password']}")

    result = CredentialBroker([source, FailingTarget()], {"write_backend": "bitwarden"}).migrate_local_logins(
        execute=True
    )

    assert result.failed == 1
    assert "never echo this" not in repr(result)
    assert result.items[0].error == "rejected [REDACTED]"


_FAKE_BW = r'''#!/usr/bin/env python3
import base64, json, os, sys
root = os.path.dirname(os.path.abspath(__file__))
state_path = os.path.join(root, "state.json")
log_path = os.path.join(root, "bw-write.log")
argv = sys.argv[1:]
stdin = sys.stdin.read()
secrets = ("first secret", "second secret")
leaked_env = [key for key, value in os.environ.items() if any(secret in value for secret in secrets)]
with open(log_path, "a", encoding="utf-8") as log:
    log.write(json.dumps({"argv": argv, "stdin": stdin, "leaked_env": leaked_env}) + "\n")
if os.environ.get("BW_SESSION") != "session-token":
    sys.stderr.write("Vault is locked.\n"); sys.exit(1)
if argv[:3] == ["get", "template", "item"]:
    print(json.dumps({"type": 1, "name": "", "notes": None, "favorite": False,
                      "fields": [], "login": None, "collectionIds": []})); sys.exit(0)
if argv[:3] == ["get", "template", "item.login"]:
    print(json.dumps({"username": None, "password": None, "totp": None,
                      "uris": [], "fido2Credentials": []})); sys.exit(0)
if argv[:2] == ["create", "item"]:
    item = json.loads(base64.b64decode(stdin).decode("utf-8"))
    item.update({"id": "bw-item-1", "creationDate": "2026-01-01T00:00:00Z"})
    with open(state_path, "w", encoding="utf-8") as state:
        json.dump(item, state)
    print(json.dumps(item)); sys.exit(0)
if argv[:2] == ["get", "item"]:
    with open(state_path, encoding="utf-8") as state:
        print(state.read())
    sys.exit(0)
if argv[:2] == ["list", "items"]:
    if not os.path.exists(state_path):
        print("[]"); sys.exit(0)
    with open(state_path, encoding="utf-8") as state:
        print(json.dumps([json.load(state)]))
    sys.exit(0)
if argv[:3] == ["edit", "item", "bw-item-1"]:
    item = json.loads(base64.b64decode(stdin).decode("utf-8"))
    with open(state_path, "w", encoding="utf-8") as state:
        json.dump(item, state)
    print(json.dumps(item)); sys.exit(0)
if argv[:3] == ["delete", "item", "bw-item-1"]:
    os.remove(state_path)
    print(json.dumps({"id": "bw-item-1"})); sys.exit(0)
sys.exit(2)
'''


@pytest.mark.skipif(os.name == "nt", reason="fake bw is a shebang script")
def test_bitwarden_create_and_update_send_secrets_only_over_stdin(tmp_path, monkeypatch):
    executable = tmp_path / "bw"
    executable.write_text(_FAKE_BW, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    unlock_mod.lock()
    generation = unlock_mod.begin_unlock("bitwarden")
    assert unlock_mod.store_session_token("bitwarden", "session-token", generation)
    backend = BitwardenLoginBackend({"binary_path": str(executable)})

    created = backend.create_login(label="Example", origin="https://example.com",
                                   identifier_type="email", identifier="jane@example.com",
                                   password="first secret", otp_secret="JBSWY3DPEHPK3PXP")
    updated = backend.update_login(created.id, label="Example", origin="https://example.com",
                                   identifier_type="email", identifier="jane@example.com",
                                   password="second secret")

    assert created.id == updated.id == "bw:bw-item-1"
    calls = [json.loads(line) for line in (tmp_path / "bw-write.log").read_text(encoding="utf-8").splitlines()]
    create_call = next(call for call in calls if call["argv"][:2] == ["create", "item"])
    edit_call = next(call for call in calls if call["argv"][:3] == ["edit", "item", "bw-item-1"])
    assert all(not call["leaked_env"] for call in calls)
    assert all("first secret" not in " ".join(call["argv"]) and
               "second secret" not in " ".join(call["argv"]) for call in calls)
    assert "first secret" not in create_call["stdin"]
    assert "second secret" not in edit_call["stdin"]
    assert "first secret" in base64.b64decode(create_call["stdin"]).decode("utf-8")
    assert "second secret" in base64.b64decode(edit_call["stdin"]).decode("utf-8")
    final_item = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert final_item["login"]["password"] == "second secret"
    assert final_item["login"]["totp"] == "JBSWY3DPEHPK3PXP"
    assert backend.remove_item(updated.id) is True
    assert backend.remove_item(updated.id) is False
    assert not (tmp_path / "state.json").exists()
    assert "BW_SESSION" not in os.environ
    unlock_mod.lock()


def test_bitwarden_write_error_redacts_password_and_encoded_payload():
    password = "failure-path secret"
    payload = {"type": 1, "login": {"username": "jane@example.com", "password": password}}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    unlock_mod.lock()
    generation = unlock_mod.begin_unlock("bitwarden")
    assert unlock_mod.store_session_token("bitwarden", "session-token", generation)
    backend = BitwardenLoginBackend({"binary_path": "/fake/bw"})

    proc = type("Proc", (), {"returncode": 1, "stdout": "", "stderr": f"rejected {password} {encoded}"})()
    with patch("agent.vault_backends.bitwarden.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError) as exc_info:
            backend._run_encoded(payload, "create", "item")

    message = str(exc_info.value)
    assert password not in message
    assert encoded not in message
    assert message.count("[REDACTED]") == 2
    unlock_mod.lock()


def test_bitwarden_remove_requires_unlock():
    unlock_mod.lock()
    backend = BitwardenLoginBackend({"binary_path": "/fake/bw"})

    with pytest.raises(UnlockRequired):
        backend.remove_item("bw:item-1")


def test_cli_remove_unlocks_external_backend_before_removal():
    from hermes_cli import vault as vault_cli

    class FakeBackend:
        display_name = "Bitwarden"
        needs_unlock = True

        def __init__(self):
            self.unlocked_with = None

        def is_unlocked(self):
            return self.unlocked_with is not None

        def unlock(self, password):
            self.unlocked_with = password

    class FakeBroker:
        def __init__(self, backend):
            self.backend = backend
            self.removed = []

        def backend_for_handle(self, handle):
            return self.backend

        def remove_item(self, handle):
            self.removed.append(handle)
            return True

    backend = FakeBackend()
    broker = FakeBroker(backend)
    console = MagicMock()
    with patch("agent.credential_broker.get_credential_broker", return_value=broker), \
         patch.object(vault_cli, "_console", return_value=console), \
         patch.object(vault_cli.getpass, "getpass", return_value="master secret"):
        vault_cli._cmd_rm(SimpleNamespace(handle="bw:item-1"))

    assert backend.unlocked_with == "master secret"
    assert broker.removed == ["bw:item-1"]
    assert "master secret" not in str(console.print.call_args_list)


def test_browser_save_routes_through_broker_without_returning_password(monkeypatch):
    from agent.vault_backends import unlock as unlock_callbacks
    from tools import browser_vault_tool

    class FakeBroker:
        def __init__(self):
            self.backend = LocalLoginBackend()
            self.calls = []

        def write_backend(self):
            return self.backend

        def save_login(self, **kwargs):
            from agent.vault_backends.base import LoginSaveResult
            from agent.vault_store import VaultItemMeta
            self.calls.append(kwargs)
            return LoginSaveResult(VaultItemMeta("vault_test", "login", kwargs["label"], kwargs["origin"],
                                                 "2026-01-01", kwargs["identifier_type"],
                                                 kwargs["identifier"]), "created")

    broker = FakeBroker()
    answer = {"identifier": "jane@example.com", "password": "browser-only secret"}
    unlock_callbacks.set_save_login_prompt_callback(lambda *_: answer)
    try:
        with patch("agent.credential_broker.get_credential_broker", return_value=broker), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=True), \
             patch.object(browser_vault_tool, "_focus_bound_origin"), \
             patch.object(browser_vault_tool, "_current_page_origin", return_value="https://example.com"), \
             patch.object(browser_vault_tool, "browser_vault_fill", return_value=json.dumps({"success": True})):
            result = json.loads(browser_vault_tool.browser_vault_save_login(task_id="test"))
    finally:
        unlock_callbacks.set_save_login_prompt_callback(None)

    assert result["success"] is True
    assert result["backend"] == "local" and result["action"] == "created"
    assert "browser-only secret" not in json.dumps(result)
    assert broker.calls[0]["password"] == "browser-only secret"
    assert answer == {}
