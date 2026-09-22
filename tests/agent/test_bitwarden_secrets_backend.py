import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from agent.credential_broker import CredentialBroker
from agent.vault_backends.bitwarden_secrets import BitwardenSecretsLoginBackend
from agent.vault_store import VaultError
from gateway.credential_capture import prepare_credential_capture, save_authorized_credential_capture
from gateway.platforms.event import MessageEvent


def _response(data=None, *, success=True, error=""):
    return SimpleNamespace(success=success, data=data, error_message=error)


class _Secrets:
    def __init__(self, organization_id, project_id):
        self.organization_id = organization_id
        self.project_id = project_id
        self.rows = {}
        self.calls = []

    def list(self, organization_id):
        self.calls.append(("list", organization_id))
        rows = [
            SimpleNamespace(id=row.id, key=row.key, project_ids=[row.project_id])
            for row in self.rows.values()
        ]
        return _response(SimpleNamespace(data=rows))

    def get(self, item_id):
        self.calls.append(("get", item_id))
        row = self.rows.get(str(item_id))
        return _response(row, success=row is not None, error="not found")

    def create(self, organization_id, key, value, note, project_ids):
        self.calls.append(("create", organization_id, key, value, note, project_ids))
        item_id = str(uuid4())
        row = SimpleNamespace(
            id=item_id, key=key, value=value, note=note,
            project_id=project_ids[0], creation_date="2026-01-01T00:00:00Z",
        )
        self.rows[item_id] = row
        return _response(row)

    def update(self, organization_id, item_id, key, value, note, project_ids):
        self.calls.append(("update", organization_id, item_id, key, value, note, project_ids))
        row = self.rows[item_id]
        row.key, row.value, row.note, row.project_id = key, value, note, project_ids[0]
        return _response(row)

    def delete(self, ids):
        self.calls.append(("delete", ids))
        for item_id in ids:
            self.rows.pop(item_id, None)
        return _response(SimpleNamespace(data=[]))


class _Client:
    def __init__(self, secrets_api, organization_id=None):
        self._secrets = secrets_api
        self._organization_id = organization_id

    def secrets(self):
        return self._secrets

    def projects(self):
        organization_id = self._organization_id

        class Projects:
            def get(self, project_id):
                return _response(SimpleNamespace(organization_id=organization_id))

        return Projects()


@pytest.fixture
def backend(monkeypatch):
    organization_id, project_id = uuid4(), uuid4()
    api = _Secrets(organization_id, project_id)
    backend = BitwardenSecretsLoginBackend({
        "enabled": True,
        "organization_id": str(organization_id),
        "project_id": str(project_id),
        "access_token_env": "TEST_BWS_ACCESS_TOKEN",
    })
    monkeypatch.setenv("TEST_BWS_ACCESS_TOKEN", "machine-token-never-log")
    monkeypatch.setattr(backend, "_make_client", lambda token: _Client(api))
    return backend, api


def test_create_list_resolve_update_and_remove_are_model_blind(backend):
    manager, api = backend
    identifier = "private-user@example.com"
    password = "private-password"
    meta = manager.create_login(
        label="Example", origin="https://example.com/login", identifier_type="email",
        identifier=identifier, password=password,
    )
    assert meta.id.startswith("bws:") and meta.identifier is None
    [listed] = manager.list_items()
    assert listed.id == meta.id and listed.origin == "https://example.com"
    assert listed.identifier is None and identifier not in listed.label
    create_call = next(call for call in api.calls if call[0] == "create")
    assert identifier not in create_call[2] and password not in create_call[2]
    assert json.loads(create_call[3])["identifier"] == identifier

    login = manager.resolve_login(meta.id)
    assert login["identifier"] == identifier and login["password"] == password
    assert manager.find_login("https://example.com", identifier).id == meta.id

    updated = manager.update_login(
        meta.id, label="Example updated", origin="https://example.com",
        identifier_type="email", identifier=identifier, password="new-private-password",
    )
    assert updated.label == "Example updated"
    assert manager.resolve_password(meta.id) == "new-private-password"
    assert manager.remove_item(meta.id)
    assert manager.get_meta(meta.id) is None


def test_save_paths_do_not_refetch_successful_writes(backend):
    manager, api = backend
    meta = manager.create_login(
        label="Example", origin="https://example.com", identifier_type="username",
        identifier="private-user", password="private-password",
    )
    assert [call[0] for call in api.calls] == ["create"]

    api.calls.clear()
    found = manager.find_login("https://example.com", "private-user")
    assert found and found.id == meta.id
    assert [call[0] for call in api.calls] == ["list", "get"]

    api.calls.clear()
    [listed] = manager.list_items()
    assert listed.id == meta.id and listed.label == "example.com"
    assert [call[0] for call in api.calls] == ["list"]

    api.calls.clear()
    manager.update_login(
        meta.id, label="Example", origin="https://example.com",
        identifier_type="username", identifier="private-user", password="updated-password",
    )
    assert [call[0] for call in api.calls] == ["get", "update"]


def test_authentication_error_scrubs_machine_token(monkeypatch):
    token = "machine-token-never-log"
    manager = BitwardenSecretsLoginBackend({"access_token_env": "TEST_BWS_ACCESS_TOKEN"})
    monkeypatch.setenv("TEST_BWS_ACCESS_TOKEN", token)

    def fail(value):
        raise RuntimeError(f"bad token {value}")

    monkeypatch.setattr(manager, "_make_client", fail)
    with pytest.raises(VaultError) as error:
        manager._client()
    assert token not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_identifier_in_user_label_is_redacted_from_metadata(backend):
    manager, _api = backend
    identifier = "label-user@example.com"
    created = manager.create_login(
        label=f"Work {identifier}", origin="https://example.com", identifier_type="email",
        identifier=identifier, password="private-password",
    )
    assert identifier not in created.label
    assert identifier not in manager.list_items()[0].label


def test_invalid_project_ids_fail_before_network(monkeypatch):
    manager = BitwardenSecretsLoginBackend({
        "organization_id": str(UUID(int=0)), "project_id": "not-a-uuid",
    })
    with pytest.raises(VaultError, match="valid project_id"):
        manager._ids()


def test_organization_id_is_derived_from_project(backend):
    manager, api = backend
    manager.cfg.pop("organization_id")
    manager._organization_id_cache = None
    manager._make_client = lambda token: _Client(api, api.organization_id)
    assert manager._ids()[0] == api.organization_id


def test_authorized_chat_save_to_bws_then_model_blind_browser_fill(backend, monkeypatch):
    from tools import browser_vault_tool

    manager, _api = backend
    broker = CredentialBroker([manager], {"write_backend": "bitwarden_secrets"})
    identifier = "chat-user@example.com"
    password = "chat-password-never-return"
    event = MessageEvent(
        "保存账号密码\n网站: https://example.com/login\n"
        f"账号: {identifier}\n密码: {password}"
    )
    assert prepare_credential_capture(event)
    saved = save_authorized_credential_capture(event, broker)
    assert saved.reply and "bws:" in saved.reply
    handle = manager.list_items()[0].id

    controls = [
        {"autocomplete": "email", "formIndex": 0, "index": 0, "name": "email", "type": "email"},
        {"autocomplete": "current-password", "formIndex": 0, "index": 1, "name": "password", "type": "password"},
    ]

    def eval_public(task_id, expression):
        if "location.href" in expression:
            return {"success": True, "result": "https://example.com/login"}
        return {"success": True, "result": json.dumps(controls)}

    secret_expressions = []

    def eval_secret(task_id, expression):
        secret_expressions.append(expression)
        return {"success": True, "result": json.dumps({"filled": 2})}

    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: [manager])
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [manager])
    monkeypatch.setattr(browser_vault_tool, "_eval_js", eval_public)
    monkeypatch.setattr(browser_vault_tool, "_eval_js_secret", eval_secret)
    listed = browser_vault_tool.browser_vault_list()
    filled = browser_vault_tool.browser_vault_fill(handle)

    assert identifier not in listed and password not in listed
    assert identifier not in filled and password not in filled
    assert json.loads(filled)["filled_fields"] == 2
    assert identifier in secret_expressions[0] and password in secret_expressions[0]
