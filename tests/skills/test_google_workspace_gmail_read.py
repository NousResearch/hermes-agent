"""Behavior contracts for the account-scoped read-only Gmail adapter."""

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gmail_read_api.py"
)
READONLY = "https://www.googleapis.com/auth/gmail.readonly"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"


@pytest.fixture
def module(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    spec = importlib.util.spec_from_file_location("gmail_read_api_test", SCRIPT_PATH)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.modules.pop(spec.name, None)


def _write_account(
    home: Path, alias="personal", email="person@example.com", scopes=None
):
    directory = home / "google-accounts" / alias
    directory.mkdir(parents=True)
    token = directory / "google_token.json"
    token.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "token": "test-token",
                "refresh_token": "test-refresh",
                "client_id": "test.apps.googleusercontent.com",
                "client_secret": "test-secret",
                "token_uri": "https://oauth2.googleapis.com/token",
                "account": email,
                "scopes": scopes or [READONLY, DRIVE_READONLY],
            }
        )
    )
    token.chmod(0o600)
    return token


class FakeRequest:
    def __init__(self, value):
        self.value = value

    def execute(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class FakeHttpError(Exception):
    def __init__(self, status):
        self.resp = type("Response", (), {"status": status})()


class FakeAttachments:
    def __init__(self, service):
        self.service = service

    def get(self, **kwargs):
        self.service.calls.append(("attachment", kwargs))
        value = self.service.attachment_responses.pop(0)
        return FakeRequest(value)


class FakeMessages:
    def __init__(self, service):
        self.service = service

    def list(self, **kwargs):
        self.service.calls.append(("list", kwargs))
        return FakeRequest(self.service.list_response)

    def get(self, **kwargs):
        self.service.calls.append(("get", kwargs))
        return FakeRequest(self.service.messages[kwargs["id"]])

    def attachments(self):
        return FakeAttachments(self.service)


class FakeUsers:
    def __init__(self, service):
        self.service = service

    def getProfile(self, **kwargs):
        self.service.calls.append(("profile", kwargs))
        value = self.service.profile_responses.pop(0)
        return FakeRequest(value)

    def messages(self):
        return FakeMessages(self.service)


class FakeService:
    def __init__(
        self,
        *,
        profile=None,
        list_response=None,
        messages=None,
        attachments=None,
    ):
        self.profile_responses = list(
            profile or [{"emailAddress": "person@example.com", "historyId": "7"}]
        )
        self.list_response = list_response or {"messages": []}
        self.messages = messages or {}
        self.attachment_responses = list(attachments or [])
        self.calls = []

    def users(self):
        return FakeUsers(self)


def _args(module, *argv):
    return module.build_parser().parse_args(["--account", "personal", *argv])


def _metadata(message_id="m1", thread_id="t1"):
    return {
        "id": message_id,
        "threadId": thread_id,
        "labelIds": ["INBOX"],
        "snippet": "preview",
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "person@example.com"},
                {"name": "Subject", "value": "Contract"},
            ]
        },
    }


def test_context_resolves_only_private_named_account(module, tmp_path):
    home = tmp_path / ".hermes"
    token = _write_account(home)

    context = module.load_account_context("personal", home)

    assert context.alias == "personal"
    assert context.expected_email == "person@example.com"
    assert context.token_path == token
    assert context.source_id == "gmail:personal"


@pytest.mark.parametrize("alias", ["../personal", "Personal", "a/b", "", "a" * 40])
def test_context_rejects_noncanonical_aliases(module, tmp_path, alias):
    with pytest.raises(module.AdapterError, match="INVALID_ACCOUNT_ALIAS"):
        module.load_account_context(alias, tmp_path / ".hermes")


def test_context_rejects_symlink_and_open_permissions(module, tmp_path):
    home = tmp_path / ".hermes"
    token = _write_account(home)
    token.chmod(0o644)
    with pytest.raises(module.AdapterError, match="UNSAFE_ACCOUNT_CREDENTIALS"):
        module.load_account_context("personal", home)

    token.chmod(0o600)
    other = home / "other-token.json"
    other.write_text(token.read_text())
    other.chmod(0o600)
    token.unlink()
    token.symlink_to(other)
    with pytest.raises(module.AdapterError, match="ACCOUNT_CREDENTIALS_MISSING"):
        module.load_account_context("personal", home)


@pytest.mark.parametrize(
    "scopes",
    [
        [DRIVE_READONLY],
        [READONLY, "https://www.googleapis.com/auth/gmail.modify"],
        ["https://mail.google.com/"],
        "not-a-list",
    ],
)
def test_context_requires_exact_readonly_gmail_scope(module, tmp_path, scopes):
    home = tmp_path / ".hermes"
    _write_account(home, scopes=scopes)

    with pytest.raises(module.AdapterError, match="GMAIL_SCOPE_MISMATCH"):
        module.load_account_context("personal", home)


def test_identity_mismatch_fails_before_mailbox_read(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(
        profile=[{"emailAddress": "other@example.com", "historyId": "7"}]
    )
    args = _args(module, "search", "subject:contract")

    with pytest.raises(module.AdapterError, match="ACCOUNT_IDENTITY_MISMATCH"):
        module.execute(args, service=service, hermes_home=home)

    assert [name for name, _ in service.calls] == ["profile"]


def test_search_is_bounded_paginated_and_deduplicated(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(
        list_response={
            "messages": [{"id": "m1"}, {"id": "m1"}, {"id": "m2"}],
            "nextPageToken": "next-1",
        },
        messages={"m1": _metadata("m1", "t1"), "m2": _metadata("m2", "t2")},
    )
    args = _args(
        module,
        "search",
        "subject:contract",
        "--max-results",
        "5",
        "--page-token",
        "page-1",
    )

    output = module.execute(args, service=service, hermes_home=home)

    assert output["source_id"] == "gmail:personal"
    assert output["identity_verified"] is True
    assert [item["source_ref"] for item in output["data"]["items"]] == [
        "gmail:personal:m1",
        "gmail:personal:m2",
    ]
    assert output["data"]["next_page_token"] == "next-1"
    assert output["receipt"]["api_calls"] == 4
    list_call = next(kwargs for name, kwargs in service.calls if name == "list")
    assert list_call == {
        "userId": "me",
        "q": "subject:contract",
        "maxResults": 5,
        "pageToken": "page-1",
    }


def test_get_recurses_mime_and_lists_attachments(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    encoded = module.base64.urlsafe_b64encode(b"hello world").decode()
    message = _metadata()
    message["payload"]["mimeType"] = "multipart/mixed"
    message["payload"]["parts"] = [
        {
            "mimeType": "multipart/alternative",
            "body": {},
            "parts": [{"mimeType": "text/plain", "body": {"data": encoded}}],
        },
        {
            "mimeType": "application/pdf",
            "filename": "contract.pdf",
            "body": {"attachmentId": "att-1", "size": 123},
        },
    ]
    service = FakeService(messages={"m1": message})

    output = module.execute(
        _args(module, "get", "m1"), service=service, hermes_home=home
    )

    assert output["data"]["body"] == "hello world"
    assert output["data"]["body_truncated"] is False
    assert output["data"]["attachments"] == [
        {
            "attachment_id": "att-1",
            "filename": "contract.pdf",
            "mime_type": "application/pdf",
            "size": 123,
        }
    ]


def test_get_truncates_body_by_utf8_bytes(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    message["payload"].update(
        {
            "mimeType": "text/plain",
            "body": {"data": module.base64.urlsafe_b64encode("ééé".encode()).decode()},
        }
    )
    service = FakeService(messages={"m1": message})
    args = _args(module, "get", "m1", "--max-body-bytes", "3")

    output = module.execute(args, service=service, hermes_home=home)

    assert output["data"]["body"] == "é"
    assert output["data"]["body_truncated"] is True


def test_get_rejects_large_message_before_full_read(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    message["sizeEstimate"] = 101
    service = FakeService(messages={"m1": message})
    args = _args(module, "get", "m1", "--max-message-bytes", "100")

    with pytest.raises(module.AdapterError, match="MESSAGE_TOO_LARGE"):
        module.execute(args, service=service, hermes_home=home)

    assert [name for name, _ in service.calls] == ["profile", "get"]


def test_get_rejects_excessive_mime_structure(module, tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    message["payload"].update(
        {
            "mimeType": "multipart/mixed",
            "parts": [{"mimeType": "text/plain", "body": {}}],
        }
    )
    monkeypatch.setattr(module, "HARD_MAX_MIME_PARTS", 1)

    with pytest.raises(module.AdapterError, match="MESSAGE_STRUCTURE_TOO_LARGE"):
        module.execute(
            _args(module, "get", "m1"),
            service=FakeService(messages={"m1": message}),
            hermes_home=home,
        )


def test_attachment_get_is_private_content_addressed_and_idempotent(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    encoded = module.base64.urlsafe_b64encode(b"pdf bytes").decode().rstrip("=")
    service = FakeService(
        profile=[
            {"emailAddress": "person@example.com", "historyId": "7"},
            {"emailAddress": "person@example.com", "historyId": "7"},
        ],
        attachments=[{"data": encoded, "size": 9}, {"data": encoded, "size": 9}],
    )
    args = _args(module, "attachment-get", "m1", "att-1", "--max-bytes", "100")

    first = module.execute(args, service=service, hermes_home=home)
    second = module.execute(args, service=service, hermes_home=home)

    path = Path(first["data"]["path"])
    assert path.read_bytes() == b"pdf bytes"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert first["data"]["reused"] is False
    assert second["data"]["reused"] is True
    assert second["data"]["path"] == first["data"]["path"]


def test_attachment_get_rejects_oversize_before_decode(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(attachments=[{"data": "A" * 200, "size": 150}])

    with pytest.raises(module.AdapterError, match="ATTACHMENT_TOO_LARGE"):
        module.execute(
            _args(module, "attachment-get", "m1", "att-1", "--max-bytes", "10"),
            service=service,
            hermes_home=home,
        )


def test_attachment_storage_rejects_symlink_component(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    outside = tmp_path / "outside"
    outside.mkdir()
    source_cache = home / "source-cache"
    source_cache.mkdir()
    (source_cache / "gmail-read").symlink_to(outside, target_is_directory=True)
    encoded = module.base64.urlsafe_b64encode(b"private").decode()

    with pytest.raises(module.AdapterError, match="UNSAFE_PRIVATE_STORAGE"):
        module.execute(
            _args(module, "attachment-get", "m1", "att-1"),
            service=FakeService(attachments=[{"data": encoded}]),
            hermes_home=home,
        )


def test_retry_and_api_call_budget_are_bounded(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(
        profile=[
            FakeHttpError(503),
            {"emailAddress": "person@example.com", "historyId": "7"},
        ]
    )
    context = module.load_account_context("personal", home)
    client = module.GmailReadClient(
        service, context, module.Limits(60, 2, 2), sleep=lambda _: None
    )

    assert client.profile()["history_id_present"] is True
    assert client.api_calls == 2

    exhausted = module.GmailReadClient(
        FakeService(), context, module.Limits(60, 1, 1), sleep=lambda _: None
    )
    exhausted.verify_identity()
    with pytest.raises(module.AdapterError, match="API_CALL_BUDGET_EXCEEDED"):
        exhausted.search("is:unread", 1, None)


def test_receipt_is_private_and_body_free(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    query = "subject:private-contract"
    output = module.execute(
        _args(module, "search", query),
        service=FakeService(),
        hermes_home=home,
    )

    receipt_path = Path(output["receipt"]["path"])
    receipt_text = receipt_path.read_text()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert query not in receipt_text
    assert "person@example.com" not in receipt_text
    assert json.loads(receipt_text)["identity_verified"] is True


def test_parser_has_no_mutation_commands(module):
    parser = module.build_parser()
    help_text = parser.format_help()
    assert "send" not in help_text
    assert "reply" not in help_text
    assert "modify" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["--account", "personal", "send"])


def test_main_returns_typed_error_without_exception_detail(module, monkeypatch, capsys):
    def fail(_args):
        raise module.AdapterError("REAUTH_REQUIRED", exit_code=3)

    monkeypatch.setattr(module, "execute", fail)
    code = module.main(["--account", "personal", "profile"])

    error = json.loads(capsys.readouterr().err)
    assert code == 3
    assert error == {
        "schema_version": 1,
        "status": "blocked",
        "error": {"code": "REAUTH_REQUIRED"},
    }
    receipts = list(
        (
            Path(module.get_hermes_home())
            / "source-cache"
            / "gmail-read"
            / "personal"
            / "receipts"
        ).glob("*.json")
    )
    assert len(receipts) == 1
    stored = json.loads(receipts[0].read_text())
    assert stored["error_code"] == "REAUTH_REQUIRED"
    assert stored["identity_verified"] is False
