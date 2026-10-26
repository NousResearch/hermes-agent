"""Behavior contracts for the account-scoped read-only Gmail adapter."""

import importlib.util
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    home: Path,
    alias="personal",
    email="person@example.com",
    scopes=None,
    *,
    mapping_email=None,
    token_email=None,
    mapping_shape="flat",
):
    accounts_root = home / "google-accounts"
    directory = accounts_root / alias
    directory.mkdir(parents=True)
    mapping = accounts_root / "accounts.json"
    expected = mapping_email or email
    if mapping_shape == "flat":
        mapping_payload = {"accounts": {alias: expected}}
    elif mapping_shape == "versioned":
        mapping_payload = {
            "schema_version": 1,
            "accounts": {alias: {"email": expected}},
        }
    else:
        raise AssertionError("unknown fixture mapping shape")
    mapping.write_text(json.dumps(mapping_payload))
    mapping.chmod(0o600)
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
                "account": token_email or email,
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
        "sizeEstimate": 512,
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
    mapping = home / "google-accounts" / "accounts.json"
    mapping_before = mapping.read_bytes()

    context = module.load_account_context("personal", home)

    assert context.alias == "personal"
    assert context.expected_email == "person@example.com"
    assert context.token_path == token
    assert context.source_id == "gmail:personal"
    assert json.loads(mapping_before) == {
        "accounts": {"personal": "person@example.com"}
    }
    assert mapping.read_bytes() == mapping_before


def test_context_supports_versioned_nested_mapping(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home, mapping_shape="versioned")

    context = module.load_account_context("personal", home)

    assert context.expected_email == "person@example.com"


@pytest.mark.parametrize(
    "mapping_payload",
    [
        {
            "accounts": {
                "personal": "person@example.com",
                "company": {"email": "company@example.com"},
            }
        },
        {"accounts": {"personal": {"email": "person@example.com"}}},
        {
            "schema_version": 1,
            "accounts": {"personal": "person@example.com"},
        },
        {
            "schema_version": 1,
            "accounts": {"personal": {"email": "person@example.com"}},
            "unexpected": True,
        },
    ],
)
def test_context_rejects_malformed_or_mixed_mapping(module, tmp_path, mapping_payload):
    home = tmp_path / ".hermes"
    _write_account(home)
    mapping = home / "google-accounts" / "accounts.json"
    mapping.write_text(json.dumps(mapping_payload))
    mapping.chmod(0o600)

    with pytest.raises(module.AdapterError, match="INVALID_ACCOUNT_MAPPING"):
        module.load_account_context("personal", home)


def test_context_rejects_duplicate_normalized_mailbox_identities(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    mapping = home / "google-accounts" / "accounts.json"
    mapping.write_text(
        json.dumps(
            {
                "accounts": {
                    "personal": "Person@Example.com",
                    "company": "person@example.com",
                }
            }
        )
    )
    mapping.chmod(0o600)

    with pytest.raises(module.AdapterError, match="INVALID_ACCOUNT_MAPPING"):
        module.load_account_context("personal", home)


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


def test_context_requires_owner_only_independent_mapping(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    mapping = home / "google-accounts" / "accounts.json"
    mapping.chmod(0o644)

    with pytest.raises(module.AdapterError, match="UNSAFE_ACCOUNT_MAPPING"):
        module.load_account_context("personal", home)


@pytest.mark.parametrize(
    "scopes",
    [
        [DRIVE_READONLY],
        [READONLY, "https://www.googleapis.com/auth/gmail.modify"],
        [READONLY, "https://www.googleapis.com/auth/calendar.readonly"],
        [READONLY, "https://www.googleapis.com/auth/drive"],
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


def test_swapped_company_token_cannot_claim_personal_alias(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(
        home,
        mapping_email="person@example.com",
        token_email="company@example.com",
    )
    service = FakeService(
        profile=[{"emailAddress": "company@example.com", "historyId": "7"}]
    )

    with pytest.raises(module.AdapterError, match="ACCOUNT_IDENTITY_MISMATCH"):
        module.execute(
            _args(module, "search", "subject:contract"),
            service=service,
            hermes_home=home,
        )

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


def test_search_receipt_key_binds_exact_page_token(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)

    first = module.execute(
        _args(module, "search", "subject:contract", "--page-token", "page-1"),
        service=FakeService(),
        hermes_home=home,
    )
    second = module.execute(
        _args(module, "search", "subject:contract", "--page-token", "page-2"),
        service=FakeService(),
        hermes_home=home,
    )

    assert first["receipt"]["operation_key"] != second["receipt"]["operation_key"]
    assert first["receipt"]["path"] != second["receipt"]["path"]


def test_search_fails_if_provider_exceeds_result_budget(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(
        list_response={"messages": [{"id": "m1"}, {"id": "m2"}]},
        messages={"m1": _metadata("m1"), "m2": _metadata("m2")},
    )

    with pytest.raises(module.AdapterError, match="RESPONSE_ITEM_BUDGET_EXCEEDED"):
        module.execute(
            _args(module, "search", "subject:contract", "--max-results", "1"),
            service=service,
            hermes_home=home,
        )

    assert [name for name, _ in service.calls] == ["profile", "list"]


def test_search_rejects_metadata_id_mismatch(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(
        list_response={"messages": [{"id": "m1"}]},
        messages={"m1": _metadata("m2")},
    )

    with pytest.raises(module.AdapterError, match="MESSAGE_ID_MISMATCH"):
        module.execute(
            _args(module, "search", "subject:contract"),
            service=service,
            hermes_home=home,
        )

    assert [name for name, _ in service.calls] == ["profile", "list", "get"]


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


def test_get_bounds_base64_decode_before_allocating_full_body(
    module, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    raw_body = b"x" * 10_000

    class RejectFullEncode(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("full base64 body must not be encoded")

    message["payload"].update(
        {
            "mimeType": "text/plain",
            "body": {
                "data": RejectFullEncode(
                    module.base64.urlsafe_b64encode(raw_body).decode()
                )
            },
        }
    )
    observed_encoded_lengths = []
    original_decode = module.base64.b64decode

    def bounded_decode(value, *args, **kwargs):
        observed_encoded_lengths.append(len(value))
        return original_decode(value, *args, **kwargs)

    monkeypatch.setattr(module.base64, "b64decode", bounded_decode)
    output = module.execute(
        _args(module, "get", "m1", "--max-body-bytes", "10"),
        service=FakeService(messages={"m1": message}),
        hermes_home=home,
    )

    assert output["data"]["body"] == "x" * 10
    assert output["data"]["body_truncated"] is True
    assert observed_encoded_lengths == [16]


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


@pytest.mark.parametrize("size_estimate", [None, True, "512", -1])
def test_get_requires_real_nonnegative_size_preflight(module, tmp_path, size_estimate):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    if size_estimate is None:
        message.pop("sizeEstimate")
    else:
        message["sizeEstimate"] = size_estimate
    service = FakeService(messages={"m1": message})

    with pytest.raises(module.AdapterError, match="INVALID_MESSAGE_SIZE"):
        module.execute(
            _args(module, "get", "m1"),
            service=service,
            hermes_home=home,
        )

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


def test_attachment_reuse_rejects_broad_mode_and_hardlinks(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    encoded = module.base64.urlsafe_b64encode(b"pdf bytes").decode()
    args = _args(module, "attachment-get", "m1", "att-1")
    first = module.execute(
        args,
        service=FakeService(attachments=[{"data": encoded}]),
        hermes_home=home,
    )
    path = Path(first["data"]["path"])
    path.chmod(0o644)

    with pytest.raises(module.AdapterError, match="UNSAFE_PRIVATE_STORAGE"):
        module.execute(
            args,
            service=FakeService(attachments=[{"data": encoded}]),
            hermes_home=home,
        )

    path.chmod(0o600)
    os.link(path, path.with_name(f"{path.name}.link"))
    with pytest.raises(module.AdapterError, match="UNSAFE_PRIVATE_STORAGE"):
        module.execute(
            args,
            service=FakeService(attachments=[{"data": encoded}]),
            hermes_home=home,
        )


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


def test_attachment_publication_is_complete_before_visible(
    module, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    data = b"complete bytes" * 1024
    digest = hashlib.sha256(data).hexdigest()
    destination = home / "cache" / "attachments" / "m1" / digest
    original_fsync = os.fsync
    synced = []

    def observe_fsync(descriptor):
        info = os.fstat(descriptor)
        synced.append("directory" if stat.S_ISDIR(info.st_mode) else "file")
        if stat.S_ISREG(info.st_mode):
            assert not destination.exists()
        else:
            assert destination.read_bytes() == data
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    path, _, reused = module._store_attachment(
        home / "cache", "m1", data, boundary=home
    )

    assert path == destination
    assert reused is False
    assert synced == ["file", "directory"]
    assert list(destination.parent.glob(".*.tmp")) == []


def test_concurrent_attachment_publishers_reuse_the_complete_winner(
    module, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    data = b"same content" * 1024
    rendezvous = threading.Barrier(2)
    release_winner = threading.Event()
    original_link = os.link

    def publish_then_pause(source, destination):
        rendezvous.wait(timeout=10)
        original_link(source, destination)
        assert release_winner.wait(timeout=10)

    monkeypatch.setattr(os, "link", publish_then_pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                module._store_attachment, home / "cache", "m1", data, boundary=home
            )
            for _ in range(2)
        ]
        try:
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            assert len(done) == 1
            path, _, reused = next(iter(done)).result()
            assert reused is True
            assert path.read_bytes() == data
            # Winner still owns its publication link while the loser reads.
            assert path.stat().st_nlink == 2
        finally:
            release_winner.set()
        results = [future.result(timeout=10) for future in futures]

    assert sorted(result[2] for result in results) == [False, True]
    assert results[0][0] == results[1][0]
    assert results[0][0].stat().st_nlink == 1


@pytest.mark.parametrize("phase", ["before_publication", "after_publication"])
def test_attachment_process_interruption_can_be_retried(module, tmp_path, phase):
    home = tmp_path / ".hermes"
    data = b"durable attachment bytes"
    script = r"""
import importlib.util, os, stat, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gmail_crash_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
home = Path(sys.argv[2])
if sys.argv[3] == "before_publication":
    real_fsync = os.fsync
    def interrupt_write(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.ftruncate(fd, 4)
            os._exit(27)
        real_fsync(fd)
    os.fsync = interrupt_write
else:
    real_link = os.link
    def interrupt_after_link(source, destination):
        real_link(source, destination)
        os._exit(27)
    os.link = interrupt_after_link
module._store_attachment(home / "cache", "m1", b"durable attachment bytes", boundary=home)
"""
    outcome = subprocess.run(
        [sys.executable, "-c", script, str(SCRIPT_PATH), str(home), phase],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert outcome.returncode == 27, outcome.stderr.decode()
    destination = (
        home / "cache" / "attachments" / "m1" / hashlib.sha256(data).hexdigest()
    )
    if phase == "before_publication":
        assert not destination.exists()
    else:
        assert destination.read_bytes() == data
        assert destination.stat().st_nlink == 2

    path, _, reused = module._store_attachment(
        home / "cache", "m1", data, boundary=home
    )

    assert path.read_bytes() == data
    assert reused is (phase == "after_publication")


def test_attachment_publication_preserves_a_conflicting_existing_file(module, tmp_path):
    home = tmp_path / ".hermes"
    data = b"expected content"
    digest = hashlib.sha256(data).hexdigest()
    destination = home / "cache" / "attachments" / "m1" / digest
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"incomplete old content")
    destination.chmod(0o600)

    with pytest.raises(module.AdapterError, match="ATTACHMENT_STORAGE_MISMATCH"):
        module._store_attachment(home / "cache", "m1", data, boundary=home)

    assert destination.read_bytes() == b"incomplete old content"


@pytest.mark.parametrize("fail_directory", [False, True])
def test_attachment_sync_failure_is_redacted_and_retryable(
    module, tmp_path, monkeypatch, fail_directory
):
    home = tmp_path / ".hermes"
    _write_account(home)
    data = b"complete private bytes"
    encoded = module.base64.urlsafe_b64encode(data).decode()
    original_fsync = os.fsync

    def fail_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) is fail_directory:
            raise OSError("private filesystem detail")
        original_fsync(descriptor)

    args = _args(module, "attachment-get", "m1", "att-1")
    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(module.AdapterError, match="^ATTACHMENT_STORAGE_UNAVAILABLE$"):
        module.execute(
            args, service=FakeService(attachments=[{"data": encoded}]), hermes_home=home
        )

    monkeypatch.setattr(os, "fsync", original_fsync)
    result = module.execute(
        args, service=FakeService(attachments=[{"data": encoded}]), hermes_home=home
    )

    assert Path(result["data"]["path"]).read_bytes() == data
    assert result["data"]["reused"] is fail_directory


@pytest.mark.parametrize("field,value", [("id", 123), ("threadId", True)])
def test_provider_message_ids_must_be_strings(module, tmp_path, field, value):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    message[field] = value
    service = FakeService(
        list_response={"messages": [{"id": "m1"}]}, messages={"m1": message}
    )

    with pytest.raises(module.AdapterError, match="INVALID_.*_ID"):
        module.execute(
            _args(module, "search", "contract"), service=service, hermes_home=home
        )


@pytest.mark.parametrize("value", [123, True, [], {}])
def test_provider_page_token_must_be_a_string(module, tmp_path, value):
    home = tmp_path / ".hermes"
    _write_account(home)
    service = FakeService(list_response={"nextPageToken": value})

    with pytest.raises(module.AdapterError, match="INVALID_PAGE_TOKEN"):
        module.execute(
            _args(module, "search", "contract"), service=service, hermes_home=home
        )


def test_provider_attachment_id_must_be_a_string(module, tmp_path):
    home = tmp_path / ".hermes"
    _write_account(home)
    message = _metadata()
    message["payload"]["body"] = {"attachmentId": 123}

    with pytest.raises(module.AdapterError, match="INVALID_ATTACHMENT_ID"):
        module.execute(
            _args(module, "get", "m1"),
            service=FakeService(messages={"m1": message}),
            hermes_home=home,
        )


@pytest.mark.parametrize("encoded", ["cGRm$", "cG Rm", "+///", "YQ===", "YR", "YQ="])
def test_attachment_rejects_malformed_base64_without_storing(module, tmp_path, encoded):
    home = tmp_path / ".hermes"
    _write_account(home)

    with pytest.raises(module.AdapterError, match="INVALID_ATTACHMENT_DATA"):
        module.execute(
            _args(module, "attachment-get", "m1", "att-1"),
            service=FakeService(attachments=[{"data": encoded}]),
            hermes_home=home,
        )

    assert not (
        home / "source-cache" / "gmail-read" / "personal" / "attachments"
    ).exists()


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


def test_request_timeout_is_capped_by_declared_and_remaining_deadline(
    module, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    _write_account(home)
    captured = {}
    built_service = FakeService()
    built_service._http = types.SimpleNamespace(
        http=types.SimpleNamespace(timeout=999.0)
    )
    now = [0.0]

    def fake_build(_context, *, timeout_s):
        captured["build_timeout"] = timeout_s
        now[0] = 4.0
        return built_service

    monkeypatch.setattr(module, "build_service", fake_build)
    module.execute(
        _args(module, "--timeout", "30", "--deadline", "5", "profile"),
        hermes_home=home,
        monotonic=lambda: now[0],
    )
    assert captured["build_timeout"] == 5
    assert built_service._http.http.timeout == 1

    clock = iter([0.0, 7.0])
    remaining_service = FakeService()
    remaining_service._http = types.SimpleNamespace(
        http=types.SimpleNamespace(timeout=999.0)
    )
    context = module.load_account_context("personal", home)
    client = module.GmailReadClient(
        remaining_service,
        context,
        module.Limits(10, 10, 1),
        monotonic=lambda: next(clock),
        request_timeout_s=30,
    )
    client.profile()
    assert remaining_service._http.http.timeout == 3


def test_refresh_error_is_typed_reauth_without_detail(module, tmp_path):
    try:
        from google.auth.exceptions import RefreshError
    except ImportError:
        RefreshError = type(
            "RefreshError",
            (Exception,),
            {"__module__": "google.auth.exceptions"},
        )

    home = tmp_path / ".hermes"
    _write_account(home)
    args = _args(module, "profile")

    with pytest.raises(module.AdapterError) as raised:
        module.execute(
            args,
            service=FakeService(
                profile=[RefreshError("invalid_grant: private detail")]
            ),
            hermes_home=home,
        )

    error = raised.value
    assert error.code == "REAUTH_REQUIRED"
    assert error.exit_code == 3
    assert error.api_calls == 1
    assert error.identity_verified is False
    assert "invalid_grant" not in str(error)


def test_error_receipt_binds_request_runtime_and_distinct_times(
    module, tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    _write_account(home)
    times = iter(
        ["2026-10-21T10:00:00Z", "2026-10-21T10:00:05Z", "2026-10-21T10:00:06Z"]
    )
    monkeypatch.setattr(module, "_utcnow", lambda: next(times))
    args = _args(
        module,
        "--attempts",
        "1",
        "search",
        "subject:private-contract",
        "--page-token",
        "page-1",
    )

    with pytest.raises(module.AdapterError) as raised:
        module.execute(
            args,
            service=FakeService(list_response=FakeHttpError(500)),
            hermes_home=home,
        )
    error = raised.value
    receipt_path = module._persist_error_receipt(args, error, home)
    receipt = json.loads(receipt_path.read_text())

    assert receipt["api_calls"] == 2
    assert receipt["identity_verified"] is True
    assert receipt["error_code"] == "GMAIL_API_UNAVAILABLE"
    assert receipt["started_at"] == "2026-10-21T10:00:00Z"
    assert receipt["finished_at"] == "2026-10-21T10:00:05Z"
    assert "private-contract" not in receipt_path.read_text()
    other_args = _args(
        module,
        "--attempts",
        "1",
        "search",
        "subject:private-contract",
        "--page-token",
        "page-2",
    )
    other_error = module.AdapterError(
        "GMAIL_API_UNAVAILABLE",
        api_calls=2,
        identity_verified=True,
        request_shape=module._search_shape(other_args),
        started_at="2026-10-21T10:00:01Z",
    )
    other_path = module._persist_error_receipt(other_args, other_error, home)
    assert other_path != receipt_path


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
