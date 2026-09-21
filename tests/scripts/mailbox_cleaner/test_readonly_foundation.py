from __future__ import annotations

import hashlib
import inspect
import stat
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from scripts.mailbox_cleaner.classify import classify, urgent_matches
from scripts.mailbox_cleaner.cli import main
from scripts.mailbox_cleaner.clients import MetadataClient, ReadOnlyAdapterError
from scripts.mailbox_cleaner.models import Observation
from scripts.mailbox_cleaner.policy import AccountPolicy, PolicyError
from scripts.mailbox_cleaner.readonly_auth import MissingReadOnlyCredential, scheduled_credential
from scripts.mailbox_cleaner.render import render_report
from scripts.mailbox_cleaner.runner import run_scheduled
from scripts.mailbox_cleaner.state import StateStore


def observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "account": "jobs@example.test",
        "provider": "outlook",
        "message_id": "message-1",
        "sender": "recruiter@example.test",
        "subject": "Interview scheduling",
        "received_at": "2026-07-18T09:00:00Z",
        "preview": "Please choose a time.",
    }
    values.update(overrides)
    return Observation(**values)


def test_observation_is_normalised_immutable_metadata_without_mutation_fields():
    item = observation(subject="  Interview\n  scheduling  ", sender=" Recruiter@Example.TEST ")

    assert item.subject == "Interview scheduling"
    assert item.sender == "recruiter@example.test"
    assert item.fingerprint == hashlib.sha256(
        b"outlook\x00jobs@example.test\x00message-1"
    ).hexdigest()
    assert "body" not in {field.name for field in fields(item)}
    assert not any("action" in field.name or "mutation" in field.name for field in fields(item))
    with pytest.raises(FrozenInstanceError):
        item.subject = "changed"  # type: ignore[misc]


def test_policy_is_immutable_and_rejects_non_read_only_configuration():
    policy = AccountPolicy(account="jobs@example.test", provider="outlook")

    assert policy.allows("list_metadata")
    assert not policy.allows("archive")
    with pytest.raises(PolicyError):
        AccountPolicy(account="jobs@example.test", provider="outlook", allow_mutations=True)
    with pytest.raises(FrozenInstanceError):
        policy.account = "other@example.test"  # type: ignore[misc]


def test_classification_is_deterministic_and_uses_metadata_only():
    item = observation(sender="news@vendor.test", subject="50% discount this week")

    first = classify(item)
    second = classify(item)

    assert first == second
    assert first.category == "promo"
    assert first.confidence == 0.95
    assert urgent_matches(observation(subject="Interview: choose a time"))
    assert not urgent_matches(observation(sender="noreply@vendor.test"))


def test_metadata_adapter_requires_explicit_matching_account_and_allows_reads_only():
    calls: list[tuple[str, str]] = []

    def read(method: str, account: str) -> list[dict[str, str]]:
        calls.append((method, account))
        return [{"id": "message-1", "subject": "Hello", "from": "sender@example.test"}]

    client = MetadataClient(AccountPolicy(account="jobs@example.test", provider="outlook"), read)

    items = client.list_metadata(account="jobs@example.test")

    assert items[0].message_id == "message-1"
    assert calls == [("list_metadata", "jobs@example.test")]
    with pytest.raises(ReadOnlyAdapterError):
        client.list_metadata(account="")
    with pytest.raises(ReadOnlyAdapterError):
        client.list_metadata(account="other@example.test")
    with pytest.raises(ReadOnlyAdapterError):
        client.call("archive", account="jobs@example.test")


def test_scheduled_credential_is_fail_closed_and_never_creates_cache(tmp_path: Path):
    cache = tmp_path / "oauth-cache.json"

    with pytest.raises(MissingReadOnlyCredential):
        scheduled_credential({}, "MAILBOX_TOKEN", cache)

    assert not cache.exists()
    assert scheduled_credential({"MAILBOX_TOKEN": "read-token"}, "MAILBOX_TOKEN", cache) == "read-token"
    assert not cache.exists()


def test_state_deduplicates_urgent_items_with_sha256_and_writes_private_atomic_file(tmp_path: Path):
    state_path = tmp_path / "state" / "urgent.json"
    store = StateStore(state_path)
    item = observation()

    assert store.record_urgent(item) is True
    assert store.record_urgent(item) is False
    assert state_path.exists()
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert item.fingerprint in state_path.read_text(encoding="utf-8")


def test_report_is_compact_and_escapes_untrusted_text():
    report = render_report([classify(observation(subject="<script>alert(1)</script>"))])

    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "Mailbox read-only report" in report
    assert len(report.splitlines()) <= 8


def test_scheduled_runner_reads_serially_records_urgent_and_returns_escaped_report(tmp_path: Path):
    calls: list[str] = []

    def reader(method: str, account: str) -> list[dict[str, str]]:
        assert method == "list_metadata"
        calls.append(account)
        return [{"id": account, "from": "recruiter@example.test", "subject": "Interview <now>"}]

    policies = [
        AccountPolicy(account="one@example.test", provider="outlook"),
        AccountPolicy(account="two@example.test", provider="outlook"),
    ]
    report = run_scheduled(policies, reader, StateStore(tmp_path / "state.json"))

    assert calls == ["one@example.test", "two@example.test"]
    assert report.urgent_count == 2
    assert "&lt;now&gt;" in report.text


def test_cli_runs_only_against_explicit_local_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    fixture = tmp_path / "messages.json"
    fixture.write_text(
        '[{"account":"jobs@example.test","provider":"outlook","id":"1",'
        '"from":"recruiter@example.test","subject":"Interview"}]',
        encoding="utf-8",
    )

    assert main(["--fixture", str(fixture), "--state", str(tmp_path / "state.json")]) == 0
    assert "Mailbox read-only report" in capsys.readouterr().out


def test_new_entrypoints_do_not_import_or_invoke_legacy_direct_scripts():
    root = Path(__file__).resolve().parents[3]
    entries = [
        root / "scripts" / "mailbox_cleaner_main.py",
        root / "scripts" / "mailbox_cleaner_jobhunt.py",
        root / "scripts" / "mailbox_cleaner_urgent.py",
    ]
    forbidden = ("mailbox_cleaner_main_direct", "mailbox_cleaner_jobhunt_direct", "mailbox_cleaner_mcp")

    for entry in entries:
        source = entry.read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden)
        assert "mailbox_cleaner" in source
        assert inspect.getsourcefile(run_scheduled)

    scheduled_sources = [
        root / "scripts" / "mailbox_cleaner" / name
        for name in ("runner.py", "clients.py", "readonly_auth.py", "cli.py")
    ]
    forbidden_transports = ("requests", "httpx", "urllib", "POST", "oauthlib", "msgraph", "googleapiclient")
    assert not any(
        token in path.read_text(encoding="utf-8")
        for path in scheduled_sources
        for token in forbidden_transports
    )
