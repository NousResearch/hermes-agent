"""Production-hardening: audit lifecycle, recovery, version checks, logging, backup.

Findings 8-13 and 16 of the readiness audit. They share a theme: NOVA knew things and did
not say them, or wrote things and never checked them. Each test below asserts the saying or
the checking, because the doing was mostly already there.
"""

from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path

import pytest

from nova.audit import AuditLog
from nova.audit.integrity import (
    read_seal,
    rotate,
    seal,
    segments,
    verify,
    write_seal,
)
from nova.backup import BACKUP_VERSION, create, read_manifest, restore
from nova.errors import RuntimeAdapterError
from nova.observability import JsonFormatter, configure, operation, set_correlation_id
from nova.runtime.hermes.compat import SUPPORTED_MAJOR_MINOR, check as compat_check, parse_version
from nova.runtime.hermes.materialize import (
    PROVENANCE_VERSION,
    Provenance,
    check_provenance_version,
)


def noisy_log(tmp_path, events: int = 200, **kwargs) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl", tenant_id="acme", **kwargs)
    for index in range(events):
        log.record("test.event", correlation_id="c", subject=f"s{index}", detail={"n": index})
    return log


# -- finding 8: the log must be finite ---------------------------------------


def test_the_log_rotates_once_it_passes_its_bound(tmp_path):
    noisy_log(tmp_path, 300, max_bytes=4096, keep=3)
    found = segments(tmp_path / "audit.jsonl")
    assert len(found) > 1


def test_retention_is_bounded_so_a_disk_cannot_fill(tmp_path):
    """The failure this prevents is an agent platform that stops being able to record what
    it did while still doing it."""
    noisy_log(tmp_path, 2000, max_bytes=2048, keep=2)
    assert len(segments(tmp_path / "audit.jsonl")) <= 3


def test_reading_spans_every_segment(tmp_path):
    """Or rotation would manufacture unfinished intents that never existed."""
    log = noisy_log(tmp_path, 300, max_bytes=4096, keep=9)
    assert len(segments(log.path)) > 1
    assert len(log.read()) > 0
    assert log.open_intents() == []


def test_an_intent_and_its_commit_survive_a_rotation_between_them(tmp_path):
    """The write-ahead guarantee must not be an artefact of both lines being in one file."""
    log = AuditLog(tmp_path / "audit.jsonl", tenant_id="acme", max_bytes=1024, keep=9)
    with log.model_visible_change("agent.materialized", correlation_id="c", subject="a"):
        for index in range(120):  # force a rotation between intent and commit
            log.record("filler", correlation_id="f", subject=str(index), detail={})
    assert len(segments(log.path)) > 1
    assert log.open_intents() == []


def test_rotation_failing_never_fails_the_write(tmp_path, monkeypatch):
    """A full disk must not also stop the record of why the disk filled."""
    import nova.audit.log as log_module

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(log_module, "rotate", explode)
    log = AuditLog(tmp_path / "audit.jsonl", tenant_id="acme", max_bytes=1)
    log.record("test.event", correlation_id="c", subject="s", detail={})
    assert len(log.read()) == 1


def test_rotation_is_a_no_op_below_the_bound(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("small\n", encoding="utf-8")
    assert rotate(path, max_bytes=10_000) is False


# -- finding 9: tampering must be visible -------------------------------------


def test_a_clean_log_verifies(tmp_path):
    log = noisy_log(tmp_path, 50)
    document = seal(log.path, tenant_id="acme")
    assert verify(log.path, document).ok


def test_growth_since_the_seal_is_not_a_finding(tmp_path):
    """A seal that could only be verified against a stopped deployment would be verified
    approximately never."""
    log = noisy_log(tmp_path, 20)
    document = seal(log.path)
    log.record("test.event", correlation_id="c", subject="later", detail={})
    result = verify(log.path, document)
    assert result.ok and result.appended == 1


def test_a_byte_preserving_edit_is_detected(tmp_path):
    """The adversarial case: an attacker careful to keep the file the same length."""
    log = noisy_log(tmp_path, 30)
    document = seal(log.path)
    raw = log.path.read_bytes()
    edited = raw.replace(b'"subject":"s1"', b'"subject":"s9"', 1)
    assert edited != raw, "the test must actually change a byte"
    assert len(edited) == len(raw), "and must not change the length"
    log.path.write_bytes(edited)

    result = verify(log.path, document)
    assert not result.ok
    assert "modified" in result.findings[0]


def test_truncation_is_detected(tmp_path):
    log = noisy_log(tmp_path, 30)
    document = seal(log.path)
    raw = log.path.read_bytes()
    log.path.write_bytes(raw[: len(raw) // 2])
    assert not verify(log.path, document).ok


def test_a_deleted_segment_is_detected(tmp_path):
    log = noisy_log(tmp_path, 300, max_bytes=2048, keep=9)
    document = seal(log.path)
    rotated = [p for p in segments(log.path) if p != log.path]
    rotated[0].unlink()
    result = verify(log.path, document)
    assert not result.ok
    assert "missing" in result.findings[0]


def test_a_segment_created_after_the_seal_is_reported_not_failed(tmp_path):
    """"The seal covers less than you think" is what someone verifying needs to know."""
    log = noisy_log(tmp_path, 20, max_bytes=1024, keep=9)
    document = seal(log.path)
    before = len(segments(log.path))
    for index in range(200):
        log.record("test.event", correlation_id="c", subject=str(index), detail={})
    assert len(segments(log.path)) > before

    result = verify(log.path, document)
    assert any("not covered by this seal" in f for f in result.findings)


def test_a_seal_is_written_unreadable_to_others(tmp_path):
    log = noisy_log(tmp_path, 5)
    path = write_seal(seal(log.path), tmp_path / "seal.json")
    assert path.stat().st_mode & 0o077 == 0


def test_a_seal_round_trips(tmp_path):
    log = noisy_log(tmp_path, 10)
    written = write_seal(seal(log.path, tenant_id="acme"), tmp_path / "s.json")
    assert read_seal(written).tenant_id == "acme"


# -- finding 12: the provenance version must be checked ------------------------


def test_a_newer_marker_is_refused_rather_than_guessed_at():
    """Guessing at a format from the future is how a downgrade destroys a profile a newer
    NOVA is still managing."""
    newer = Provenance(PROVENANCE_VERSION + 1, "a", "d", "0")
    with pytest.raises(RuntimeAdapterError, match="newer NOVA"):
        check_provenance_version(newer, profile_dir=Path("/x"))


def test_the_current_version_is_silent():
    current = Provenance(PROVENANCE_VERSION, "a", "d", "0")
    assert check_provenance_version(current, profile_dir=Path("/x")) == []


def test_an_absent_marker_needs_no_check():
    assert check_provenance_version(None, profile_dir=Path("/x")) == []


# -- finding 13: the runtime version must be checked ---------------------------


def test_the_verified_version_produces_no_warning():
    major, minor = SUPPORTED_MAJOR_MINOR
    assert compat_check(f"{major}.{minor}.1") == []


def test_patch_drift_is_tolerated():
    major, minor = SUPPORTED_MAJOR_MINOR
    assert compat_check(f"{major}.{minor}.99") == []


@pytest.mark.parametrize("version", ["0.22.0", "1.0.0", "0.20.9"])
def test_a_different_minor_warns(version):
    warnings = compat_check(version)
    assert warnings and "verified" in warnings[0]


def test_an_unreadable_version_says_so_rather_than_passing():
    assert compat_check("") != []
    assert compat_check("banana") != []


def test_a_development_version_parses():
    """Refusing it would turn every developer install into a warning."""
    assert parse_version("0.21.1.dev3+g1a2b3c") == (0, 21, 1)


def test_compatibility_reaches_the_apply_report(bundle, runtime, audit, monkeypatch):
    monkeypatch.setattr(type(runtime), "compatibility", lambda self: ["a version warning"])
    from nova.apply import apply_bundle

    report = apply_bundle(bundle, runtime, audit=audit, dry_run=True)
    assert "a version warning" in report.warnings


# -- finding 11: an interrupted run must be diagnosed --------------------------


def test_a_clean_log_reconciles_clean(tmp_path):
    from nova.reconcile import reconcile

    assert reconcile(noisy_log(tmp_path, 5)).clean


def test_an_unfinished_intent_for_a_present_agent_is_resolved(bundle, runtime, audit):
    """A crash between the write and the commit record. Re-running would be a no-op, and
    an operator should be told that rather than left to guess."""
    from nova.apply import apply_bundle
    from nova.reconcile import RESOLVED, reconcile

    apply_bundle(bundle, runtime, audit=audit)
    audit.record(
        "agent.materialized", correlation_id="crashed", subject="customer-support",
        phase="intent",
    )
    result = reconcile(audit, bundle, runtime)
    finding = next(f for f in result.findings if f.subject == "customer-support")
    assert finding.state == RESOLVED
    assert not finding.needs_action


def test_an_unfinished_intent_for_a_missing_agent_needs_action(bundle, runtime, audit):
    from nova.reconcile import INCOMPLETE, reconcile

    audit.record(
        "agent.materialized", correlation_id="crashed", subject="never-created",
        phase="intent",
    )
    result = reconcile(audit, bundle, runtime)
    finding = result.findings[0]
    assert finding.state == INCOMPLETE
    assert result.actionable


def test_reconciliation_names_the_command_that_finishes_the_job(bundle, runtime, audit):
    from nova.reconcile import reconcile

    audit.record(
        "knowledge.indexed", correlation_id="crashed", subject="handbook", phase="intent"
    )
    finding = reconcile(audit, bundle, runtime).findings[0]
    assert "nova knowledge ingest" in finding.detail


def test_reconciliation_without_a_runtime_still_names_what_was_in_flight(tmp_path):
    """Better than a bare count, which told an operator something was wrong and nothing
    about what."""
    from nova.reconcile import UNKNOWN, reconcile

    log = AuditLog(tmp_path / "a.jsonl", tenant_id="acme")
    log.record("agent.materialized", correlation_id="c", subject="x", phase="intent")
    finding = reconcile(log).findings[0]
    assert finding.state == UNKNOWN
    assert finding.subject == "x"


# -- finding 10: operational logging ------------------------------------------


def test_reserved_logrecord_names_do_not_crash(caplog):
    """`created=2` is the obvious field for an apply to log, and it raised KeyError."""
    configure(level="INFO")
    with operation("apply", tenant_id="acme") as trace:
        trace.add(created=2, name="x", module="y", args="z")
    # Reaching here without an exception is the assertion.


def test_the_envelope_survives_a_colliding_caller_field():
    """A field named `message` silently replaced the log line's own."""
    record = logging.LogRecord("nova.apply", logging.INFO, "f", 1, "real message", None, None)
    record.nova_fields = {"message": "a caller field", "created": 2}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "real message"
    assert payload["field_message"] == "a caller field"
    assert payload["created"] == 2


def test_a_trace_logs_its_failure_and_re_raises():
    """Driving the context manager by hand logged the start and never the failure."""
    configure(level="INFO")
    with pytest.raises(ValueError):
        with operation("ingest"):
            raise ValueError("disk full")


def test_the_correlation_id_reaches_the_payload():
    set_correlation_id("abc123")
    record = logging.LogRecord("nova.apply", logging.INFO, "f", 1, "m", None, None)
    assert json.loads(JsonFormatter().format(record))["correlation_id"] == "abc123"
    set_correlation_id("")


def test_logging_is_off_unless_asked_for():
    """A CLI printing structured logs over its own output is hostile to whoever runs it."""
    logger = configure(level="")
    assert not logger.isEnabledFor(logging.CRITICAL)


def test_an_unserialisable_field_does_not_take_the_logs_down():
    record = logging.LogRecord("nova.x", logging.INFO, "f", 1, "m", None, None)
    record.nova_fields = {"obj": object()}
    assert "obj" in json.loads(JsonFormatter().format(record))


# -- finding 16: backup and restore -------------------------------------------


def deployment(tmp_path, bundle, runtime, audit):
    from nova.apply import apply_bundle

    apply_bundle(bundle, runtime, audit=audit)
    return runtime.state_location


def test_a_backup_excludes_credentials_by_default(tmp_path, bundle, runtime, audit, home):
    """An archive that swept up .env would turn every copy into a secret-bearing artifact."""
    deployment(tmp_path, bundle, runtime, audit)
    (home / "profiles" / "operations" / ".env").write_text(
        "ACME_LLM_KEY=super-secret\n", encoding="utf-8"
    )
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme", required_env={"operations": ["ACME_LLM_KEY"]})

    with tarfile.open(archive) as handle:
        names = handle.getnames()
        assert not any(name.endswith(".env") for name in names)
        blob = b"".join(
            handle.extractfile(m).read() for m in handle.getmembers() if m.isfile()
        )
    assert b"super-secret" not in blob


def test_the_manifest_names_the_variables_without_their_values(tmp_path, bundle, runtime, audit, home):
    deployment(tmp_path, bundle, runtime, audit)
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme", required_env={"operations": ["ACME_LLM_KEY"]})
    manifest = read_manifest(archive)
    assert manifest.required_env["operations"] == ["ACME_LLM_KEY"]
    assert manifest.includes_secrets is False


def test_including_secrets_is_explicit_and_recorded(tmp_path, bundle, runtime, audit, home):
    deployment(tmp_path, bundle, runtime, audit)
    (home / "profiles" / "operations" / ".env").write_text("K=v\n", encoding="utf-8")
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme", include_secrets=True)
    assert read_manifest(archive).includes_secrets is True
    with tarfile.open(archive) as handle:
        assert any(name.endswith(".env") for name in handle.getnames())


def test_an_archive_is_written_unreadable_to_others(tmp_path, bundle, runtime, audit, home):
    deployment(tmp_path, bundle, runtime, audit)
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme")
    assert archive.stat().st_mode & 0o077 == 0


def test_the_work_board_is_never_backed_up(tmp_path, bundle, runtime, audit, home):
    """It is the runtime's, shared with its own CLI and dashboard."""
    deployment(tmp_path, bundle, runtime, audit)
    (home / "kanban.db").write_bytes(b"not nova's")
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme")
    with tarfile.open(archive) as handle:
        assert not any("kanban.db" in name for name in handle.getnames())


def test_restore_reports_the_credentials_still_needed(tmp_path, bundle, runtime, audit, home):
    deployment(tmp_path, bundle, runtime, audit)
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme", required_env={"operations": ["ACME_LLM_KEY"]})

    target = tmp_path / "restored"
    report = restore(archive, target, environ={})
    assert report.restored > 0
    assert report.missing_env["operations"] == ["ACME_LLM_KEY"]


def test_a_dry_run_restores_nothing(tmp_path, bundle, runtime, audit, home):
    deployment(tmp_path, bundle, runtime, audit)
    archive = tmp_path / "b.tar.gz"
    create(home, archive, tenant_id="acme")
    target = tmp_path / "restored"
    report = restore(archive, target, dry_run=True, environ={})
    assert report.dry_run and not (target / "profiles").exists()


def test_a_path_traversal_member_is_refused(tmp_path):
    """An archive is untrusted input even when NOVA wrote it — it may have been swapped."""
    import io

    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        manifest = json.dumps({"version": BACKUP_VERSION}).encode()
        info = tarfile.TarInfo("nova-backup.json")
        info.size = len(manifest)
        handle.addfile(info, io.BytesIO(manifest))

        payload = b"pwned"
        evil = tarfile.TarInfo("home/../../../../tmp/nova-escaped")
        evil.size = len(payload)
        handle.addfile(evil, io.BytesIO(payload))

    target = tmp_path / "restored"
    report = restore(archive, target, environ={})
    assert report.restored == 0
    assert not Path("/tmp/nova-escaped").exists()


def test_a_newer_backup_is_refused(tmp_path):
    import io

    archive = tmp_path / "future.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        manifest = json.dumps({"version": BACKUP_VERSION + 1}).encode()
        info = tarfile.TarInfo("nova-backup.json")
        info.size = len(manifest)
        handle.addfile(info, io.BytesIO(manifest))
    with pytest.raises(ValueError, match="newer NOVA"):
        restore(archive, tmp_path / "restored", environ={})
