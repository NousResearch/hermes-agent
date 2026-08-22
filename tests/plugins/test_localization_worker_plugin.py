"""Behavior tests for the native localization-worker plugin."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def worker(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "localization-worker"
    package = "hermes_plugins.localization_worker"
    parent = sys.modules.setdefault("hermes_plugins", types.ModuleType("hermes_plugins"))
    parent.__path__ = []
    spec = importlib.util.spec_from_file_location(
        package, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    setattr(module, "_SOURCE_ROOTS", (tmp_path.resolve(),))
    return module, home


def call(fn, **args):
    result = fn(args)
    assert isinstance(result, str)
    assert len(result.encode("utf-8")) <= 64_000
    return json.loads(result)


def claimed_job(plugin, tmp_path, text="Hello {name}"):
    source = tmp_path / "source.txt"
    source.write_text(text, encoding="utf-8")
    job = call(plugin.create_job, source_path=str(source), target_locale="ko")
    job_id = job["job_id"]
    call(plugin.inspect_job, job_id=job_id)
    call(plugin.extract_segments, job_id=job_id)
    call(plugin.create_chunks, job_id=job_id)
    lease = call(plugin.claim_chunk, job_id=job_id, worker_id="translator")
    lease["output_path"] = job["output_path"]
    return job_id, lease


def test_relative_source_path_is_anchored_to_configured_root(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    created = call(plugin.create_job, source_path="source.txt", target_locale="ko")
    repeated = call(plugin.create_job, source_path="source.txt", target_locale="ko")

    assert created["ok"] is True
    assert repeated["job_id"] == created["job_id"]
    assert call(plugin.inspect_job, job_id=created["job_id"])["state"] == "INSPECTED"


def test_create_job_rejects_source_swapped_to_external_symlink(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("inside", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("external secret", encoding="utf-8")
    original_open = plugin.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == source.name and dir_fd is not None and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin.os, "open", swapping_open)
    rejected = call(plugin.create_job, source_path=str(source), target_locale="ko")

    assert rejected["error"]["code"] == "SOURCE_OUTSIDE_ALLOWED_ROOTS"


def test_source_resists_allowed_root_ancestor_swap(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    trusted_parent = tmp_path / "trusted"
    source_root = trusted_parent / "allowed"
    source_root.mkdir(parents=True)
    source = source_root / "source.txt"
    source.write_text("inside", encoding="utf-8")
    outside_parent = tmp_path / "outside-parent"
    (outside_parent / "allowed").mkdir(parents=True)
    (outside_parent / "allowed" / "source.txt").write_text("outside", encoding="utf-8")
    plugin._SOURCE_ROOTS = (source_root,)
    original_open = plugin.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        hits_current_absolute_open = Path(path) == source_root
        hits_component_open = path == trusted_parent.name and dir_fd is not None
        if not swapped and (hits_current_absolute_open or hits_component_open):
            swapped = True
            trusted_parent.rename(tmp_path / "trusted-original")
            trusted_parent.symlink_to(outside_parent, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin.os, "open", swapping_open)
    rejected = call(plugin.create_job, source_path=str(source), target_locale="ko")

    assert rejected["error"]["code"] == "SOURCE_OUTSIDE_ALLOWED_ROOTS"


def test_source_read_failure_marks_job_failed(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    def fail_read(_fd, _size):
        raise OSError("injected source read failure")

    monkeypatch.setattr(plugin.os, "read", fail_read)
    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "SOURCE_READ_FAILED"
    status = call(plugin.get_job_status, job_id=created["job_id"])
    assert status["state"] == "FAILED"
    assert status["verification_receipt"] is None


def test_source_fifo_is_rejected_without_blocking(worker, tmp_path):
    plugin, _ = worker
    fifo = tmp_path / "source.txt"
    os.mkfifo(fifo)
    result: list[dict] = []
    thread = threading.Thread(
        target=lambda: result.append(call(plugin.create_job, source_path=str(fifo), target_locale="ko")),
        daemon=True,
    )

    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive(), "source FIFO open blocked"
    assert result[0]["error"]["code"] == "SOURCE_NOT_FILE"


def test_source_unix_socket_is_rejected(worker, tmp_path):
    plugin, _ = worker
    short_tmp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="lw-sock-", dir=short_tmp_root) as directory:
        source_root = Path(directory)
        plugin._SOURCE_ROOTS = (source_root,)
        socket_path = source_root / "source.txt"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(socket_path))
            rejected = call(plugin.create_job, source_path=str(socket_path), target_locale="ko")
        finally:
            listener.close()

    assert rejected["error"]["code"] == "SOURCE_NOT_FILE"


def test_invalid_utf8_source_fails_durably(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"valid\n\xffinvalid")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "SOURCE_NOT_UTF8"
    assert "codec" not in rejected["error"]["message"]
    status = call(plugin.get_job_status, job_id=created["job_id"])
    assert status["state"] == "FAILED"
    assert status["verification_receipt"] is None


def test_extract_large_document_returns_bounded_metadata(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "large.txt"
    source.write_text("\n".join(f"line {i}" for i in range(10_000)), encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])

    extracted = call(plugin.extract_segments, job_id=created["job_id"])

    assert extracted["state"] == "EXTRACTED"
    assert extracted["segment_count"] == 10_000
    assert "segments" not in extracted
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "EXTRACTED"


def test_unicode_line_separator_is_rejected(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "unicode-lines.txt"
    source.write_text("A\u2028B", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "UNSUPPORTED_NEWLINE_STYLE"
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "FAILED"


def test_txt_job_runs_end_to_end_and_preserves_text_framing(worker, tmp_path):
    plugin, home = worker
    source = tmp_path / "source.txt"
    output = tmp_path / "translated.txt"
    source.write_bytes(b"\xef\xbb\xbfHello {name}\r\nWorld\r\n")

    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    repeated = call(plugin.create_job, source_path=str(source), target_locale="ko")
    output = Path(created["output_path"])
    assert repeated["job_id"] == created["job_id"]
    job_id = created["job_id"]
    assert Path(created["data_dir"]).is_relative_to(home)

    assert call(plugin.inspect_job, job_id=job_id)["state"] == "INSPECTED"
    extracted = call(plugin.extract_segments, job_id=job_id)
    assert extracted["state"] == "EXTRACTED"
    assert extracted["segment_count"] == 2
    assert "segments" not in extracted
    chunked = call(plugin.create_chunks, job_id=job_id)
    assert chunked["state"] == "CHUNKED"
    lease = call(plugin.claim_chunk, job_id=job_id, worker_id="translator")
    assert lease["state"] == "PROCESSING"
    assert [segment["source_text"] for segment in lease["segments"]] == ["Hello {name}", "World"]
    submitted = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[
            {"segment_id": lease["segments"][0]["segment_id"], "text": "안녕 {name}"},
            {"segment_id": lease["segments"][1]["segment_id"], "text": "세계"},
        ],
    )
    assert submitted["accepted"] is True
    repeated_submit = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[
            {"segment_id": lease["segments"][0]["segment_id"], "text": "안녕 {name}"},
            {"segment_id": lease["segments"][1]["segment_id"], "text": "세계"},
        ],
    )
    assert repeated_submit["accepted"] is True
    assert repeated_submit["idempotent"] is True
    assert call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])["state"] == "VALIDATING"
    assert call(plugin.assemble_output, job_id=job_id)["state"] == "ASSEMBLING"
    verified = call(plugin.verify_output, job_id=job_id)
    assert verified["state"] == "COMPLETED"
    assert verified["verification_receipt"]
    assert output.read_bytes() == "\ufeff안녕 {name}\r\n세계\r\n".encode("utf-8")
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "COMPLETED"
    assert status["verification_receipt"] == verified["verification_receipt"]
    assert status["audit_event_count"] >= 9


def test_translation_unicode_separator_is_preserved_as_content(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    translated = "안녕\u2028세계"
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": translated}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)

    verified = call(plugin.verify_output, job_id=job_id)

    assert verified["state"] == "COMPLETED"
    assert Path(assembled["output_path"]).read_text(encoding="utf-8") == translated


def test_final_chunk_submit_retry_is_idempotent_after_validation(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    translations = [{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}]
    accepted = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=translations,
    )
    assert accepted["idempotent"] is False
    assert call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])["state"] == "VALIDATING"

    retried = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=translations,
    )

    assert retried == {"ok": True, "accepted": True, "idempotent": True, "state": "VALIDATING"}


def test_invalid_transition_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.md"
    source.write_text("Hello", encoding="utf-8")
    job = call(plugin.create_job, source_path=str(source), target_locale="fr")
    assert call(plugin.inspect_job, job_id=job["job_id"])["state"] == "INSPECTED"

    rejected = call(plugin.inspect_job, job_id=job["job_id"])

    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "INVALID_TRANSITION"


def test_stale_lease_submit_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path)

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token="stale-token",
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕 {name}"}],
    )

    assert rejected["error"]["code"] == "STALE_LEASE"


def test_placeholder_mismatch_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path)

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )

    assert rejected["error"]["code"] == "PLACEHOLDER_MISMATCH"


def test_output_reparse_failure_withholds_receipt(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    call(plugin.assemble_output, job_id=job_id)
    Path(lease["output_path"]).write_text("tampered", encoding="utf-8")

    rejected = call(plugin.verify_output, job_id=job_id)
    status = call(plugin.get_job_status, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_REPARSE_MISMATCH"
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_completed_job_detects_later_output_tampering(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    call(plugin.assemble_output, job_id=job_id)
    verified = call(plugin.verify_output, job_id=job_id)
    assert verified["state"] == "COMPLETED"

    Path(lease["output_path"]).write_text("tampered after verification", encoding="utf-8")
    status = call(plugin.get_job_status, job_id=job_id)

    assert status["ok"] is False
    assert status["error"]["code"] == "OUTPUT_CHANGED_AFTER_VERIFICATION"
    assert call(plugin.get_job_status, job_id=job_id)["error"]["code"] == "OUTPUT_CHANGED_AFTER_VERIFICATION"


def test_create_job_owns_output_path_inside_profile_data(worker, tmp_path):
    plugin, home = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")

    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    assert created["ok"] is True
    output = Path(created["output_path"])
    assert output.is_relative_to(home / "plugins" / "localization-worker" / "jobs" / created["job_id"])
    assert output.name == "source.ko.txt"


def test_create_job_rejects_caller_controlled_output_path(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    external = tmp_path / "external.txt"
    source.write_text("Hello", encoding="utf-8")
    external.write_text("do not overwrite", encoding="utf-8")

    rejected = call(
        plugin.create_job,
        source_path=str(source),
        output_path=str(external),
        target_locale="ko",
    )

    assert rejected["error"]["code"] == "CALLER_CONTROLLED_OUTPUT_PATH"
    assert external.read_text(encoding="utf-8") == "do not overwrite"


def test_source_change_after_inspection_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    source.write_text("changed", encoding="utf-8")

    rejected = call(plugin.extract_segments, job_id=created["job_id"])

    assert rejected["error"]["code"] == "SOURCE_CHANGED"
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "FAILED"


def test_expired_lease_is_reclaimed_with_new_fencing_token(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    now = [1_000.0]
    monkeypatch.setattr(plugin, "_epoch", lambda: now[0])
    job_id, first = claimed_job(plugin, tmp_path, text="Hello")
    now[0] += 301

    second = call(plugin.claim_chunk, job_id=job_id, worker_id="replacement")

    assert second["ok"] is True
    assert second["fencing_token"] != first["fencing_token"]
    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=first["chunk_id"],
        fencing_token=first["fencing_token"],
        translations=[{"segment_id": first["segments"][0]["segment_id"], "text": "안녕"}],
    )
    assert rejected["error"]["code"] == "STALE_LEASE"


def test_concurrent_claim_allows_exactly_one_worker(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    call(plugin.extract_segments, job_id=created["job_id"])
    call(plugin.create_chunks, job_id=created["job_id"])
    barrier = threading.Barrier(2)
    results = []

    def claim(worker_id):
        barrier.wait()
        results.append(call(plugin.claim_chunk, job_id=created["job_id"], worker_id=worker_id))

    threads = [threading.Thread(target=claim, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result.get("ok") is True for result in results) == 1
    assert sum(result.get("error", {}).get("code") == "NO_CHUNK_AVAILABLE" for result in results) == 1


def test_chunks_respect_local_source_budget(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("\n".join(["x" * 400 for _ in range(30)]), encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    call(plugin.extract_segments, job_id=created["job_id"])

    chunked = call(plugin.create_chunks, job_id=created["job_id"])

    assert chunked["chunk_count"] > 1
    assert chunked["max_estimated_tokens"] <= 2048
    assert "chunks" not in chunked
    assert "chunk_ids" not in chunked


def test_large_chunk_plan_returns_bounded_summary(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "many-chunks.txt"
    source.write_text("\n".join(["x" * 8000 for _ in range(1500)]), encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    call(plugin.extract_segments, job_id=created["job_id"])

    chunked = call(plugin.create_chunks, job_id=created["job_id"])

    assert chunked["state"] == "CHUNKED"
    assert chunked["chunk_count"] == 1500
    assert chunked["max_estimated_tokens"] <= 2048
    assert "chunks" not in chunked
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "CHUNKED"


def test_many_short_segments_keep_every_claim_bounded(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "short-lines.txt"
    source.write_text("\n".join(["x" for _ in range(2500)]), encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    call(plugin.extract_segments, job_id=created["job_id"])
    chunked = call(plugin.create_chunks, job_id=created["job_id"])
    claimed = 0

    while True:
        lease = call(plugin.claim_chunk, job_id=created["job_id"], worker_id="bounded-worker")
        assert lease["ok"] is True
        call(
            plugin.submit_chunk,
            job_id=created["job_id"],
            chunk_id=lease["chunk_id"],
            fencing_token=lease["fencing_token"],
            translations=[{"segment_id": s["segment_id"], "text": s["source_text"]} for s in lease["segments"]],
        )
        validation = call(plugin.validate_chunk, job_id=created["job_id"], chunk_id=lease["chunk_id"])
        claimed += 1
        if validation["state"] == "VALIDATING":
            break

    assert claimed == chunked["chunk_count"]


def test_oversized_single_segment_fails_closed(worker, tmp_path):
    plugin, home = worker
    source = tmp_path / "source.txt"
    source.write_text("x" * 9000, encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])
    call(plugin.extract_segments, job_id=created["job_id"])

    rejected = call(plugin.create_chunks, job_id=created["job_id"])

    assert rejected["error"]["code"] == "OVERSIZED_SEGMENT"
    status = call(plugin.get_job_status, job_id=created["job_id"])
    assert status["state"] == "FAILED"
    assert status["verification_receipt"] is None
    db_path = home / "plugins" / "localization-worker" / "localization-worker.sqlite3"
    with sqlite3.connect(db_path) as conn:
        events = conn.execute(
            "SELECT event FROM audit_events WHERE job_id=? ORDER BY seq",
            (created["job_id"],),
        ).fetchall()
    assert ("OVERSIZED_SEGMENT",) in events


def test_empty_document_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.md"
    source.write_text("", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    call(plugin.inspect_job, job_id=created["job_id"])

    rejected = call(plugin.extract_segments, job_id=created["job_id"])

    assert rejected["error"]["code"] == "EMPTY_DOCUMENT"


def test_register_fails_closed_without_posix_path_primitives(worker, monkeypatch):
    plugin, _ = worker

    class Context:
        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            raise AssertionError("tools must not register without security primitives")

    monkeypatch.delattr(plugin.os, "O_NOFOLLOW")

    with pytest.raises(RuntimeError, match="UNSUPPORTED_PLATFORM_SECURITY_PRIMITIVES"):
        plugin.register(Context())


def test_registered_tool_schemas_are_strict(worker):
    plugin, _ = worker
    registrations = []

    class Context:
        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            registrations.append(kwargs)

    plugin.register(Context())

    assert len(registrations) == 11
    for registration in registrations:
        parameters = registration["schema"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert parameters["required"]
        assert set(parameters["required"]).issubset(parameters["properties"])


def test_empty_translation_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": ""}],
    )

    assert rejected["error"]["code"] == "EMPTY_TRANSLATION"


@pytest.mark.parametrize("translated", ["안녕\n세계", "안녕\r세계", "안녕\r\n세계"])
def test_embedded_translation_newline_is_rejected_at_submission(worker, tmp_path, translated):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": translated}],
    )

    assert rejected["error"]["code"] == "TRANSLATION_CONTAINS_NEWLINE"


def test_number_change_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Version 2.5 costs 100 USD")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "버전 2.6의 가격은 100 USD"}],
    )

    assert rejected["error"]["code"] == "NUMBER_MISMATCH"


def test_changed_source_content_creates_a_new_job(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("first", encoding="utf-8")
    first = call(plugin.create_job, source_path=str(source), target_locale="ko")
    source.write_text("second", encoding="utf-8")

    second = call(plugin.create_job, source_path=str(source), target_locale="ko")

    assert second["job_id"] != first["job_id"]
    assert second["idempotent"] is False


def test_source_change_before_assembly_fails_closed(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    (tmp_path / "source.txt").write_text("changed after validation", encoding="utf-8")

    rejected = call(plugin.assemble_output, job_id=job_id)

    assert rejected["error"]["code"] == "SOURCE_CHANGED"
    assert call(plugin.get_job_status, job_id=job_id)["state"] == "FAILED"
    assert not Path(lease["output_path"]).exists()


def test_existing_database_schema_is_migrated_without_data_loss(worker):
    plugin, home = worker
    data_dir = home / "plugins" / "localization-worker"
    db_path = data_dir / "localization-worker.sqlite3"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jobs(id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,"
            "source_path TEXT NOT NULL,output_path TEXT NOT NULL,target_locale TEXT NOT NULL,"
            "state TEXT NOT NULL,source_hash TEXT,bom INTEGER,newline TEXT,final_newline INTEGER,"
            "verification_receipt TEXT,created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE chunks(job_id TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,"
            "fencing_token TEXT,worker_id TEXT,submission_hash TEXT,PRIMARY KEY(job_id,id))"
        )
        conn.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("legacy", "key", "/source.txt", "/out.txt", "ko", "CREATED", None, None, None, None, None, "now"),
        )
        conn.execute("PRAGMA user_version=0")

    conn = plugin._connect()
    try:
        job_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        legacy = conn.execute("SELECT id,state FROM jobs WHERE id='legacy'").fetchone()
    finally:
        conn.close()

    assert "output_hash" in job_columns
    assert "lease_expires_at" in chunk_columns
    assert version >= 1
    assert tuple(legacy) == ("legacy", "CREATED")


def test_create_job_rejects_source_outside_configured_roots(worker, tmp_path):
    plugin, _ = worker
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        result = call(plugin.create_job, source_path=str(outside), target_locale="ko")
    finally:
        outside.unlink(missing_ok=True)

    assert result["error"]["code"] == "SOURCE_OUTSIDE_ALLOWED_ROOTS"


def test_create_job_rejects_symlink_escape(worker, tmp_path):
    plugin, _ = worker
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
        result = call(plugin.create_job, source_path=str(link), target_locale="ko")
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)

    assert result["error"]["code"] == "SOURCE_OUTSIDE_ALLOWED_ROOTS"


def test_register_uses_profile_scoped_default_source_root(worker):
    plugin, home = worker

    class Context:
        def __init__(self):
            self.tools = []

        def get_config(self, key, default=None):
            assert key == "source_roots"
            return default

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

    context = Context()
    plugin.register(context)

    assert plugin._SOURCE_ROOTS == ((home / "localization-input").resolve(),)
    assert len(context.tools) == 11


def test_source_replaced_by_symlink_before_inspection_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    source.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    source.unlink()
    try:
        source.symlink_to(outside)
        rejected = call(plugin.inspect_job, job_id=created["job_id"])
    finally:
        source.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)

    assert rejected["error"]["code"] == "SOURCE_OUTSIDE_ALLOWED_ROOTS"


def test_aborted_job_rejects_submit_from_existing_lease(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(plugin.abort_job, job_id=job_id, reason="stop")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )

    assert rejected["error"]["code"] == "INVALID_TRANSITION"
    assert call(plugin.get_job_status, job_id=job_id)["state"] == "ABORTED"


def test_missing_assembled_output_moves_job_to_review(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    Path(assembled["output_path"]).unlink()

    rejected = call(plugin.verify_output, job_id=job_id)
    status = call(plugin.get_job_status, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_MISSING"
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_oversized_translation_payload_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "가" * 30_000}],
    )

    assert rejected["error"]["code"] == "SUBMISSION_TOO_LARGE"


def test_invalid_utf8_output_moves_job_to_review(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    Path(assembled["output_path"]).write_bytes(b"\xff\xfe")

    rejected = call(plugin.verify_output, job_id=job_id)
    status = call(plugin.get_job_status, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_NOT_UTF8"
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_source_change_between_create_and_inspect_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("version A", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    source.write_text("version B", encoding="utf-8")

    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "SOURCE_CHANGED"
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "FAILED"


def test_identical_replay_with_wrong_fencing_token_is_rejected(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    translations = [{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}]
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=translations,
    )

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token="wrong-token",
        translations=translations,
    )

    assert rejected["error"]["code"] == "STALE_LEASE"


def test_assembly_rejects_output_directory_symlink_escape(worker, tmp_path):
    plugin, home = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    job_dir = Path(lease["output_path"]).parent
    outside = tmp_path / "outside-output"
    outside.mkdir()
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    job_dir.symlink_to(outside, target_is_directory=True)

    rejected = call(plugin.assemble_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    assert list(outside.iterdir()) == []
    assert call(plugin.get_job_status, job_id=job_id)["state"] == "NEEDS_REVIEW"
    assert Path(lease["output_path"]).is_relative_to(home)


def test_blank_line_round_trip_completes(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("a\n\nb\n", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    job_id = created["job_id"]
    call(plugin.inspect_job, job_id=job_id)
    call(plugin.extract_segments, job_id=job_id)
    call(plugin.create_chunks, job_id=job_id)
    lease = call(plugin.claim_chunk, job_id=job_id, worker_id="worker")
    translations = [
        {"segment_id": segment["segment_id"], "text": {"a": "A", "": "", "b": "B"}[segment["source_text"]]}
        for segment in lease["segments"]
    ]
    call(plugin.submit_chunk, job_id=job_id, chunk_id=lease["chunk_id"], fencing_token=lease["fencing_token"], translations=translations)
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    verified = call(plugin.verify_output, job_id=job_id)

    assert verified["state"] == "COMPLETED"
    assert Path(assembled["output_path"]).read_text(encoding="utf-8") == "A\n\nB\n"


def test_bare_cr_newlines_are_rejected(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_bytes(b"a\rb\r")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "UNSUPPORTED_NEWLINE_STYLE"


def test_mixed_newlines_are_rejected(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_bytes(b"a\r\nb\nc\r\n")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    rejected = call(plugin.inspect_job, job_id=created["job_id"])

    assert rejected["error"]["code"] == "UNSUPPORTED_NEWLINE_STYLE"


def test_number_followed_by_korean_particle_is_preserved(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Version 2.5 costs 100 USD")

    accepted = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "버전 2.5의 가격은 100 USD"}],
    )

    assert accepted["accepted"] is True


def test_malformed_translation_item_returns_bounded_json(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")

    rejected = call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=["not-an-object"],
    )

    assert rejected["error"]["code"] == "INVALID_ARGUMENTS"


def test_sqlite_failure_returns_bounded_json(worker, monkeypatch):
    plugin, _ = worker

    def broken_connect():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(plugin, "_connect", broken_connect)
    rejected = call(plugin.get_job_status, job_id="job")

    assert rejected["error"]["code"] == "DATABASE_ERROR"
    assert "locked" not in rejected["error"]["message"]


def test_oversized_worker_id_is_rejected(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    job_id = created["job_id"]
    call(plugin.inspect_job, job_id=job_id)
    call(plugin.extract_segments, job_id=job_id)
    call(plugin.create_chunks, job_id=job_id)

    rejected = call(plugin.claim_chunk, job_id=job_id, worker_id="w" * 300)

    assert rejected["error"]["code"] == "INVALID_WORKER_ID"


def test_oversized_abort_reason_is_rejected(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")

    rejected = call(plugin.abort_job, job_id=created["job_id"], reason="r" * 5000)

    assert rejected["error"]["code"] == "ABORT_REASON_TOO_LARGE"
    assert call(plugin.get_job_status, job_id=created["job_id"])["state"] == "CREATED"


@pytest.mark.parametrize("failure_point", ["file_fsync", "rename", "directory_fsync"])
def test_assembly_atomic_failure_points_move_job_to_review(worker, tmp_path, monkeypatch, failure_point):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    output = Path(lease["output_path"])
    original_fsync = plugin.os.fsync
    original_rename = plugin.os.rename
    fsync_calls = 0

    def failing_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        if (failure_point == "file_fsync" and fsync_calls == 1) or (
            failure_point == "directory_fsync" and fsync_calls == 2
        ):
            raise OSError("injected fsync failure")
        return original_fsync(fd)

    def failing_rename(*args, **kwargs):
        if failure_point == "rename":
            raise OSError("injected rename failure")
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(plugin.os, "fsync", failing_fsync)
    monkeypatch.setattr(plugin.os, "rename", failing_rename)
    rejected = call(plugin.assemble_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_WRITE_FAILED"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None
    if output.parent.exists() and not output.parent.is_symlink():
        assert not any(path.name.endswith(".tmp") for path in output.parent.iterdir())


def test_verify_output_read_failure_moves_job_to_review(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    call(plugin.assemble_output, job_id=job_id)
    original_read = plugin.os.read
    output = Path(lease["output_path"])
    output_identity = (output.stat().st_dev, output.stat().st_ino)

    def failing_read(fd, size):
        identity = (plugin.os.fstat(fd).st_dev, plugin.os.fstat(fd).st_ino)
        if identity == output_identity:
            raise OSError("injected read failure")
        return original_read(fd, size)

    monkeypatch.setattr(plugin.os, "read", failing_read)
    rejected = call(plugin.verify_output, job_id=job_id)
    monkeypatch.setattr(plugin.os, "read", original_read)

    assert rejected["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_assembly_write_failure_moves_job_to_review(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])

    def fail_write(_fd, _data):
        raise OSError("injected write failure")

    monkeypatch.setattr(plugin.os, "write", fail_write)
    rejected = call(plugin.assemble_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_WRITE_FAILED"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_output_dir_open_failure_closes_descriptors(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    original_close = plugin.os.close
    opened: set[int] = set()
    closed: set[int] = set()
    original_open = plugin.os.open

    def track_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "jobs":
            raise OSError("injected jobs open failure")
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.add(fd)
        return fd

    def track_close(fd):
        closed.add(fd)
        return original_close(fd)

    monkeypatch.setattr(plugin.os, "open", track_open)
    monkeypatch.setattr(plugin.os, "close", track_close)
    rejected = call(plugin.assemble_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    assert opened <= closed


def test_assembly_resists_profile_plugins_ancestor_swap(worker, tmp_path, monkeypatch):
    plugin, home = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    plugins_root = home / "plugins"
    data_root = plugins_root / "localization-worker"
    outside_plugins = tmp_path / "outside-plugins"
    (outside_plugins / "localization-worker" / "jobs" / job_id).mkdir(parents=True)
    original_open = plugin.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        hits_current_absolute_open = Path(path) == data_root
        hits_component_open = path == "plugins" and dir_fd is not None
        if not swapped and (hits_current_absolute_open or hits_component_open):
            swapped = True
            plugins_root.rename(home / "plugins-original")
            plugins_root.symlink_to(outside_plugins, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin.os, "open", swapping_open)
    result = call(plugin.assemble_output, job_id=job_id)

    assert list((outside_plugins / "localization-worker" / "jobs" / job_id).iterdir()) == []
    assert result.get("state") == "ASSEMBLING" or result["error"]["code"] == "OUTPUT_PATH_UNSAFE"


def test_verify_rejects_fifo_without_reading(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    output = Path(assembled["output_path"])
    output.unlink()
    os.mkfifo(output)
    result: list[dict] = []
    thread = threading.Thread(
        target=lambda: result.append(call(plugin.verify_output, job_id=job_id)),
        daemon=True,
    )

    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive(), "output FIFO open blocked"
    assert result[0]["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_assembly_dirfd_resists_ancestor_symlink_swap(worker, tmp_path, monkeypatch):
    plugin, home = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    jobs_root = home / "plugins" / "localization-worker" / "jobs"
    outside_jobs = tmp_path / "outside-jobs"
    (outside_jobs / job_id).mkdir(parents=True)
    original_open = plugin.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "jobs" and dir_fd is not None and not swapped:
            swapped = True
            jobs_root.rename(jobs_root.with_name("jobs-original"))
            jobs_root.symlink_to(outside_jobs, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin.os, "open", swapping_open)
    result = call(plugin.assemble_output, job_id=job_id)

    assert list((outside_jobs / job_id).iterdir()) == []
    assert result.get("state") == "ASSEMBLING" or result["error"]["code"] == "OUTPUT_PATH_UNSAFE"


def test_assembly_dirfd_resists_parent_symlink_swap(worker, tmp_path, monkeypatch):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    output = Path(lease["output_path"])
    outside = tmp_path / "outside-race"
    outside.mkdir()
    original_open = plugin.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is not None and isinstance(path, str) and path.startswith(".") and not swapped:
            swapped = True
            output.parent.rename(output.parent.with_name(output.parent.name + "-original"))
            output.parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(plugin.os, "open", swapping_open)
    result = call(plugin.assemble_output, job_id=job_id)

    assert list(outside.iterdir()) == []
    assert result.get("state") == "ASSEMBLING" or result["error"]["code"] == "OUTPUT_PATH_UNSAFE"


def test_verify_unsafe_output_path_moves_job_to_review(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    output = Path(assembled["output_path"])
    outside = tmp_path / "outside.txt"
    outside.write_text("external secret", encoding="utf-8")
    output.unlink()
    output.symlink_to(outside)

    rejected = call(plugin.verify_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_crlf_output_changed_to_lf_fails_verification(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.txt"
    source.write_bytes(b"A\r\nB\r\n")
    created = call(plugin.create_job, source_path=str(source), target_locale="ko")
    job_id = created["job_id"]
    call(plugin.inspect_job, job_id=job_id)
    call(plugin.extract_segments, job_id=job_id)
    call(plugin.create_chunks, job_id=job_id)
    lease = call(plugin.claim_chunk, job_id=job_id, worker_id="worker")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[
            {"segment_id": segment["segment_id"], "text": segment["source_text"]}
            for segment in lease["segments"]
        ],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    Path(assembled["output_path"]).write_bytes(b"A\nB\n")

    rejected = call(plugin.verify_output, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_REPARSE_MISMATCH"
    status = call(plugin.get_job_status, job_id=job_id)
    assert status["state"] == "NEEDS_REVIEW"
    assert status["verification_receipt"] is None


def test_completed_output_replaced_by_symlink_is_rejected_without_reading_target(worker, tmp_path):
    plugin, _ = worker
    job_id, lease = claimed_job(plugin, tmp_path, text="Hello")
    call(
        plugin.submit_chunk,
        job_id=job_id,
        chunk_id=lease["chunk_id"],
        fencing_token=lease["fencing_token"],
        translations=[{"segment_id": lease["segments"][0]["segment_id"], "text": "안녕"}],
    )
    call(plugin.validate_chunk, job_id=job_id, chunk_id=lease["chunk_id"])
    assembled = call(plugin.assemble_output, job_id=job_id)
    call(plugin.verify_output, job_id=job_id)
    output = Path(assembled["output_path"])
    outside = tmp_path / "outside.txt"
    outside.write_text("external secret", encoding="utf-8")
    output.unlink()
    output.symlink_to(outside)

    rejected = call(plugin.get_job_status, job_id=job_id)

    assert rejected["error"]["code"] == "OUTPUT_PATH_UNSAFE"
    assert outside.read_text(encoding="utf-8") == "external secret"


def test_unsupported_format_fails_closed(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    rejected = call(plugin.create_job, source_path=str(source), target_locale="fr")

    assert rejected["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_abort_is_terminal_and_audited(worker, tmp_path):
    plugin, _ = worker
    source = tmp_path / "source.md"
    source.write_text("Hello", encoding="utf-8")
    job = call(plugin.create_job, source_path=str(source), target_locale="fr")

    aborted = call(plugin.abort_job, job_id=job["job_id"], reason="operator request")
    rejected = call(plugin.inspect_job, job_id=job["job_id"])
    status = call(plugin.get_job_status, job_id=job["job_id"])

    assert aborted["state"] == "ABORTED"
    assert rejected["error"]["code"] == "INVALID_TRANSITION"
    assert status["state"] == "ABORTED"
    assert status["audit_event_count"] == 2


def test_plugin_registers_only_declared_bounded_json_tools(worker):
    plugin, _ = worker
    registrations = []

    class Context:
        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            registrations.append(kwargs)

    plugin.register(Context())

    assert {item["name"] for item in registrations} == {
        "localization_create_job", "localization_inspect_job", "localization_extract_segments",
        "localization_create_chunks", "localization_claim_chunk", "localization_submit_chunk",
        "localization_validate_chunk", "localization_assemble_output", "localization_verify_output",
        "localization_get_job_status", "localization_abort_job",
    }
    assert {item["toolset"] for item in registrations} == {"localization_worker"}
