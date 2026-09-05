import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_register_digest_reaction_resolves_single_source_response(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import (
        format_digest_detail_response,
        register_digest_delivery,
        resolve_digest_delivery,
    )

    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    source_output = source_dir / "2026-06-28_08-00-00.md"
    source_output.write_text(
        "# Cron Job: Source Job\n\n"
        "## Prompt\n"
        "internal collection prompt that should not be sent\n\n"
        "## Response\n"
        "**⚠️ Source finding**\n\n"
        "📌 **Befund**\n"
        "- actionable detail\n",
        encoding="utf-8",
    )

    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-job"],
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
        source_names={"source-job": "Source Job"},
        now=1000.0,
    )

    record = resolve_digest_delivery("!room:example.org", "$digest", now=1001.0)
    assert record is not None
    assert record["digest_job_id"] == "digest-job"
    assert record["sources"][0]["job_id"] == "source-job"

    text = format_digest_detail_response(record, source_index=0)
    assert "**🧾 Einzelbericht: Source Job**" in text
    assert "actionable detail" in text
    assert "internal collection prompt" not in text


def test_prompt_embedded_response_heading_does_not_leak_prompt_content(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response

    source = tmp_path / "cron" / "output" / "source-job" / "detail.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Cron Job: Source Job\n\n"
        "## Prompt\n\n"
        "Summarize this untrusted input:\n"
        "## Response\n"
        "decoy heading inside the prompt\n"
        "## Response\n"
        "PRIVATE PROMPT DATA\n\n"
        "## Response\n\n"
        "public final response\n",
        encoding="utf-8",
    )
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source),
            }
        ]
    }

    text = format_digest_detail_response(record)

    assert "public final response" in text
    assert "PRIVATE PROMPT DATA" not in text


def test_digest_source_selection_tolerates_malformed_sources():
    from cron.digest_reactions import format_digest_source_selection

    assert format_digest_source_selection({"sources": None}) == (
        "**🧾 Mehrere Einzelberichte verfügbar**\n\nWähle per Reaction:"
    )


def test_digest_registration_pins_the_source_artifact_used_in_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from cron.scheduler_prompt import _inject_context_from
    from gateway.config import Platform

    source_job_id = "a" * 32
    source_dir = tmp_path / "cron" / "output" / source_job_id
    source_dir.mkdir(parents=True)
    summarized = source_dir / "2026-06-28_08-00-00.md"
    summarized.write_text("## Response\nsummarized run", encoding="utf-8")
    os.utime(summarized, (1000, 1000))
    job = {
        "id": "b" * 32,
        "name": "Morning Digest",
        "context_from": [source_job_id],
    }

    prompt, injected = _inject_context_from(job, "Summarize")
    assert injected is True
    assert "summarized run" in prompt

    newer = source_dir / "2026-06-28_08-01-00.md"
    newer.write_text("## Response\nnewer unrelated run", encoding="utf-8")
    os.utime(newer, (2000, 2000))
    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=SimpleNamespace(success=True, message_id="$digest"),
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert record["sources"][0]["output_path"] == str(summarized)


def test_digest_registration_pins_self_context_to_the_actual_job_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from cron.scheduler_prompt import _inject_context_from
    from gateway.config import Platform

    job_id = "a" * 32
    source_dir = tmp_path / "cron" / "output" / job_id
    source_dir.mkdir(parents=True)
    summarized = source_dir / "2026-06-28_08-00-00.md"
    summarized.write_text("## Response\nprevious self run", encoding="utf-8")
    job = {
        "id": job_id,
        "name": "Continuing Digest",
        "context_from": ["self"],
    }

    prompt, injected = _inject_context_from(job, "Continue")
    assert injected is True
    assert "previous self run" in prompt
    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=SimpleNamespace(success=True, message_id="$digest"),
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert record["sources"][0]["job_id"] == job_id
    assert record["sources"][0]["output_path"] == str(summarized)


def test_digest_registration_excludes_sources_not_used_in_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from cron.scheduler_prompt import _inject_context_from
    from gateway.config import Platform

    source_job_id = "a" * 32
    source_dir = tmp_path / "cron" / "output" / source_job_id
    source_dir.mkdir(parents=True)
    empty = source_dir / "2026-06-28_08-00-00.md"
    empty.write_text("", encoding="utf-8")
    job = {
        "id": "b" * 32,
        "name": "Morning Digest",
        "context_from": [source_job_id],
    }

    prompt, injected = _inject_context_from(job, "Summarize")
    assert injected is False
    assert prompt == "Summarize"

    (source_dir / "2026-06-28_08-01-00.md").write_text(
        "## Response\nnewer unrelated run", encoding="utf-8"
    )
    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=SimpleNamespace(success=True, message_id="$digest"),
    )

    assert resolve_digest_delivery("!room:example.org", "$digest") is None


def test_missing_pinned_source_does_not_fall_forward_to_a_newer_run(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response

    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    summarized = source_dir / "summarized.md"
    summarized.write_text("## Response\nsummarized run", encoding="utf-8")
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(summarized),
            }
        ]
    }
    summarized.unlink()
    (source_dir / "newer.md").write_text(
        "## Response\nnewer unrelated run", encoding="utf-8"
    )

    text = format_digest_detail_response(record)

    assert "detail output is no longer available" in text
    assert "newer unrelated run" not in text


def test_source_reader_never_requests_an_unbounded_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron import digest_reactions

    source = tmp_path / "cron" / "output" / "source-job" / "detail.md"
    source.parent.mkdir(parents=True)
    source.write_text("## Prompt\nprivate\n\n## Response\nsafe detail", encoding="utf-8")
    real_fdopen = digest_reactions.os.fdopen

    class BoundedHandle:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def read(self, size=-1):
            assert size >= 0, "artifact reads must always be bounded"
            return self._inner.read(size)

        def readline(self, size=-1):
            assert size > 0, "artifact line reads must always be bounded"
            return self._inner.readline(size)

    monkeypatch.setattr(
        digest_reactions.os,
        "fdopen",
        lambda *args, **kwargs: BoundedHandle(real_fdopen(*args, **kwargs)),
    )
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source),
            }
        ]
    }

    text = digest_reactions.format_digest_detail_response(record)

    assert "safe detail" in text
    assert "private" not in text


def test_source_reader_fails_closed_after_bounded_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron import digest_reactions

    source = tmp_path / "cron" / "output" / "source-job" / "detail.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "x" * (1024 * 1024 + 1) + "\n## Response\nlate detail",
        encoding="utf-8",
    )
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source),
            }
        ]
    }

    text = digest_reactions.format_digest_detail_response(record)

    assert "detail output is no longer available" in text
    assert "late detail" not in text


def test_response_scanner_never_reads_past_its_total_budget():
    from cron import digest_reactions

    class EndlessLine:
        def __init__(self):
            self.total = 0

        def readline(self, size=-1):
            assert size > 0
            self.total += size
            return "x" * size

    handle = EndlessLine()

    assert digest_reactions._read_response_section_bounded(handle) == ""
    assert handle.total == digest_reactions._MAX_ARTIFACT_SCAN_CHARS


def test_register_digest_reaction_ignores_unsafe_source_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response

    record = {
        "room_id": "!room:example.org",
        "event_id": "$digest",
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(tmp_path / ".." / "outside.md"),
            }
        ],
    }

    text = format_digest_detail_response(record, source_index=0)
    assert "detail output is no longer available" in text
    assert "outside.md" not in text


def test_latest_output_fallback_ignores_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import (
        format_digest_detail_response,
        register_digest_delivery,
        resolve_digest_delivery,
    )

    outside = tmp_path / "outside.md"
    outside.write_text("## Response\nSECRET OUTSIDE DETAIL", encoding="utf-8")
    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    (source_dir / "latest.md").symlink_to(outside)

    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-job"],
        source_names={"source-job": "Source Job"},
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert record["sources"][0]["output_path"] == ""
    text = format_digest_detail_response(record, source_index=0)
    assert "SECRET OUTSIDE DETAIL" not in text
    assert "detail output is no longer available" in text


def test_source_path_swap_after_validation_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron import digest_reactions

    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    source_output = source_dir / "detail.md"
    source_output.write_text("## Response\nSAFE DETAIL", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("## Response\nSECRET OUTSIDE DETAIL", encoding="utf-8")
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source_output),
            }
        ]
    }
    real_safe_output_path = digest_reactions._safe_output_path
    validation_calls = 0

    def swap_after_first_validation(path):
        nonlocal validation_calls
        safe_path = real_safe_output_path(path)
        validation_calls += 1
        if validation_calls == 1:
            source_output.unlink()
            source_output.symlink_to(outside)
        return safe_path

    monkeypatch.setattr(
        digest_reactions,
        "_safe_output_path",
        swap_after_first_validation,
    )

    text = digest_reactions.format_digest_detail_response(record)

    assert validation_calls >= 1
    assert "SECRET OUTSIDE DETAIL" not in text
    assert "detail output is no longer available" in text


def test_source_fifo_swap_after_validation_does_not_block(tmp_path):
    if not hasattr(os, "mkfifo"):
        return

    worker = r"""
import os
import sys
from pathlib import Path

home = Path(sys.argv[1])
os.environ["HERMES_HOME"] = str(home)

from cron import digest_reactions

source = home / "cron" / "output" / "source-job" / "detail.md"
source.parent.mkdir(parents=True)
source.write_text("## Response\nSAFE DETAIL", encoding="utf-8")
record = {
    "sources": [
        {
            "job_id": "source-job",
            "name": "Source Job",
            "output_path": str(source),
        }
    ]
}
real_safe_output_path = digest_reactions._safe_output_path
calls = 0


def swap_to_fifo_after_first_validation(path):
    global calls
    safe_path = real_safe_output_path(path)
    calls += 1
    if calls == 1:
        source.unlink()
        os.mkfifo(source)
    return safe_path


digest_reactions._safe_output_path = swap_to_fifo_after_first_validation
text = digest_reactions.format_digest_detail_response(record)
assert "detail output is no longer available" in text
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    result = subprocess.run(
        [sys.executable, "-c", worker, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing_flag", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_safe_output_read_fails_closed_without_required_open_flag(
    tmp_path, monkeypatch, missing_flag
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron import digest_reactions

    source = tmp_path / "cron" / "output" / "source-job" / "detail.md"
    source.parent.mkdir(parents=True)
    source.write_text("## Response\nSAFE DETAIL", encoding="utf-8")
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source),
            }
        ]
    }
    monkeypatch.delattr(digest_reactions.os, missing_flag, raising=False)

    def output_opened(*_args, **_kwargs):
        raise AssertionError("unsafe platform fallback must not open artifact files")

    monkeypatch.setattr(digest_reactions.os, "open", output_opened)

    text = digest_reactions.format_digest_detail_response(record)

    assert "detail output is no longer available" in text
    assert "Source Job" in text


def test_detail_without_response_section_never_returns_whole_artifact(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import format_digest_detail_response

    source_dir = tmp_path / "cron" / "output" / "source-job"
    source_dir.mkdir(parents=True)
    source_output = source_dir / "legacy.md"
    source_output.write_text(
        "## Prompt\nPRIVATE PROMPT\n\n## Script Output\nPRIVATE RAW OUTPUT\n",
        encoding="utf-8",
    )
    record = {
        "sources": [
            {
                "job_id": "source-job",
                "name": "Source Job",
                "output_path": str(source_output),
            }
        ]
    }

    text = format_digest_detail_response(record)

    assert "detail output is no longer available" in text
    assert "PRIVATE PROMPT" not in text
    assert "PRIVATE RAW OUTPUT" not in text


def test_scheduler_registers_matrix_digest_metadata_only_after_confirmed_send(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=True, message_id="$digest")
    job = {
        "id": "digest-job",
        "name": "Morning Digest",
        "context_from": ["source-a", "source-b"],
    }

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
    )

    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert [src["job_id"] for src in record["sources"]] == ["source-a", "source-b"]


def test_scheduler_does_not_register_failed_matrix_send_with_event_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import _registry_path
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=False, message_id="$digest")
    job = {"id": "digest-job", "name": "Morning Digest", "context_from": ["source-a"]}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
    )

    assert not _registry_path().exists()


def test_scheduler_does_not_register_filtered_matrix_send_with_event_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import _registry_path
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = {
        "success": True,
        "delivered": False,
        "message_id": "$filtered-digest",
    }
    job = {"id": "digest-job", "name": "Morning Digest", "context_from": ["source-a"]}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "digest-job" / "latest.md",
    )

    assert not _registry_path().exists()


def test_scheduler_does_not_register_non_digest_matrix_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import _registry_path
    from cron.scheduler_delivery import _register_matrix_digest_details_if_applicable
    from gateway.config import Platform

    send_result = SimpleNamespace(success=True, message_id="$normal")
    job = {"id": "normal-job", "name": "Normal Job"}

    _register_matrix_digest_details_if_applicable(
        job=job,
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        send_result=send_result,
        output_file=tmp_path / "cron" / "output" / "normal-job" / "latest.md",
    )

    assert not _registry_path().exists()


def test_deliver_result_registers_standalone_matrix_digest_with_output_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _deliver_result
    from gateway.config import Platform

    digest_output = tmp_path / "cron" / "output" / "digest-job" / "latest.md"
    digest_output.parent.mkdir(parents=True)
    digest_output.write_text("## Response\ndigest", encoding="utf-8")
    pconfig = MagicMock(enabled=True)
    pconfig.extra = {}
    config = MagicMock()
    config.platforms = {Platform.MATRIX: pconfig}
    send = AsyncMock(return_value={"success": True, "message_id": "$digest"})
    job = {
        "id": "digest-job",
        "name": "Morning Digest",
        "deliver": "origin",
        "origin": {"platform": "matrix", "chat_id": "!room:example.org"},
        "context_from": ["source-a"],
    }

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", new=send),
    ):
        error = _deliver_result(job, "digest", output_file=digest_output)

    assert error is None
    record = resolve_digest_delivery("!room:example.org", "$digest")
    assert record is not None
    assert record["digest_output_path"] == str(digest_output)


def test_deliver_result_registers_live_matrix_digest_with_output_file(
    tmp_path, monkeypatch
):
    from concurrent.futures import Future

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from cron.digest_reactions import resolve_digest_delivery
    from cron.scheduler_delivery import _deliver_result
    from gateway.config import Platform

    digest_output = tmp_path / "cron" / "output" / "digest-job" / "live.md"
    digest_output.parent.mkdir(parents=True)
    digest_output.write_text("## Response\ndigest", encoding="utf-8")
    pconfig = MagicMock(enabled=True)
    pconfig.extra = {}
    config = MagicMock()
    config.platforms = {Platform.MATRIX: pconfig}
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="$live-digest")
    )
    adapter.supports_inchannel_continuable = False
    loop = MagicMock()
    loop.is_running.return_value = True
    standalone_send = AsyncMock(return_value={"success": True, "message_id": "$wrong"})
    job = {
        "id": "digest-job",
        "name": "Morning Digest",
        "deliver": "origin",
        "origin": {"platform": "matrix", "chat_id": "!room:example.org"},
        "context_from": ["source-a"],
    }

    def run_coro(coro, _loop):
        future = Future()
        try:
            import asyncio

            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("asyncio.run_coroutine_threadsafe", side_effect=run_coro),
        patch("tools.send_message_tool._send_to_platform", new=standalone_send),
    ):
        error = _deliver_result(
            job,
            "digest",
            adapters={Platform.MATRIX: adapter},
            loop=loop,
            output_file=digest_output,
        )

    assert error is None
    adapter.send.assert_awaited_once()
    standalone_send.assert_not_awaited()
    record = resolve_digest_delivery("!room:example.org", "$live-digest")
    assert record is not None
    assert record["digest_output_path"] == str(digest_output)


def test_concurrent_digest_registrations_preserve_both_processes(tmp_path):
    """The shared registry must serialize its full read-modify-write transaction."""
    worker = r"""
import os
import sys
import time
from pathlib import Path

home = Path(sys.argv[1])
name = sys.argv[2]
os.environ["HERMES_HOME"] = str(home)

from cron import digest_reactions as registry

real_save = registry._save_registry


def slow_save(data):
    time.sleep(0.02)
    real_save(data)


registry._save_registry = slow_save
(home / f"ready-{name}").touch()
go = home / "go"
deadline = time.monotonic() + 10
while not go.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("concurrency test start barrier timed out")
    time.sleep(0.005)

for index in range(8):
    registry.register_digest_delivery(
        room_id="!room:example.org",
        event_id=f"${name}-{index}",
        digest_job={"id": f"digest-{name}", "name": f"Digest {name}"},
        source_job_ids=[f"source-{name}"],
        now=1000.0 + index,
    )
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path), name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in ("a", "b")
    ]
    deadline = time.monotonic() + 10
    while not all((tmp_path / f"ready-{name}").exists() for name in ("a", "b")):
        if time.monotonic() >= deadline:
            for process in processes:
                process.kill()
            raise AssertionError("workers did not reach the start barrier")
        time.sleep(0.005)
    (tmp_path / "go").touch()

    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        if process.returncode:
            failures.append((process.returncode, stdout, stderr))
    assert failures == []

    registry = json.loads(
        (tmp_path / "state" / "matrix-digest-reactions.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        f"!room:example.org\0${name}-{index}"
        for name in ("a", "b")
        for index in range(8)
    }
    assert set(registry) == expected
