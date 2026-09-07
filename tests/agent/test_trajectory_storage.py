"""Tests for lossless trajectory storage formats."""

import builtins
import errno
import gzip
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import agent.trajectory as trajectory_storage
from agent.trajectory import save_trajectory
import trajectory_compressor
from scripts.sample_and_compress import merge_output_to_single_jsonl
from trajectory_compressor import (
    _default_output_path,
    _open_jsonl,
)


def test_default_trajectory_output_is_gzip_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    saved = save_trajectory(
        [{"from": "human", "value": "hello"}], "test-model", True
    )

    assert saved is True
    path = tmp_path / "trajectory_samples.jsonl.gz"
    assert path.is_file()
    assert not (tmp_path / "trajectory_samples.jsonl").exists()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        entry = json.loads(stream.readline())
    assert entry["conversations"] == [{"from": "human", "value": "hello"}]
    assert entry["completed"] is True


def test_explicit_plain_jsonl_path_remains_backward_compatible(tmp_path):
    path = tmp_path / "legacy.jsonl"

    saved = save_trajectory(
        [{"from": "gpt", "value": "done"}], "test-model", True, str(path)
    )

    assert saved is True
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "test-model"


def test_explicit_gzip_jsonl_path_is_supported(tmp_path):
    path = tmp_path / "custom.jsonl.gz"

    saved = save_trajectory(
        [{"from": "tool", "value": "result"}], "test-model", False, str(path)
    )

    assert saved is True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["completed"] is False


def test_pathlike_gzip_filename_is_supported(tmp_path):
    path = Path(tmp_path) / "pathlike.jsonl.gz"

    saved = save_trajectory([], "test-model", True, path)

    assert saved is True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["model"] == "test-model"


def test_trajectory_compressor_reader_accepts_gzip_jsonl(tmp_path):
    path = tmp_path / "trajectories.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write('{"conversations": []}\n')

    with _open_jsonl(path, "rt") as stream:
        assert json.loads(stream.readline())["conversations"] == []


def test_compressed_input_gets_clean_default_output_name(tmp_path):
    input_path = tmp_path / "trajectories.jsonl.gz"

    assert _default_output_path(input_path) == tmp_path / "trajectories_compressed.jsonl"


def test_directory_sampling_reads_gzip_jsonl_after_writing_sample(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "trajectories.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write('{"conversations": []}\n')

    class FakeCompressor:
        def __init__(self, config):
            pass

        def process_directory(self, sampled_dir, output_dir):
            sampled = sampled_dir / source.name
            with trajectory_compressor._open_jsonl(sampled) as stream:
                assert json.loads(stream.readline())["conversations"] == []

    monkeypatch.setattr(trajectory_compressor, "TrajectoryCompressor", FakeCompressor)
    trajectory_compressor.main(
        input=str(input_dir),
        output=str(tmp_path / "output"),
        sample_percent=100,
    )


def test_concurrent_default_writes_remain_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payloads = [
        [{"from": "tool", "value": f"result-{index}-" + ("x" * 100_000)}]
        for index in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda trajectory: save_trajectory(trajectory, "test-model", True),
                payloads,
            )
        )

    with gzip.open(tmp_path / "trajectory_samples.jsonl.gz", "rt", encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream if line.strip()]
    assert len(entries) == len(payloads)


def test_real_process_appends_form_readable_concatenated_gzip_members(tmp_path):
    path = tmp_path / "multiprocess.jsonl.gz"
    code = (
        "import sys;"
        "from agent.trajectory import save_trajectory;"
        "payload=[{'from':'tool','value':sys.argv[2]+'-'+'雪'*100000}];"
        "raise SystemExit(0 if save_trajectory(payload,'test-model',True,sys.argv[1]) else 2)"
    )
    processes = []
    try:
        for index in range(8):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", code, str(path), str(index)],
                    cwd=Path(__file__).resolve().parents[2],
                )
            )
        return_codes = [process.wait(timeout=20) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    assert return_codes == [0] * len(processes)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        standard_entries = [json.loads(line) for line in stream]
    with _open_jsonl(path, "rt") as stream:
        project_entries = [json.loads(line) for line in stream]
    assert {entry["conversations"][0]["value"].split("-", 1)[0] for entry in standard_entries} == {
        str(index) for index in range(8)
    }
    assert project_entries == standard_entries


def test_truncated_final_member_preserves_prefix_then_raises(tmp_path):
    path = tmp_path / "interrupted.jsonl.gz"
    assert save_trajectory([], "first", True, path) is True
    assert save_trajectory([], "second", True, path) is True
    truncated = gzip.compress(b'{"model":"interrupted"}\n')[:-8]
    with path.open("ab") as stream:
        stream.write(truncated)

    for opener in (
        lambda: gzip.open(path, "rt", encoding="utf-8"),
        lambda: _open_jsonl(path, "rt"),
    ):
        with opener() as stream:
            assert json.loads(next(stream))["model"] == "first"
            assert json.loads(next(stream))["model"] == "second"
            assert json.loads(next(stream))["model"] == "interrupted"
            with pytest.raises(EOFError):
                next(stream)


def test_append_rejects_truncated_existing_gzip_without_changing_it(
    tmp_path, caplog
):
    path = tmp_path / "truncated-before-append.jsonl.gz"
    assert save_trajectory([], "valid", True, path) is True
    with path.open("ab") as stream:
        stream.write(gzip.compress(b'{"model":"interrupted"}\n')[:-8])
    before = path.read_bytes()

    with caplog.at_level(logging.INFO, logger="agent.trajectory"):
        saved = save_trajectory([], "must-not-append", True, path)

    assert saved is False
    assert path.read_bytes() == before
    assert "Failed to save trajectory" in caplog.text
    assert "Trajectory saved to" not in caplog.text


def test_append_accepts_and_preserves_valid_existing_gzip_members(tmp_path):
    path = tmp_path / "valid-before-append.jsonl.gz"
    assert save_trajectory([], "first", True, path) is True

    assert save_trajectory([], "second", True, path) is True

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["model"] for line in stream] == ["first", "second"]


def test_permission_failure_is_deterministic_and_not_logged_as_saved(
    tmp_path, monkeypatch, caplog
):
    path = tmp_path / "trajectory.jsonl.gz"
    real_open = builtins.open

    def deny_target(candidate, *args, **kwargs):
        if os.path.abspath(os.fspath(candidate)) == os.path.abspath(path):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_target)
    with caplog.at_level(logging.INFO, logger="agent.trajectory"):
        saved = save_trajectory([], "test-model", True, path)

    assert saved is False
    assert "Failed to save trajectory" in caplog.text
    assert "Trajectory saved to" not in caplog.text
    assert not path.exists()


@pytest.mark.skipif(
    os.name != "posix" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="requires non-root POSIX permission enforcement",
)
def test_read_only_directory_integration_rejects_append(tmp_path):
    directory = tmp_path / "readonly"
    directory.mkdir()
    path = directory / "trajectory.jsonl.gz"
    directory.chmod(0o500)
    try:
        assert save_trajectory([], "test-model", True, path) is False
    finally:
        directory.chmod(0o700)

    assert not path.exists()


def test_partial_gzip_write_rolls_back_and_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "partial.jsonl.gz"
    assert save_trajectory([], "before", True, path) is True
    original = path.read_bytes()
    real_write = trajectory_storage._write_all

    def partial_then_full(stream, data):
        stream.write(data[: max(1, len(data) // 2)])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trajectory_storage, "_write_all", partial_then_full)
    assert save_trajectory([], "failed", True, path) is False
    assert path.read_bytes() == original

    monkeypatch.setattr(trajectory_storage, "_write_all", real_write)
    assert save_trajectory([], "after", True, path) is True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["model"] for line in stream] == ["before", "after"]


def test_fsync_failure_rolls_back_append(tmp_path, monkeypatch):
    path = tmp_path / "fsync.jsonl.gz"
    assert save_trajectory([], "before", True, path) is True
    original = path.read_bytes()
    calls = 0
    real_fsync = os.fsync

    def fail_first_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(28, "No space left on device")
        return real_fsync(fd)

    monkeypatch.setattr(trajectory_storage.os, "fsync", fail_first_fsync)

    assert save_trajectory([], "failed", True, path) is False
    assert path.read_bytes() == original


@pytest.mark.skipif(trajectory_storage.fcntl is None, reason="requires fcntl")
def test_unexpected_os_lock_error_is_reported_without_retrying(
    tmp_path, monkeypatch, caplog
):
    path = tmp_path / "lock-error.jsonl.gz"
    attempts = 0

    def fail_lock(*_args):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(trajectory_storage.fcntl, "flock", fail_lock)
    with caplog.at_level(logging.WARNING, logger="agent.trajectory"):
        saved = save_trajectory([], "test-model", True, path)

    assert saved is False
    assert attempts == 1
    assert "Input/output error" in caplog.text
    assert path.read_bytes() == b""


def _assert_lock_contention_times_out(tmp_path, monkeypatch):
    path = tmp_path / "contended.jsonl.gz"
    code = (
        "import fcntl,sys,time;"
        "f=open(sys.argv[1],'a+b');"
        "fcntl.flock(f.fileno(),fcntl.LOCK_EX);"
        "print('locked',flush=True);"
        "time.sleep(1)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        monkeypatch.setattr(
            trajectory_storage,
            "_TRAJECTORY_LOCK_TIMEOUT_SECONDS",
            0.1,
            raising=False,
        )
        started = time.monotonic()
        saved = save_trajectory([], "test-model", True, path)
        elapsed = time.monotonic() - started
    finally:
        if holder.poll() is None:
            holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)

    assert saved is False
    assert elapsed < 0.75
    assert path.read_bytes() == b""


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_linux_lock_contention_times_out_without_writing(tmp_path, monkeypatch):
    _assert_lock_contention_times_out(tmp_path, monkeypatch)


@pytest.mark.macos_only
@pytest.mark.live_system_guard_bypass
def test_macos_lock_contention_times_out_without_writing(tmp_path, monkeypatch):
    _assert_lock_contention_times_out(tmp_path, monkeypatch)


def test_stale_sidecar_file_does_not_block_append(tmp_path):
    path = tmp_path / "stale.jsonl.gz"
    lock_path = Path(f"{path}.lock")
    lock_path.write_bytes(b"left by terminated process")

    assert save_trajectory([], "test-model", True, path) is True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert json.loads(next(stream))["model"] == "test-model"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_aliases_share_one_append_critical_section(
    tmp_path, monkeypatch, alias_kind
):
    primary = tmp_path / "primary.jsonl.gz"
    alias = tmp_path / "alias.jsonl.gz"
    assert save_trajectory([], "initial", True, primary) is True
    if alias_kind == "symlink":
        alias.symlink_to(primary)
    else:
        os.link(primary, alias)

    real_write_all = trajectory_storage._write_all
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered_early = threading.Event()
    calls_guard = threading.Lock()
    calls = 0

    def controlled_write(stream, payload):
        nonlocal calls
        with calls_guard:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        elif not release_first.is_set():
            second_entered_early.set()
        return real_write_all(stream, payload)

    monkeypatch.setattr(trajectory_storage, "_write_all", controlled_write)
    results = []

    first = threading.Thread(
        target=lambda: results.append(save_trajectory([], "primary", True, primary))
    )
    second = threading.Thread(
        target=lambda: results.append(save_trajectory([], "alias", True, alias))
    )
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    try:
        overlapped = second_entered_early.wait(timeout=0.5)
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert overlapped is False
    assert results == [True, True]
    with gzip.open(primary, "rt", encoding="utf-8") as stream:
        assert {json.loads(line)["model"] for line in stream} == {
            "initial",
            "primary",
            "alias",
        }


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_plain_named_alias_to_gzip_is_rejected_across_processes(
    tmp_path, alias_kind
):
    target = tmp_path / "target.jsonl.gz"
    alias = tmp_path / "alias.jsonl"
    assert save_trajectory([], "initial", True, target) is True
    if alias_kind == "symlink":
        alias.symlink_to(target)
    else:
        os.link(target, alias)
    before = target.read_bytes()
    code = (
        "import sys;"
        "from agent.trajectory import save_trajectory;"
        "saved=save_trajectory([], 'must-not-append', True, sys.argv[1]);"
        "print(saved, flush=True);"
        "raise SystemExit(0 if not saved else 2)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(alias)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert target.read_bytes() == before
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["model"] for line in stream] == ["initial"]


def test_gzip_named_alias_to_plain_is_rejected_without_modifying_stream(tmp_path):
    target = tmp_path / "target.jsonl"
    alias = tmp_path / "alias.jsonl.gz"
    assert save_trajectory([], "initial", True, target) is True
    alias.symlink_to(target)
    before = target.read_bytes()

    assert save_trajectory([], "must-not-append", True, alias) is False

    assert target.read_bytes() == before
    assert json.loads(target.read_text(encoding="utf-8"))["model"] == "initial"


def test_mixed_directory_discovery_is_sorted_and_has_no_duplicates(tmp_path):
    plain = tmp_path / "a.jsonl"
    compressed = tmp_path / "b.jsonl.gz"
    ignored = tmp_path / "c.jsonl.gz.bak"
    plain.write_text("{}\n", encoding="utf-8")
    with gzip.open(compressed, "wt", encoding="utf-8") as stream:
        stream.write("{}\n")
    ignored.write_text("ignored", encoding="utf-8")

    assert trajectory_compressor._discover_jsonl_files(tmp_path) == [
        plain,
        compressed,
    ]


def test_downstream_merge_reads_mixed_inputs_and_writes_explicit_gzip(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "plain.jsonl").write_text(
        '{"model":"plain"}\n', encoding="utf-8"
    )
    with gzip.open(input_dir / "compressed.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write('{"model":"gzip"}\n')
    output = input_dir / "merged.jsonl.gz"

    assert merge_output_to_single_jsonl(input_dir, output) == output

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["model"] for line in stream] == ["gzip", "plain"]


def test_single_file_tool_honors_explicit_gzip_output(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text('{"model":"source","conversations":[]}\n', encoding="utf-8")
    output = tmp_path / "result.jsonl.gz"

    class CopyingCompressor:
        def __init__(self, config):
            pass

        def process_directory(self, input_dir, output_dir):
            output_dir.mkdir(parents=True)
            for input_file in trajectory_compressor._discover_jsonl_files(input_dir):
                with trajectory_compressor._open_jsonl(input_file) as source_stream:
                    with trajectory_compressor._open_jsonl(
                        output_dir / input_file.name, "wt"
                    ) as output_stream:
                        output_stream.write(source_stream.read())

    monkeypatch.setattr(
        trajectory_compressor, "TrajectoryCompressor", CopyingCompressor
    )

    trajectory_compressor.main(
        input=str(source),
        output=str(output),
        config=str(tmp_path / "missing-config.yaml"),
    )

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert json.loads(next(stream))["model"] == "source"
