"""``get_read_block_error`` — the published plugin path check — refuses the device and /proc
files Hermes's own read_file refuses, so a plugin that follows the docs cannot read process
secrets (environment, argv, memory layout) or hang on an endless device."""

from __future__ import annotations

import os

import pytest

from agent.file_safety import get_read_block_error


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("path", [
    "/proc/self/environ", "/proc/1/environ", "/proc/self/cmdline", "/proc/self/task/1/maps",
    "/proc/self/mem", "/dev/zero", "/dev/urandom", "/dev/stdin", "/dev/../dev/zero",
])
def test_device_and_proc_reads_are_refused(path):
    err = get_read_block_error(path)
    assert err and "device or process file" in err


@pytest.mark.platforms("linux")
def test_symlink_to_proc_environ_is_refused(tmp_path):
    link = tmp_path / "innocent.txt"
    os.symlink("/proc/self/environ", link)
    assert get_read_block_error(str(link))


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("path", ["/dev/null", "/proc/cpuinfo", "/proc/self/status"])
def test_harmless_proc_and_dev_paths_stay_readable(path):
    assert get_read_block_error(path) is None


def test_ordinary_file_is_not_a_device(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hi")
    assert get_read_block_error(str(target)) is None
