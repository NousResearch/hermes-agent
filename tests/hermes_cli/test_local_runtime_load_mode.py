"""Managed launch preserves direct I/O across engine command-line versions."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import supervisor


@pytest.mark.parametrize("help_text, expected", [
    ("-lm, --load-mode MODE\n  dio: use DirectIO if available", ["--load-mode", "dio"]),
    ("-dio, --direct-io  use DirectIO if available", ["-dio"]),
])
def test_spawn_selects_supported_direct_io_option(tmp_path, monkeypatch, help_text, expected):
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    executable = tmp_path / "engine" / "llama-server"
    monkeypatch.setattr(supervisor, "server_binary", lambda _: executable)
    probes = []

    def run(argv, **kwargs):
        probes.append(argv)
        return subprocess.CompletedProcess(argv, 0, help_text, "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    commands = []
    monkeypatch.setattr(supervisor, "spawn_server", lambda argv, **kwargs: (
        commands.append(argv) or SimpleNamespace(pid=123, poll=lambda: 0), None,
    ))
    sup = supervisor.LlamaServerSupervisor(tmp_path / "engine", tmp_path / "models", port=19001)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
        index = commands[0].index(expected[0])
        assert commands[0][index:index + len(expected)] == expected
        if expected[0] == "--load-mode":
            assert "-dio" not in commands[0]
        assert probes == [[str(executable), "--help"]]
    finally:
        sup.stop()


def test_direct_io_probe_is_reused_for_the_same_engine(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "--load-mode MODE", "")

    monkeypatch.setattr(supervisor.subprocess, "run", run)
    executable = tmp_path / "llama-server"
    assert supervisor._direct_io_args(executable) == ("--load-mode", "dio")
    assert supervisor._direct_io_args(executable) == ("--load-mode", "dio")
    assert len(calls) == 1


def test_failed_engine_probe_can_be_retried(tmp_path, monkeypatch):
    executable = tmp_path / "llama-server"

    def fail(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 15)

    monkeypatch.setattr(supervisor.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor._direct_io_args(executable)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, "--load-mode MODE", ""))
    assert supervisor._direct_io_args(executable) == ("--load-mode", "dio")


def test_engine_without_direct_io_uses_its_supported_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, "--model FILE", ""))
    assert supervisor._direct_io_args(tmp_path / "llama-server") == ()
