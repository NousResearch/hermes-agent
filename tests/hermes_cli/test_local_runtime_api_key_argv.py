"""The managed llama-server's endpoint key stays off argv and out of llama-server.log (#137804).

Argv is readable by every local user (``ps``) and the supervisor logs each spawn command, so a
persisted key goes to the server by file. Only a file llama-server would read back as exactly
that key qualifies; anything else keeps the argv form so the endpoint never 401s every request.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import supervisor


def _spawn_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda exe: [])
    spawns = []
    monkeypatch.setattr(supervisor, "spawn_server", lambda argv, **kwargs: (
        spawns.append(argv) or SimpleNamespace(pid=123, poll=lambda: 0), None,
    ))
    sup = supervisor.LlamaServerSupervisor(tmp_path / "engine" / "llama-server", tmp_path / "models",
                                           port=19003, log_path=tmp_path / "logs" / "llama-server.log")
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    try:
        sup._spawn()
    finally:
        sup.stop()
    return sup, spawns[0], (tmp_path / "logs" / "llama-server.log").read_text(encoding="utf-8")


def test_persisted_key_goes_by_file_and_never_reaches_argv_or_log(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path / "runtime")
    sup, argv, log = _spawn_argv(tmp_path, monkeypatch)
    key_file = argv[argv.index("--api-key-file") + 1]
    assert Path(key_file).read_text(encoding="utf-8").strip() == sup.api_key
    assert "--api-key" not in argv
    assert sup.api_key not in argv
    assert sup.api_key not in log


@pytest.mark.parametrize("persisted", [b"\xef\xbb\xbf{key}", b"{key}\r\n", b"  {key}"])
def test_a_file_llama_server_would_misread_falls_back_to_argv_masked_in_log(tmp_path, monkeypatch, persisted):
    root = tmp_path / "runtime"
    root.mkdir()
    key = "k" * 32
    (root / ".api_key").write_bytes(persisted.replace(b"{key}", key.encode()))
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: root)
    sup, argv, log = _spawn_argv(tmp_path, monkeypatch)
    assert sup.api_key == key
    assert argv[argv.index("--api-key") + 1] == key
    assert "--api-key-file" not in argv
    assert key not in log
