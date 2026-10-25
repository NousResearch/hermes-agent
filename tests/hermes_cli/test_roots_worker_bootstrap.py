"""Behavior contracts for the ESN-0030 managed Roots worker bootstrap."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import hermes_cli.roots_worker_bootstrap as bootstrap
from hermes_cli.roots_worker_bootstrap import (
    BootstrapRefusal,
    build_receipt,
    read_inherited_token,
    validate_assignment,
    verify_proofs,
    write_new_receipt,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    now = int(time.time())
    work = tmp_path / "work"
    work.mkdir()
    provider_profile = tmp_path / "provider-profile"
    provider_profile.mkdir()
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("bounded worker\n")
    instruction.chmod(0o600)
    executable = str(Path("/bin/echo").resolve())
    assignment = {
        "schema": "hermes.roots-worker-assignment/v1",
        "principal": "agent:runner",
        "role": "runner",
        "owner_identity": "operator@example.test",
        "capabilities": ["read_main"],
        "partitions": ["main"],
        "authority_scope": "assigned-ticket-only",
        "ticket_ref": "ESN-0030",
        "ticket_updated_at": "2026-10-25T14:46:11+00:00",
        "issued_at": now - 1,
        "expires_at": now + 300,
        "roots_url": "https://roots.example.test",
        "workdir": str(work),
        "instruction_files": [str(instruction)],
        "claude_config_dir": str(provider_profile),
        "claude_executable": executable,
        "roots_mcp_executable": executable,
        "channel_plugin": "plugin:roots@internal",
        "session_id": "11111111-1111-4111-8111-111111111111",
    }
    request = tmp_path / "assignment.json"
    request.write_text(json.dumps(assignment))
    request.chmod(0o600)
    whoami = {
        "principal": "agent:runner", "kind": "agent", "role": "runner",
        "acting_for": "operator@example.test", "capabilities": ["read_main"],
        "partitions": ["main"], "channel_plugins": ["plugin:roots@internal"],
    }
    ticket = {
        "ref": "ESN-0030", "status": "IN PROGRESS",
        "updated_at": "2026-10-25T14:46:11+00:00", "why": "Needed", "definition_of_done": "Done",
        "owner_identity": "operator@example.test", "assigned_principal": "agent:runner",
    }
    return request, assignment, whoami, ticket


def test_exact_grant_and_ticket_produce_credential_free_receipt(tmp_path: Path) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    assignment = validate_assignment(raw)
    verify_proofs(assignment, whoami, ticket)
    raw_bytes = request.read_bytes()
    receipt = build_receipt(
        raw_bytes, assignment, whoami,
        {assignment["instruction_files"][0]: "a" * 64}, launched_pid=None,
        executable_digests={"claude": "b" * 64, "roots_mcp": "c" * 64},
    )
    target = tmp_path / "receipt.json"
    write_new_receipt(target, receipt)
    persisted = json.loads(target.read_text())
    assert persisted["verified"] is True
    assert persisted["ready"] is False
    assert persisted["principal"] == whoami["principal"]
    assert "token" not in target.read_text().lower()
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda who, ticket: who["capabilities"].append("write_ticket"), "wider"),
        (lambda who, ticket: who.update(principal="agent:other"), "principal"),
        (lambda who, ticket: who.update(acting_for="somebody-else"), "acting_for"),
        (lambda who, ticket: who["channel_plugins"].append("plugin:other@internal"), "channel plugin"),
        (lambda who, ticket: ticket.update(status="DONE"), "status"),
        (lambda who, ticket: ticket.update(updated_at="stale"), "updated_at"),
        (lambda who, ticket: ticket.update(assigned_principal="agent:other"), "assigned_principal"),
    ],
)
def test_identity_grant_and_ticket_must_match_exactly(tmp_path: Path, change, message: str) -> None:
    _, raw, whoami, ticket = _fixture(tmp_path)
    assignment = validate_assignment(raw)
    change(whoami, ticket)
    with pytest.raises(BootstrapRefusal, match=message):
        verify_proofs(assignment, whoami, ticket)


def test_expired_assignment_and_plaintext_non_loopback_roots_are_refused(tmp_path: Path) -> None:
    _, raw, _, _ = _fixture(tmp_path)
    raw["expires_at"] = int(time.time()) - 1
    with pytest.raises(BootstrapRefusal, match="not currently valid"):
        validate_assignment(raw)
    raw["expires_at"] = int(time.time()) + 60
    raw["roots_url"] = "http://roots.example.test"
    with pytest.raises(BootstrapRefusal, match="HTTPS"):
        validate_assignment(raw)


def test_token_is_read_once_from_inherited_descriptor_and_closed() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"bounded-token\n")
    os.close(write_fd)
    assert read_inherited_token(read_fd) == "bounded-token"
    with pytest.raises(OSError):
        os.read(read_fd, 1)


def test_executable_probes_roots_over_inherited_token_before_readiness(tmp_path: Path) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            body = whoami if self.path == "/whoami" else ticket
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    raw["roots_url"] = f"http://127.0.0.1:{server.server_port}"
    request.write_text(json.dumps(raw))
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"bounded-token\n")
    os.close(write_fd)
    receipt = tmp_path / "receipt.json"
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.roots_worker_bootstrap",
                "--request", str(request), "--token-fd", str(read_fd),
                "--receipt", str(receipt), "--verify-only",
            ],
            pass_fds=(read_fd,), capture_output=True, text=True, timeout=10,
        )
    finally:
        os.close(read_fd)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == 0, result.stderr
    assert [path for path, _ in seen] == ["/whoami", "/tickets/ESN-0030"]
    assert {auth for _, auth in seen} == {"Bearer bounded-token"}
    persisted = json.loads(receipt.read_text())
    assert persisted["verified"] is True
    assert persisted["ready"] is False
    assert "bounded-token" not in receipt.read_text()


def _token_fd() -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"bounded-token\n")
    os.close(write_fd)
    return read_fd


def _fake_proofs(monkeypatch: pytest.MonkeyPatch, whoami: dict, ticket: dict) -> None:
    monkeypatch.setattr(
        bootstrap, "_get_json",
        lambda url, token: whoami if url.endswith("/whoami") else ticket,
    )


def test_launch_uses_pinned_inodes_and_keeps_token_out_of_claude_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    record = tmp_path / "child.json"
    executable = tmp_path / "claude"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "mcp = args[args.index('--mcp-config') + 1]\n"
        "instructions = args[args.index('--append-system-prompt-file') + 1]\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{"
        "'argv': args, 'cwd': os.getcwd(), 'has_token_env': 'ROOTS_TOKEN' in os.environ, "
        "'config_dir': os.environ.get('CLAUDE_CONFIG_DIR'), "
        "'mcp': pathlib.Path(mcp).read_text(), 'instructions': pathlib.Path(instructions).read_text()}))\n"
    )
    executable.chmod(0o700)
    malicious = tmp_path / "replacement"
    malicious.write_text("#!/bin/sh\nexit 99\n")
    malicious.chmod(0o700)
    raw["claude_executable"] = str(executable)
    raw["roots_mcp_executable"] = str(executable)
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket)
    monkeypatch.setattr(bootstrap, "_stable_directory_path", lambda fd, name: raw["claude_config_dir"])

    original_pin_directory = bootstrap._pin_directory
    moved_workdir = tmp_path / "reviewed-workdir"
    swaps_done = False

    def pin_and_swap(path: str, name: str) -> int:
        nonlocal swaps_done
        fd = original_pin_directory(path, name)
        if name == "workdir" and not swaps_done:
            os.replace(executable, tmp_path / "reviewed-claude")
            os.replace(malicious, executable)
            os.replace(raw["workdir"], moved_workdir)
            Path(raw["workdir"]).mkdir()
            swaps_done = True
        return fd

    monkeypatch.setattr(bootstrap, "_pin_directory", pin_and_swap)
    receipt = tmp_path / "receipt.json"
    bootstrap.run(request, _token_fd(), receipt)
    deadline = time.time() + 3
    while not record.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert record.exists()
    child = json.loads(record.read_text())
    persisted = json.loads(receipt.read_text())
    assert persisted["state"] == "launched"
    assert persisted["ready"] is False
    assert child["has_token_env"] is False
    mcp_env = json.loads(child["mcp"])["mcpServers"]["roots"]["env"]
    assert mcp_env["ROOTS_TOKEN"] == "bounded-token"
    assert mcp_env["ROOTS_URL"] == raw["roots_url"]
    assert "bounded worker" in child["instructions"]
    assert os.path.samefile(child["cwd"], moved_workdir)
    assert not os.path.samefile(child["cwd"], raw["workdir"])
    assert "bounded-token" not in receipt.read_text()


def test_receipt_race_terminates_and_reaps_sigterm_ignoring_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    started = tmp_path / "started.json"
    executable = tmp_path / "claude"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(started)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
        "while True: time.sleep(1)\n"
    )
    executable.chmod(0o700)
    raw["claude_executable"] = str(executable)
    raw["roots_mcp_executable"] = str(executable)
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket)
    monkeypatch.setattr(bootstrap, "_stable_directory_path", lambda fd, name: raw["claude_config_dir"])
    receipt = tmp_path / "receipt.json"
    original_publish = bootstrap._ReceiptPublisher.publish

    def raced_publish(self, value) -> None:
        deadline = time.time() + 3
        while not started.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert started.exists()
        receipt.write_text("raced\n")
        original_publish(self, value)

    monkeypatch.setattr(bootstrap._ReceiptPublisher, "publish", raced_publish)
    with pytest.raises(BootstrapRefusal, match="publish launch receipt"):
        bootstrap.run(request, _token_fd(), receipt)
    pid = json.loads(started.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert receipt.read_text() == "raced\n"


def test_launch_refuses_when_platform_cannot_bind_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    _fake_proofs(monkeypatch, whoami, ticket)
    monkeypatch.setattr(
        bootstrap, "_stable_directory_path",
        lambda fd, name: (_ for _ in ()).throw(BootstrapRefusal("cannot bind claude_config_dir")),
    )
    with pytest.raises(BootstrapRefusal, match="cannot bind claude_config_dir"):
        bootstrap.run(request, _token_fd(), tmp_path / "receipt.json")


def test_staged_digest_covers_exact_bytes_and_detects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"a" * 100_000)
    executable.chmod(0o700)
    pinned = bootstrap._pin_executable(str(executable), "tool")
    stage = tmp_path / "stage"
    stage.mkdir()
    original_read = os.read
    changed = False

    def mutate_during_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(fd, size)
        if fd == pinned.read_fd and chunk and not changed:
            executable.write_bytes(b"b" * 100_000)
            executable.chmod(0o700)
            changed = True
        return chunk

    monkeypatch.setattr(os, "read", mutate_during_read)
    try:
        with pytest.raises(BootstrapRefusal, match="changed while it was staged"):
            bootstrap._stage_executable(pinned, stage, "staged")
    finally:
        pinned.close()


def test_cleanup_kills_descendant_after_group_leader_exits(tmp_path: Path) -> None:
    child_record = tmp_path / "descendant"
    script = tmp_path / "forker.py"
    script.write_text(
        "import os, pathlib, signal, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f" pathlib.Path({str(child_record)!r}).write_text(str(os.getpid()))\n"
        " while True: time.sleep(1)\n"
        "os._exit(0)\n"
    )
    leader = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    deadline = time.time() + 3
    while not child_record.exists() and time.time() < deadline:
        time.sleep(0.01)
    assert child_record.exists()
    leader.wait(timeout=3)
    descendant = int(child_record.read_text())
    bootstrap._stop_and_reap(leader)
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)


