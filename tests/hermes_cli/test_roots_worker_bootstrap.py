"""Behavior contracts for the ESN-0030 managed Roots worker bootstrap."""
from __future__ import annotations

import hashlib
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
    BootstrapRefusal, build_receipt, read_inherited_token, validate_assignment,
    verify_proofs, write_new_receipt,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    now = int(time.time())
    work = tmp_path / "work"
    work.mkdir()
    provider_profile = tmp_path / "provider-profile"
    provider_profile.mkdir()
    claim_dir = tmp_path / "claims"
    claim_dir.mkdir()
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("bounded worker\n")
    instruction.chmod(0o600)
    executable = str(Path("/bin/echo").resolve())
    assignment = {
        "schema": "hermes.roots-worker-assignment/v1",
        "assignment_id": "33333333-3333-4333-8333-333333333333", "principal": "agent:runner",
        "role": "runner", "owner_identity": "operator@example.test",
        "capabilities": ["read_main"], "partitions": ["main"],
        "authority_scope": "assigned-ticket-with-shared-main-read", "ticket_ref": "ESN-0030",
        "ticket_updated_at": "2026-10-25T14:46:11+00:00", "issued_at": now - 1,
        "expires_at": now + 300, "roots_url": "https://roots.example.test",
        "workdir": str(work), "claim_dir": str(claim_dir), "instruction_files": [str(instruction)],
        "claude_config_dir": str(provider_profile), "claude_executable": executable,
        "roots_mcp_executable": executable,
        "allowed_tools": ["mcp__roots__roots_whoami", "mcp__roots__ticket_view"],
        "session_id": "11111111-1111-4111-8111-111111111111",
        "correlation_id": "task-001", "task": "Return the ticket title.", "resume": False,
    }
    request = tmp_path / "assignment.json"
    request.write_text(json.dumps(assignment))
    request.chmod(0o600)
    whoami = {
        "principal": "agent:runner", "kind": "agent", "role": "runner",
        "acting_for": None, "owner_identity": "operator@example.test",
        "capabilities": ["read_main"], "partitions": ["main"],
        "ticket_ref": assignment["ticket_ref"],
        "ticket_updated_at": assignment["ticket_updated_at"],
        "assignment_id": assignment["assignment_id"],
        "assignment_expires_at": assignment["expires_at"],
        "token_expires_at": assignment["expires_at"], "token_jti": "jti-001",
    }
    ticket = {
        "ref": "ESN-0030", "status": "IN PROGRESS",
        "updated_at": "2026-10-25T14:46:11+00:00", "why": "Needed",
        "definition_of_done": "Done", "owner_identity": "operator@example.test",
        "assigned_principal": "agent:runner",
    }
    return request, assignment, whoami, ticket


def _token_fd() -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"bounded-token\n")
    os.close(write_fd)
    return read_fd


def _fake_proofs(monkeypatch: pytest.MonkeyPatch, whoami: dict, ticket: dict,
                 assignment: dict) -> None:
    whoami.update(
        ticket_ref=assignment["ticket_ref"],
        ticket_updated_at=assignment["ticket_updated_at"],
        assignment_id=assignment["assignment_id"],
        assignment_expires_at=assignment["expires_at"],
        token_expires_at=assignment["expires_at"],
    )
    monkeypatch.setattr(
        bootstrap, "_get_json", lambda url, token: whoami if url.endswith("/whoami") else ticket,
    )


def _seed_completed_predecessor(raw: dict) -> str:
    previous = "44444444-4444-4444-8444-444444444444"
    instruction_digests = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in raw["instruction_files"]
    }
    claim = {
        "schema": "hermes.roots-worker-claim/v1", "assignment_id": previous,
        "assignment_sha256": "a" * 64, "claimed_at": int(time.time()),
        "phase": "completed", "session_id": raw["session_id"],
        "correlation_id": "previous", "previous_assignment_id": None,
        "expires_at": int(time.time()) + 60, "accepted_at": time.time(),
        "authority": bootstrap._authority_context(raw, instruction_digests),
    }
    (Path(raw["claim_dir"]) / f"{previous}.json").write_text(json.dumps(claim))
    return previous


def _worker_script(path: Path, record: Path, *, mutation: str = "") -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "prompt = json.loads(sys.stdin.read())\n"
        "mcp = args[args.index('--mcp-config') + 1]\n"
        "instructions = args[args.index('--append-system-prompt-file') + 1]\n"
        "mcp_content = pathlib.Path(mcp).read_text()\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{"
        "'argv': args, 'prompt': prompt, 'cwd': os.getcwd(), 'pid': os.getpid(), "
        "'has_token_env': 'ROOTS_TOKEN' in os.environ, "
        "'config_dir': os.environ.get('CLAUDE_CONFIG_DIR'), "
        "'mcp': mcp_content, "
        "'instructions': pathlib.Path(instructions).read_text()}))\n"
        f"mutation = {mutation!r}\n"
        "session = args[args.index('--resume') + 1] if '--resume' in args else args[args.index('--session-id') + 1]\n"
        "correlation = prompt['correlation_id'] if mutation != 'correlation' else 'wrong'\n"
        "session = session if mutation != 'session' else '22222222-2222-4222-8222-222222222222'\n"
        "if mutation == 'nonzero': print('failed deliberately', file=sys.stderr); sys.exit(7)\n"
        "if mutation == 'leak-token':\n"
        " token = json.loads(mcp_content)['mcpServers']['roots']['env']['ROOTS_TOKEN']\n"
        " print(token, file=sys.stderr); sys.exit(7)\n"
        "if mutation == 'oversized': print('x' * 1048577); sys.exit(0)\n"
        "if mutation == 'hang':\n"
        " import signal, time\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        " while True: time.sleep(1)\n"
        "if mutation == 'invalid-json': print('not json')\n"
        "else:\n"
        " answer = (json.loads(mcp_content)['mcpServers']['roots']['env']['ROOTS_TOKEN'] "
        "if mutation == 'token-success' else 'completed')\n"
        " print(json.dumps({'session_id': session, 'structured_output': {"
        "'correlation_id': correlation, 'status': 'blocked' if mutation == 'blocked' else 'completed', "
        "'answer': answer}}))\n"
    )
    path.chmod(0o700)


def test_exact_grant_and_ticket_produce_credential_free_receipt(tmp_path: Path) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    assignment = validate_assignment(raw)
    verify_proofs(assignment, whoami, ticket)
    receipt = build_receipt(
        request.read_bytes(), assignment, whoami,
        {assignment["instruction_files"][0]: "a" * 64}, launched_pid=None,
        executable_digests={"claude": "b" * 64, "roots_mcp": "c" * 64},
    )
    target = tmp_path / "receipt.json"
    write_new_receipt(target, receipt)
    persisted = json.loads(target.read_text())
    assert persisted["verified"] is True and persisted["ready"] is False
    assert persisted["state"] == "verified"
    assert "token" not in target.read_text().lower()
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(("change", "message"), [
    (lambda who, ticket: who["capabilities"].append("write_ticket"), "wider"),
    (lambda who, ticket: who.update(principal="agent:other"), "principal"),
    (lambda who, ticket: who.update(owner_identity="somebody-else"), "owner_identity"),
    (lambda who, ticket: who.update(acting_for="somebody-else"), "acting_for"),
    (lambda who, ticket: ticket.update(status="DONE"), "status"),
    (lambda who, ticket: ticket.update(updated_at="stale"), "updated_at"),
    (lambda who, ticket: ticket.update(assigned_principal="agent:other"), "assigned_principal"),
])
def test_identity_grant_and_ticket_must_match_exactly(tmp_path: Path, change, message: str) -> None:
    _, raw, whoami, ticket = _fixture(tmp_path)
    assignment = validate_assignment(raw)
    change(whoami, ticket)
    with pytest.raises(BootstrapRefusal, match=message):
        verify_proofs(assignment, whoami, ticket)


@pytest.mark.parametrize(("change", "message"), [
    (lambda raw: raw.update(correlation_id="has spaces"), "correlation_id"),
    (lambda raw: raw.update(resume="yes"), "resume"),
    (lambda raw: raw.update(assignment_id="not-a-uuid"), "assignment_id"),
    (lambda raw: raw.update(session_id="not-a-uuid"), "canonical UUID"),
    (lambda raw: raw.update(task="   "), "non-whitespace"),
    (lambda raw: raw.update(allowed_tools=["Bash"]), "Roots MCP"),
    (lambda raw: raw.update(allowed_tools=["mcp__roots__*"]), "Roots MCP"),
    (lambda raw: raw.update(allowed_tools=["mcp__roots__ticket_comment"]), "read-only"),
    (lambda raw: raw.update(resume=True), "previous_assignment_id"),
    (lambda raw: raw.update(previous_assignment_id="44444444-4444-4444-8444-444444444444"),
     "only valid with resume"),
    (lambda raw: raw.update(task="x" * 65_537), "task is too large"),
])
def test_assignment_refuses_ambiguous_or_overbroad_task_contract(tmp_path: Path, change, message: str) -> None:
    _, raw, _, _ = _fixture(tmp_path)
    change(raw)
    with pytest.raises(BootstrapRefusal, match=message):
        validate_assignment(raw)


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
    read_fd = _token_fd()
    assert read_inherited_token(read_fd) == "bounded-token"
    with pytest.raises(OSError):
        os.read(read_fd, 1)


def test_executable_probes_roots_over_inherited_token_before_readiness(tmp_path: Path) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            payload = json.dumps(whoami if self.path == "/whoami" else ticket).encode()
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
    read_fd = _token_fd()
    receipt = tmp_path / "receipt.json"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.roots_worker_bootstrap", "--request", str(request),
             "--token-fd", str(read_fd), "--receipt", str(receipt), "--verify-only"],
            pass_fds=(read_fd,), capture_output=True, text=True, timeout=10,
        )
    finally:
        os.close(read_fd)
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    assert result.returncode == 0, result.stderr
    assert [path for path, _ in seen] == ["/whoami", "/tickets/ESN-0030"]
    assert {auth for _, auth in seen} == {"Bearer bounded-token"}
    assert json.loads(receipt.read_text())["ready"] is False
    assert "bounded-token" not in receipt.read_text()


def test_headless_task_returns_exact_correlated_result_without_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    record, executable = tmp_path / "child.json", tmp_path / "claude"
    _worker_script(executable, record)
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    receipt = tmp_path / "receipt.json"
    bootstrap.run(request, _token_fd(), receipt)
    child, persisted = json.loads(record.read_text()), json.loads(receipt.read_text())
    assert persisted["state"] == "completed" and persisted["ready"] is True
    assert persisted["result"] == {
        "correlation_id": "task-001", "status": "completed", "answer": "completed",
    }
    assert persisted["response_sha256"] and child["prompt"]["task"] == raw["task"]
    assert child["has_token_env"] is False and child["config_dir"] == raw["claude_config_dir"]
    assert json.loads(child["mcp"])["mcpServers"]["roots"]["env"]["ROOTS_TOKEN"] == "bounded-token"
    assert "bounded worker" in child["instructions"] and child["cwd"] == raw["workdir"]
    assert "-p" in child["argv"] and "--channels" not in child["argv"]
    assert child["argv"][child["argv"].index("--tools") + 1] == ""
    assert child["argv"][child["argv"].index("--setting-sources") + 1] == ""
    assert set(raw["allowed_tools"]).issubset(child["argv"])
    assert "bounded-token" not in receipt.read_text()
    assert not Path(persisted["launch_stage_dir"]).exists()


def test_resume_uses_existing_session_and_new_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    record, executable = tmp_path / "child.json", tmp_path / "claude"
    _worker_script(executable, record)
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable),
               assignment_id="55555555-5555-4555-8555-555555555555",
               previous_assignment_id=_seed_completed_predecessor(raw),
               resume=True, correlation_id="retry-002")
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    receipt = tmp_path / "receipt.json"
    bootstrap.run(request, _token_fd(), receipt)
    child, persisted = json.loads(record.read_text()), json.loads(receipt.read_text())
    assert "--resume" in child["argv"] and "--session-id" not in child["argv"]
    assert persisted["resumed"] is True and persisted["result"]["correlation_id"] == "retry-002"


def test_assignment_claim_prevents_sequential_replay_with_another_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    record, executable = tmp_path / "child.json", tmp_path / "claude"
    _worker_script(executable, record)
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    bootstrap.run(request, _token_fd(), tmp_path / "first-receipt.json")
    first_record = record.read_bytes()
    with pytest.raises(BootstrapRefusal, match="already been claimed"):
        bootstrap.run(request, _token_fd(), tmp_path / "second-receipt.json")
    assert record.read_bytes() == first_record
    assert not (tmp_path / "second-receipt.json").exists()
    claim = Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json"
    assert json.loads(claim.read_text())["assignment_id"] == raw["assignment_id"]


def test_assignment_claim_is_atomic_across_concurrent_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    outcomes: list[str] = []

    def launch(index: int) -> None:
        try:
            bootstrap.run(request, _token_fd(), tmp_path / f"receipt-{index}.json")
            outcomes.append("completed")
        except BootstrapRefusal as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=launch, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert outcomes.count("completed") == 1
    assert sum("already been claimed" in outcome for outcome in outcomes) == 1
    assert len(list(tmp_path.glob("receipt-*.json"))) == 1


def test_crash_before_spawn_leaves_uncertain_claim_that_cannot_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        bootstrap.run(request, _token_fd(), tmp_path / "first-receipt.json")
    monkeypatch.setattr(subprocess, "Popen", original_popen)
    prior = json.loads((Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json").read_text())
    assert prior["phase"] == "launching"
    raw.update(
        assignment_id="66666666-6666-4666-8666-666666666666",
        previous_assignment_id=prior["assignment_id"], resume=True, correlation_id="recovery",
    )
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    with pytest.raises(BootstrapRefusal, match="not a completed assignment"):
        bootstrap.run(request, _token_fd(), tmp_path / "recovery-receipt.json")
    assert not (Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json").exists()


def test_completed_claim_recovers_result_when_receipt_publication_crashes_and_allows_next_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    original_publish = bootstrap._ReceiptPublisher.publish
    monkeypatch.setattr(
        bootstrap._ReceiptPublisher, "publish",
        lambda self, receipt: (_ for _ in ()).throw(BootstrapRefusal("publication crash")),
    )
    with pytest.raises(BootstrapRefusal, match="publication crash"):
        bootstrap.run(request, _token_fd(), tmp_path / "first-receipt.json")
    prior = json.loads((Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json").read_text())
    assert prior["phase"] == "completed"
    assert prior["result"] == {
        "correlation_id": "task-001", "status": "completed", "answer": "completed",
    }
    monkeypatch.setattr(bootstrap._ReceiptPublisher, "publish", original_publish)
    raw.update(
        assignment_id="77777777-7777-4777-8777-777777777777",
        previous_assignment_id=prior["assignment_id"], resume=True, correlation_id="next-task",
    )
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    bootstrap.run(request, _token_fd(), tmp_path / "next-receipt.json")
    assert json.loads((tmp_path / "next-receipt.json").read_text())["ready"] is True


@pytest.mark.parametrize("authority_change", ["ticket", "principal"])
def test_resume_rejects_completed_predecessor_from_another_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authority_change: str,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    previous = _seed_completed_predecessor(raw)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(
        claude_executable=str(executable), roots_mcp_executable=str(executable),
        assignment_id="88888888-8888-4888-8888-888888888888",
        previous_assignment_id=previous, resume=True,
    )
    if authority_change == "ticket":
        raw["ticket_ref"] = ticket["ref"] = "ESN-9999"
    else:
        raw["principal"] = whoami["principal"] = ticket["assigned_principal"] = "agent:other"
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    with pytest.raises(BootstrapRefusal, match="not a completed assignment"):
        bootstrap.run(request, _token_fd(), tmp_path / "receipt.json")
    assert not (Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json").exists()


def test_expiry_after_result_never_commits_completed_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    original = bootstrap._validated_result

    def expire_after_validation(stdout: bytes, assignment: dict, token: str):
        result = original(stdout, assignment, token)
        assignment["expires_at"] = int(time.time())
        return result

    monkeypatch.setattr(bootstrap, "_validated_result", expire_after_validation)
    with pytest.raises(BootstrapRefusal, match="expired before result acceptance"):
        bootstrap.run(request, _token_fd(), tmp_path / "receipt.json")
    claim = json.loads((Path(raw["claim_dir"]) / f"{raw['assignment_id']}.json").read_text())
    assert claim["phase"] == "spawned"
    assert "accepted_at" not in claim and not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(("mutation", "message"), [
    ("correlation", "correlation_id"), ("session", "session_id"), ("invalid-json", "not JSON"),
    ("nonzero", "diagnostics withheld"), ("leak-token", "diagnostics withheld"),
    ("token-success", "credential material"), ("oversized", "response limit"),
    ("blocked", "reported.*blocked"),
])
def test_wrong_or_malformed_claude_result_never_publishes_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, message: str,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json", mutation=mutation)
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(BootstrapRefusal, match=message) as failure:
        bootstrap.run(request, _token_fd(), receipt)
    assert "bounded-token" not in str(failure.value)
    assert not receipt.exists() and not list(tmp_path.glob(".roots-worker-stage-*"))


def test_timeout_kills_worker_and_never_publishes_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    record, executable = tmp_path / "child.json", tmp_path / "claude"
    _worker_script(executable, record, mutation="hang")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable),
               expires_at=int(time.time()) + 3)
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(BootstrapRefusal, match="exceeded the assignment lifetime"):
        bootstrap.run(request, _token_fd(), receipt)
    pid = json.loads(record.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not receipt.exists() and not list(tmp_path.glob(".roots-worker-stage-*"))


def test_config_directory_replacement_is_detected_after_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, raw, whoami, ticket = _fixture(tmp_path)
    executable = tmp_path / "claude"
    _worker_script(executable, tmp_path / "child.json")
    raw.update(claude_executable=str(executable), roots_mcp_executable=str(executable))
    request.write_text(json.dumps(raw))
    _fake_proofs(monkeypatch, whoami, ticket, raw)
    original, checks = bootstrap._verify_pinned_directory_path, 0

    def replace_on_second_check(path: str, fd: int, name: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            os.replace(path, tmp_path / "old-provider-profile")
            Path(path).mkdir()
        original(path, fd, name)

    monkeypatch.setattr(bootstrap, "_verify_pinned_directory_path", replace_on_second_check)
    with pytest.raises(BootstrapRefusal, match="changed after review"):
        bootstrap.run(request, _token_fd(), tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


def test_staged_digest_detects_in_place_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"a" * 100_000)
    executable.chmod(0o700)
    pinned = bootstrap._pin_executable(str(executable), "tool")
    stage = tmp_path / "stage"; stage.mkdir()
    original_read, changed = os.read, False

    def mutate_during_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(fd, size)
        if fd == pinned.read_fd and chunk and not changed:
            executable.write_bytes(b"b" * 100_000); executable.chmod(0o700); changed = True
        return chunk

    monkeypatch.setattr(os, "read", mutate_during_read)
    try:
        with pytest.raises(BootstrapRefusal, match="changed while it was staged"):
            bootstrap._stage_executable(pinned, stage, "staged")
    finally:
        pinned.close()


def test_cleanup_kills_descendant_after_group_leader_exits(tmp_path: Path) -> None:
    child_record, script = tmp_path / "descendant", tmp_path / "forker.py"
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
