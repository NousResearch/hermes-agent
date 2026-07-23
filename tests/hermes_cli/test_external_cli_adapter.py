from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.external_cli_adapter import (
    BLOCKED_ENV_NAMES,
    MAX_CAPTURE_BYTES,
    ClaudeCliStrategy,
    CodexCliStrategy,
    ExternalCliAgentAdapter,
    ExternalCliExecutionRequest,
    ExternalCliStructuredPayload,
    ExternalCliWorkerConfig,
    _build_subprocess_env,
    active_external_cli_registry,
    load_external_cli_worker_config,
    sanitize_external_cli_payload,
    worker_config_defaults,
)


class _Readable:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        return None


class _Writable:
    def __init__(self) -> None:
        self.payload = b""
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise ValueError("closed")
        self.payload += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _BlockingWritable(_Writable):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def write(self, data: bytes) -> int:
        self.release.wait(timeout=5)
        if self.closed:
            raise ValueError("closed")
        return super().write(data)

    def close(self) -> None:
        self.closed = True
        self.release.set()


class _FakeProc:
    def __init__(self, *, returncode=0, stdout=b"", stderr=b"", stdin=None) -> None:
        self.args = []
        self.pid = 4321
        self.returncode = returncode
        self.stdin = stdin or _Writable()
        self.stdout = _Readable(stdout)
        self.stderr = _Readable(stderr)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        return self.returncode

    def terminate(self):
        self.returncode = self.returncode if self.returncode is not None else 143

    def kill(self):
        self.returncode = self.returncode if self.returncode is not None else 137


def _request(tmp_path: Path, *, cancellation_requested=None, timeout_seconds=None, executable="claude") -> ExternalCliExecutionRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ExternalCliExecutionRequest(
        task_id="task-123",
        profile_name=f"{executable}-coder",
        prompt="Fix the failing test",
        workspace_path=str(workspace),
        timeout_seconds=timeout_seconds,
        evidence_dir=str(tmp_path / "evidence"),
        cancellation_requested=cancellation_requested,
    )


def _cfg(executable="claude") -> ExternalCliWorkerConfig:
    return ExternalCliWorkerConfig(
        execution_backend="external_cli",
        executable=executable,
        authentication_mode="cli_managed_subscription",
        output_mode="structured",
    )


def _contract(summary="done", metadata=None) -> str:
    return json.dumps({
        "hermes_external_cli_result": {
            "action": "complete",
            "summary": summary,
            "metadata": metadata or {"tests_run": ["pytest"]},
        }
    })


def _claude_stdout(summary="done", metadata=None) -> bytes:
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": _contract(summary, metadata),
        "session_id": "fixture",
    }).encode()


def _codex_stdout(messages: list[str], terminal="turn.completed") -> bytes:
    events = [{"type": "item.completed", "item": {"type": "agent_message", "text": text}} for text in messages]
    events.append({"type": terminal})
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


def _run_with_output(tmp_path, monkeypatch, *, executable="claude", stdout=b"", stderr=b"", returncode=0, stdin=None):
    proc = _FakeProc(returncode=returncode, stdout=stdout, stderr=stderr, stdin=stdin)
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: f"/usr/bin/{executable}")
    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", lambda argv, **kwargs: proc)
    return ExternalCliAgentAdapter().run(_cfg(executable), _request(tmp_path, executable=executable)), proc


def test_worker_config_defaults_shape():
    defaults = worker_config_defaults()
    assert defaults["execution_backend"] == "hermes_internal"
    assert defaults["external_cli"]["output_mode"] == "auto"
    assert defaults["external_cli"]["allow_resume"] is False


def test_load_external_cli_worker_config_defaults_to_internal_backend():
    cfg = load_external_cli_worker_config({})
    assert cfg.execution_backend == "hermes_internal"
    assert cfg.executable == ""
    assert cfg.args == ()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"worker": {"execution_backend": "bogus"}}, "execution_backend"),
        ({"worker": {"execution_backend": "external_cli", "external_cli": {"executable": "python", "authentication_mode": "cli_managed_subscription"}}}, "executable"),
        ({"worker": {"execution_backend": "external_cli", "external_cli": {"executable": "claude", "authentication_mode": "oauth"}}}, "authentication_mode"),
        ({"worker": {"execution_backend": "external_cli", "external_cli": {"executable": "claude", "authentication_mode": "cli_managed_subscription", "args": ["--sandbox", "workspace-write"]}}}, "must be empty"),
        ({"worker": {"execution_backend": "external_cli", "external_cli": {"executable": "claude", "authentication_mode": "cli_managed_subscription", "api_key": "secret"}}}, "forbidden fields"),
        ({"worker": {"execution_backend": "external_cli", "external_cli": {"executable": "claude", "authentication_mode": "cli_managed_subscription", "allow_resume": True}}}, "allow_resume"),
    ],
)
def test_load_external_cli_worker_config_rejects_invalid_values(config, expected):
    with pytest.raises(ValueError, match=expected):
        load_external_cli_worker_config(config)


def test_load_external_cli_worker_config_accepts_valid_external_cli():
    cfg = load_external_cli_worker_config({
        "worker": {
            "execution_backend": "external_cli",
            "external_cli": {
                "executable": "codex",
                "args": [],
                "authentication_mode": "cli_managed_subscription",
                "output_mode": "structured",
            },
        }
    })
    assert cfg.execution_backend == "external_cli"
    assert cfg.executable == "codex"
    assert cfg.args == ()


def test_load_external_cli_worker_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown fields"):
        load_external_cli_worker_config({
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "codex",
                    "authentication_mode": "cli_managed_subscription",
                    "env_allowlist": ["HOME"],
                },
            }
        })


def test_build_subprocess_env_only_keeps_safe_names(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/home")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    cfg = ExternalCliWorkerConfig(execution_backend="external_cli", executable="claude", authentication_mode="cli_managed_subscription")
    env = _build_subprocess_env(cfg)
    assert env["HOME"] == "/tmp/home"
    assert env["PATH"] == "/usr/bin"
    assert env["TMPDIR"] == "/tmp"
    assert not (BLOCKED_ENV_NAMES & set(env))


def test_claude_strategy_builds_non_shell_argv():
    argv = ClaudeCliStrategy().build_argv(_cfg("claude"), ExternalCliExecutionRequest(
        task_id="t1", profile_name="claude-coder", prompt="Do the task", workspace_path="/tmp/work",
        timeout_seconds=30, evidence_dir="/tmp/evidence", model="claude-sonnet",
    ))
    assert argv == ["claude", "-p", "--output-format", "json", "--model", "claude-sonnet"]


def test_codex_strategy_writes_strict_secure_schema_and_builds_argv(tmp_path):
    req = _request(tmp_path, executable="codex")
    req = ExternalCliExecutionRequest(**{**req.__dict__, "model": "gpt-5-codex"})
    argv = CodexCliStrategy().build_argv(_cfg("codex"), req)
    schema_index = argv.index("--output-schema") + 1
    schema_path = Path(argv[schema_index])
    assert argv[-2:] == ["--model", "gpt-5-codex"]
    schema = json.loads(schema_path.read_text())
    assert schema["additionalProperties"] is False
    result_schema = schema["properties"]["hermes_external_cli_result"]
    assert result_schema["additionalProperties"] is False
    assert set(result_schema["required"]) == set(result_schema["properties"])
    metadata_schema = result_schema["properties"]["metadata"]
    assert set(metadata_schema["required"]) == set(metadata_schema["properties"])
    assert stat.S_IMODE(schema_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(schema_path.parent.stat().st_mode) == 0o700


def test_codex_strategy_forces_explicit_workspace_write_sandbox(tmp_path):
    """P0-F M4: cwd alone is not a security boundary. Codex must always get
    an explicit sandbox, an explicit --cd, and must never load ambient
    $CODEX_HOME config/rules or persist a resumable session."""
    req = _request(tmp_path, executable="codex")
    argv = CodexCliStrategy().build_argv(_cfg("codex"), req)

    assert argv[0:2] == ["codex", "exec"]
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert "--cd" in argv
    workspace_arg = argv[argv.index("--cd") + 1]
    assert Path(workspace_arg) == Path(req.workspace_path)
    for required_flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check"):
        assert required_flag in argv, f"missing containment flag: {required_flag}"

    forbidden_substrings = ("bypass", "resume", "--profile", "-p ", "danger-full-access", "read-only")
    joined = " ".join(argv)
    for forbidden in forbidden_substrings:
        assert forbidden not in joined, f"forbidden token present in codex argv: {forbidden!r}"


def test_codex_worker_config_cannot_inject_sandbox_bypass_args():
    """Structural guarantee backing M4: v1 config has no args-passthrough at
    all, so a misconfigured or compromised config cannot add
    --dangerously-bypass-approvals-and-sandbox or similar to the real argv."""
    with pytest.raises(ValueError, match="must be empty"):
        load_external_cli_worker_config({
            "worker": {
                "execution_backend": "external_cli",
                "external_cli": {
                    "executable": "codex",
                    "authentication_mode": "cli_managed_subscription",
                    "args": ["--dangerously-bypass-approvals-and-sandbox"],
                },
            }
        })


def test_claude_documented_wrapper_success():
    payload = ClaudeCliStrategy().parse_output(_claude_stdout().decode())
    assert payload and payload.action == "complete" and payload.summary == "done"


def test_claude_wrapper_error_is_rejected():
    output = json.dumps({"type": "result", "subtype": "success", "is_error": True, "result": _contract()})
    assert ClaudeCliStrategy().parse_output(output) is None


def test_claude_invalid_result_string_is_rejected():
    output = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "not-json"})
    assert ClaudeCliStrategy().parse_output(output) is None


def test_codex_jsonl_chooses_last_agent_message():
    stdout = _codex_stdout([_contract("early"), _contract("final")]).decode()
    payload = CodexCliStrategy().parse_output(stdout)
    assert payload and payload.summary == "final"


@pytest.mark.parametrize("stdout", [
    (json.dumps({"type": "turn.failed"}) + "\n").encode(),
    (json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": _contract()}}) + "\n").encode(),
    b"not-json\n",
])
def test_codex_invalid_terminal_stream_is_rejected(stdout):
    assert CodexCliStrategy().parse_output(stdout.decode()) is None


def test_payload_sanitizes_and_allowlists_metadata():
    payload = sanitize_external_cli_payload(ExternalCliStructuredPayload(
        action="complete",
        summary="x" * 3000 + "\x00",
        metadata={"changed_files": ["a.py"], "tests_run": ["pytest"], "unknown": {"raw": "data"}},
    ))
    assert len(payload.summary or "") <= 2000
    assert payload.metadata == {"changed_files": ["a.py"], "tests_run": ["pytest"]}


@pytest.mark.parametrize("bad_path", [
    "/etc/passwd",
    "/home/curioctylab/.hermes/secrets.env",
    "../../etc/passwd",
    "a/../../b",
    "~/.ssh/id_rsa",
    "C:/Users/admin/secrets.txt",
])
def test_path_like_metadata_rejects_absolute_and_traversal_paths(bad_path):
    payload = sanitize_external_cli_payload(ExternalCliStructuredPayload(
        action="complete",
        summary="done",
        metadata={"changed_files": [bad_path, "src/ok.py"], "evidence": [bad_path]},
    ))
    assert payload.metadata["changed_files"] == ["src/ok.py"]
    assert payload.metadata.get("evidence", []) == []


def test_path_like_metadata_accepts_clean_relative_paths():
    payload = sanitize_external_cli_payload(ExternalCliStructuredPayload(
        action="complete",
        summary="done",
        metadata={"changed_files": ["src/app.py", "./tests/test_app.py"], "evidence": ["logs/run.log"]},
    ))
    assert payload.metadata["changed_files"] == ["src/app.py", "tests/test_app.py"]
    assert payload.metadata["evidence"] == ["logs/run.log"]


def test_adapter_reports_missing_executable(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: None)
    result = ExternalCliAgentAdapter().run(_cfg(), _request(tmp_path))
    assert result.status == "FAILED_EXECUTABLE_MISSING"
    assert result.exit_code == 127


def test_adapter_normalizes_popen_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    result = ExternalCliAgentAdapter().run(_cfg(), _request(tmp_path))
    assert result.status == "FAILED_EXECUTABLE_MISSING"


def test_adapter_normalizes_invalid_cwd(tmp_path):
    req = _request(tmp_path)
    req = ExternalCliExecutionRequest(**{**req.__dict__, "workspace_path": str(tmp_path / "missing")})
    result = ExternalCliAgentAdapter().run(_cfg(), req)
    assert result.status == "FAILED_INVALID_INVOCATION"


def test_adapter_uses_shell_false_start_new_session_and_stdin_prompt(tmp_path, monkeypatch):
    created = {}
    stdout = _claude_stdout()

    def fake_popen(argv, **kwargs):
        proc = _FakeProc(returncode=0, stdout=stdout)
        proc.args = argv
        created.update(argv=argv, kwargs=kwargs, proc=proc)
        return proc

    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", fake_popen)
    result = ExternalCliAgentAdapter().run(_cfg(), _request(tmp_path))
    assert created["kwargs"]["shell"] is False
    assert created["kwargs"]["start_new_session"] is True
    assert b"Task ID: task-123" in created["proc"].stdin.payload
    assert result.status == "COMPLETED"
    assert result.structured_payload is not None
    assert result.stdout_artifact_path
    metadata = result.as_metadata()
    assert "stdout_text" not in metadata and "stderr_text" not in metadata


def test_adapter_codex_jsonl_success(tmp_path, monkeypatch):
    result, _ = _run_with_output(tmp_path, monkeypatch, executable="codex", stdout=_codex_stdout([_contract("final")]))
    assert result.status == "COMPLETED"
    assert result.structured_payload and result.structured_payload.summary == "final"


def test_adapter_classifies_preserved_codex_invalid_schema_terminal_failure(
    tmp_path, monkeypatch
):
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "codex_invalid_schema_turn_failed.jsonl"
    ).read_bytes()
    result, _ = _run_with_output(
        tmp_path,
        monkeypatch,
        executable="codex",
        stdout=fixture,
        returncode=1,
    )
    assert result.status == "FAILED_INVALID_INVOCATION"
    assert result.structured_output_status == "terminal_failure"
    assert "invalid_json_schema" in result.stdout_summary


def test_artifact_modes_are_private(tmp_path, monkeypatch):
    result, _ = _run_with_output(tmp_path, monkeypatch, stdout=_claude_stdout())
    assert result.stdout_artifact_path
    assert stat.S_IMODE(Path(result.stdout_artifact_path).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(result.stdout_artifact_path).parent.stat().st_mode) == 0o700


def test_symlink_artifact_overwrite_is_rejected(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text("safe")
    (evidence / "task-123-stdout-fixed.log").symlink_to(target)
    monkeypatch.setattr("hermes_cli.external_cli_adapter.secrets.token_hex", lambda _n: "fixed")
    result, _ = _run_with_output(tmp_path, monkeypatch, stdout=_claude_stdout())
    assert result.status == "FAILED_INTERNAL"
    assert target.read_text() == "safe"


@pytest.mark.parametrize(("stderr", "exit_code", "expected"), [
    (b"login required", 1, "BLOCKED_AUTH"),
    (b"quota exhausted", 1, "BLOCKED_QUOTA"),
    (b"permission denied", 1, "BLOCKED_PERMISSION"),
    (b"denied by sandbox policy", 1, "BLOCKED_PERMISSION"),
    # P0-F m4: a bare mention of "sandbox" in an unrelated runtime-init
    # failure must NOT be classified as an actionable permission block.
    (b"fatal: no sandbox helper binary found on PATH", 1, "FAILED_INTERNAL"),
    (b"invalid sandbox mode 'bogus' in config", 1, "FAILED_INTERNAL"),
])
def test_adapter_classifies_blocking_failures(tmp_path, monkeypatch, stderr, exit_code, expected):
    result, _ = _run_with_output(tmp_path, monkeypatch, stderr=stderr, returncode=exit_code)
    assert result.status == expected


def test_adapter_timeout_remains_active_during_blocked_stdin(tmp_path, monkeypatch):
    proc = _FakeProc(returncode=None, stdin=_BlockingWritable())
    clock = {"now": 0.0}
    terminations = []
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr("hermes_cli.external_cli_adapter.time.sleep", lambda _s: None)
    monkeypatch.setattr("hermes_cli.external_cli_adapter.time.monotonic", lambda: clock.update(now=clock["now"] + 1.0) or clock["now"])
    monkeypatch.setattr(ExternalCliAgentAdapter, "_terminate_process_tree", staticmethod(lambda p, force=False: (terminations.append(force), setattr(p, "returncode", 124))))
    result = ExternalCliAgentAdapter().run(_cfg(), _request(tmp_path, timeout_seconds=1))
    assert result.status == "FAILED_TIMEOUT"
    assert terminations == [False]


def test_adapter_cancellation_remains_active_during_blocked_stdin(tmp_path, monkeypatch):
    proc = _FakeProc(returncode=None, stdin=_BlockingWritable())
    terminations = []
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr("hermes_cli.external_cli_adapter.time.sleep", lambda _s: None)
    monkeypatch.setattr(ExternalCliAgentAdapter, "_terminate_process_tree", staticmethod(lambda p, force=False: (terminations.append(force), setattr(p, "returncode", 130))))
    result = ExternalCliAgentAdapter().run(_cfg(), _request(tmp_path, cancellation_requested=lambda: True))
    assert result.status == "FAILED_CANCELLED"
    assert terminations == [False]


# ---------------------------------------------------------------------------
# P0-F M3 regression: Codex JSONL exceeding the bounded capture buffer.
#
# _StreamCapture only ever retains MAX_CAPTURE_BYTES for the on-disk evidence
# artifact. Before the incremental parser, parsing ran against that truncated
# buffer, so any valid Codex transcript longer than 256 KiB silently lost its
# terminal turn.completed / last agent_message and was misreported as
# FAILED_MALFORMED_OUTPUT. The incremental parser now runs off the live
# stream instead, independent of the capture cap.
# ---------------------------------------------------------------------------

def test_adapter_codex_success_survives_output_exceeding_capture_cap(tmp_path, monkeypatch):
    # Padding entries the incremental parser must not choke on before the
    # true final agent_message + turn.completed arrive.
    filler_events = [
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "x" * 4000}})
        for _ in range(400)
    ]
    body = "\n".join(filler_events) + "\n"
    stdout = (body.encode() + _codex_stdout([_contract("early"), _contract("final")]))
    assert len(stdout) > MAX_CAPTURE_BYTES * 4, "fixture must exceed the bounded capture buffer by a wide margin"

    result, _ = _run_with_output(tmp_path, monkeypatch, executable="codex", stdout=stdout)

    assert result.status == "COMPLETED"
    assert result.structured_payload is not None
    assert result.structured_payload.summary == "final"
    # The raw evidence artifact is still capped, independent of parsing.
    assert result.stdout_truncated is True
    assert result.stdout_total_size == len(stdout)


def test_adapter_codex_incremental_parser_rejects_malformed_line_anywhere(tmp_path, monkeypatch):
    filler = "\n".join(
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "x" * 4000}})
        for _ in range(400)
    )
    stdout = (filler + "\nnot-json\n").encode() + _codex_stdout([_contract("final")])
    assert len(stdout) > MAX_CAPTURE_BYTES * 4

    result, _ = _run_with_output(tmp_path, monkeypatch, executable="codex", stdout=stdout)
    assert result.status == "FAILED_MALFORMED_OUTPUT"


def test_codex_incremental_parser_handles_chunk_split_mid_line():
    from hermes_cli.external_cli_adapter import _CodexIncrementalParser

    stdout = _codex_stdout([_contract("final")])
    parser = _CodexIncrementalParser()
    # Feed one byte at a time to force lines to split across feed() calls.
    for i in range(0, len(stdout), 1):
        parser.feed(stdout[i : i + 1])
    payload = parser.result()
    assert payload is not None
    assert payload.summary == "final"


def test_output_truncation_metadata(tmp_path, monkeypatch):
    output = b"x" * (MAX_CAPTURE_BYTES + 100)
    result, _ = _run_with_output(tmp_path, monkeypatch, stdout=output)
    assert result.stdout_size == MAX_CAPTURE_BYTES
    assert result.stdout_total_size == MAX_CAPTURE_BYTES + 100
    assert result.stdout_truncated is True


# ---------------------------------------------------------------------------
# P0-F M1 regression: real OS subprocess/pipe fixtures.
#
# The fake fixtures above (_FakeProc, _BlockingWritable) cannot reproduce the
# actual bug: a real io.BufferedWriter holds an internal lock while write()
# blocks in the kernel. _BlockingWritable.close() releases its own test event
# instead, so it can't catch a close-before-terminate ordering regression.
# These tests spawn a real child that never reads stdin and never exits on
# its own, forcing an actual blocked write() in the writer thread.
# ---------------------------------------------------------------------------

def _real_popen_capturing(monkeypatch, sink: list) -> None:
    """Let subprocess.Popen run for real, but keep a handle to the child so
    the test can assert on its actual OS-level lifecycle afterwards."""
    real_popen = subprocess.Popen

    def _capture(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        sink.append(proc)
        return proc

    monkeypatch.setattr("hermes_cli.external_cli_adapter.subprocess.Popen", _capture)


def _never_reads_stdin_argv() -> list[str]:
    # Never touches stdin and never exits on its own; the adapter must be the
    # one to terminate it.
    return [sys.executable, "-c", "import time; time.sleep(30)"]


def _assert_child_reaped(proc: subprocess.Popen) -> None:
    proc.wait(timeout=5)
    assert proc.returncode is not None
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)


def test_adapter_timeout_kills_real_child_blocked_on_stdin_write(tmp_path, monkeypatch):
    procs: list[subprocess.Popen] = []
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _n: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCliStrategy, "build_argv", lambda self, cfg, req: _never_reads_stdin_argv())
    _real_popen_capturing(monkeypatch, procs)

    req = _request(tmp_path, timeout_seconds=1)
    # Comfortably larger than a pipe's kernel buffer (commonly 64 KiB on
    # Linux) so the writer thread's stdin.write() actually blocks, and just
    # under the adapter's own MAX_PROMPT_BYTES limit.
    req = ExternalCliExecutionRequest(**{**req.__dict__, "prompt": "A" * 900_000})

    started = time.monotonic()
    result = ExternalCliAgentAdapter().run(_cfg("claude"), req)
    elapsed = time.monotonic() - started

    assert elapsed < 8.0, "adapter did not return boundedly; stdin-close-before-terminate deadlock regressed"
    assert result.status == "FAILED_TIMEOUT"
    assert result.timed_out is True
    assert len(procs) == 1
    _assert_child_reaped(procs[0])


def test_adapter_cancellation_kills_real_child_blocked_on_stdin_write(tmp_path, monkeypatch):
    procs: list[subprocess.Popen] = []
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _n: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCliStrategy, "build_argv", lambda self, cfg, req: _never_reads_stdin_argv())
    _real_popen_capturing(monkeypatch, procs)

    req = _request(tmp_path, cancellation_requested=lambda: True)
    req = ExternalCliExecutionRequest(**{**req.__dict__, "prompt": "A" * 900_000})

    started = time.monotonic()
    result = ExternalCliAgentAdapter().run(_cfg("claude"), req)
    elapsed = time.monotonic() - started

    assert elapsed < 8.0
    assert result.status == "FAILED_CANCELLED"
    assert result.cancelled is True
    assert len(procs) == 1
    _assert_child_reaped(procs[0])


def test_active_external_cli_registry_cancel_kills_real_child(tmp_path, monkeypatch):
    """Exercises the P0-F M2 mechanism directly: a request_cancel() call from
    outside the adapter's own poll loop (as cli.py's SIGTERM handler makes)
    must be observed and must terminate the real child boundedly."""
    procs: list[subprocess.Popen] = []
    monkeypatch.setattr("hermes_cli.external_cli_adapter.shutil.which", lambda _n: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCliStrategy, "build_argv", lambda self, cfg, req: _never_reads_stdin_argv())
    _real_popen_capturing(monkeypatch, procs)

    def _fire_external_cancel() -> None:
        # Give the adapter a moment to Popen() and register itself active.
        for _ in range(100):
            if active_external_cli_registry.request_cancel():
                return
            time.sleep(0.02)
        pytest.fail("registry never became active; adapter did not start in time")

    trigger = threading.Thread(target=_fire_external_cancel, daemon=True)
    trigger.start()

    req = _request(tmp_path, timeout_seconds=30)
    started = time.monotonic()
    result = ExternalCliAgentAdapter().run(_cfg("claude"), req)
    elapsed = time.monotonic() - started
    trigger.join(timeout=5)

    assert elapsed < 8.0
    assert result.status == "FAILED_CANCELLED"
    assert len(procs) == 1
    _assert_child_reaped(procs[0])
    # Registry must be deactivated once the adapter returns, so a later
    # signal doesn't mistake a stale flag for a still-live child.
    assert active_external_cli_registry.request_cancel() is False
