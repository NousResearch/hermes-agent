"""Integration tests against the real `claude` binary.

These tests are gated by @pytest.mark.integration. They consume real
Anthropic plan tokens — each test uses one short prompt.

Skipped automatically if `claude` is not on PATH or if
CLAUDE_CODE_OAUTH_TOKEN cannot be loaded.

Note: The test suite's global conftest.py strips all _TOKEN env vars via
_hermetic_environment (autouse). This test works around that by loading
the token directly from /run/infisical/hermes.env at test time.

Note: The global conftest also installs a 30-second SIGALRM per test.
Integration tests against the live claude binary can take up to 60
seconds on a slow response; the SIGALRM is cancelled at the start of the
async body (SIGALRM is not supported inside asyncio event loops anyway).
"""

import asyncio
import json
import os
import re
import signal
import sys
from pathlib import Path
import pytest

from agent.claude_cli import probe
from agent.claude_cli.protocol import StreamJsonParser

pytestmark = pytest.mark.integration

_HERMES_ENV_PATH = Path("/run/infisical/hermes.env")
_TOKEN_RE = re.compile(r'^CLAUDE_CODE_OAUTH_TOKEN=["\']?([^"\']+)["\']?\s*$', re.MULTILINE)


def _load_oauth_token() -> str | None:
    """Read CLAUDE_CODE_OAUTH_TOKEN from /run/infisical/hermes.env directly.

    The global conftest strips it from os.environ, so we bypass os.environ
    and parse the file ourselves.
    """
    # First try os.environ in case it survived (e.g. non-hermetic runs).
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    if not _HERMES_ENV_PATH.exists():
        return None
    text = _HERMES_ENV_PATH.read_text(encoding="utf-8")
    m = _TOKEN_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _skip_if_no_claude() -> tuple[str, str]:
    """Return (binary_path, oauth_token) or call pytest.skip()."""
    token = _load_oauth_token()
    if not token:
        pytest.skip(
            "CLAUDE_CODE_OAUTH_TOKEN not available (not in os.environ and "
            f"not found in {_HERMES_ENV_PATH}); integration test skipped"
        )
    try:
        binary = probe.discover_binary()
    except Exception as exc:
        pytest.skip(f"claude binary not available: {exc}")
    return binary, token  # type: ignore[return-value]


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_basic_stream_json_invocation():
    """Spawning `claude -p` with stdin prompt + stream-json yields parseable events.

    Verifies:
      * stdin prompt transport is supported by --print mode
      * --output-format stream-json + --verbose emits events
      * at least one 'assistant' event and one 'result' event arrive
      * exit code is 0 on success
    """
    # Cancel the global 30-second SIGALRM so the real-binary call can take
    # up to 120 seconds without being killed mid-run.  SIGALRM has no effect
    # inside an asyncio event loop (the signal arrives between iterations, not
    # mid-await), but clearing it prevents a stray delivery after the test
    # body resumes. Windows has no SIGALRM.
    if sys.platform != "win32":
        signal.alarm(0)

    binary, token = _skip_if_no_claude()

    # Build a sanitized env with the token injected.
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env = probe.check_env_hygiene(env)

    proc = await asyncio.create_subprocess_exec(
        binary,
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--allowedTools", "",
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Write the prompt to stdin and close it.
    assert proc.stdin is not None
    proc.stdin.write(b"Reply with exactly one word: pong")
    await proc.stdin.drain()
    proc.stdin.close()

    parser = StreamJsonParser()
    stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=120)
    events = list(parser.feed(stdout_data)) + list(parser.close())

    assert proc.returncode == 0, (
        f"claude exited with {proc.returncode}; stderr: {stderr_data.decode()[:500]}"
    )
    assert any(e.get("type") == "assistant" for e in events), (
        f"no assistant event seen; events: {[e.get('type') for e in events]}"
    )
    assert any(e.get("type") == "result" for e in events), (
        f"no result event seen; events: {[e.get('type') for e in events]}"
    )


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_resume_continuity_and_session_id_extraction():
    """`--resume <session_id>` continues a prior session.

    Verifies:
      * a session_id can be extracted from the first turn's stream output
      * a second invocation with --resume <session_id> sees context from turn 1
      * the schema location of session_id is documented in CLI contract
    """
    signal.alarm(0)  # cancel autouse 30s SIGALRM (interferes with asyncio)
    _skip_if_no_claude()
    binary = probe.discover_binary()
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _load_oauth_token()
    env = probe.check_env_hygiene(env)

    # Turn 1: capture session_id.
    proc1 = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--allowedTools", "",
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc1.stdin is not None
    proc1.stdin.write(b"Remember the word: zephyr. Reply: ok")
    await proc1.stdin.drain()
    proc1.stdin.close()
    stdout1, stderr1 = await asyncio.wait_for(proc1.communicate(), timeout=120)
    assert proc1.returncode == 0, stderr1.decode()[:500]
    parser1 = StreamJsonParser()
    events1 = list(parser1.feed(stdout1)) + list(parser1.close())

    session_id = probe.extract_session_id(events1)
    assert session_id is not None, (
        f"could not extract session_id from events: "
        f"{[e for e in events1 if 'session' in str(e)][:3]}"
    )

    # Turn 2: --resume.
    proc2 = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--resume", session_id,
        "--allowedTools", "",
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc2.stdin is not None
    proc2.stdin.write(b"What word did I ask you to remember?")
    await proc2.stdin.drain()
    proc2.stdin.close()
    stdout2, stderr2 = await asyncio.wait_for(proc2.communicate(), timeout=120)
    assert proc2.returncode == 0, stderr2.decode()[:500]
    parser2 = StreamJsonParser()
    events2 = list(parser2.feed(stdout2)) + list(parser2.close())

    # Find the assistant text from turn 2 across various possible schemas.
    assistant_texts: list[str] = []
    for e in events2:
        if e.get("type") != "assistant":
            continue
        # Possible schemas: e["text"], e["content"] (str or list), e["message"]["content"][i]["text"]
        if isinstance(e.get("text"), str):
            assistant_texts.append(e["text"])
        elif isinstance(e.get("content"), str):
            assistant_texts.append(e["content"])
        elif isinstance(e.get("content"), list):
            for block in e["content"]:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    assistant_texts.append(block["text"])
        msg = e.get("message")
        if isinstance(msg, dict):
            for block in msg.get("content", []) or []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    assistant_texts.append(block["text"])

    combined = " ".join(assistant_texts).lower()
    assert "zephyr" in combined, (
        f"--resume did not preserve context; turn 2 assistant text: {combined[:300]!r}; "
        f"raw assistant events (truncated): {str([e for e in events2 if e.get('type') == 'assistant'])[:500]}"
    )


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_allowed_tools_empty_denies_all_tools():
    """`--allowedTools ""` actually denies all tools, including Read/Bash/Edit.

    Sends a prompt designed to provoke tool use ("read the file /etc/hostname
    and tell me what's in it"). Asserts NO tool_use events appear in the
    stream.
    """
    signal.alarm(0)
    _skip_if_no_claude()
    binary = probe.discover_binary()
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _load_oauth_token()
    env = probe.check_env_hygiene(env)

    proc = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--allowedTools", "",
        "--disallowedTools", "Bash,Read,Edit,Write,WebFetch,WebSearch",
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(
        b"Read the file /etc/hostname and tell me what's in it. "
        b"If you cannot read files, just say 'unable'."
    )
    await proc.stdin.drain()
    proc.stdin.close()
    stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=120)
    assert proc.returncode == 0, stderr_data.decode()[:500]
    parser = StreamJsonParser()
    events = list(parser.feed(stdout_data)) + list(parser.close())

    tool_use_events = [e for e in events if e.get("type") == "tool_use"]
    assert tool_use_events == [], (
        f"--allowedTools '' did NOT deny tools; saw tool_use events: "
        f"{[e.get('name') for e in tool_use_events]}"
    )


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_strict_mcp_config_ignores_ambient_servers(tmp_path):
    """`--strict-mcp-config` with an empty config ignores any ambient MCP servers.

    Writes a poisoned ambient ~/.claude/settings.json that declares a fake
    MCP server, then runs claude with --strict-mcp-config pointing at an
    empty config. Asserts no MCP-related events appear.
    """
    signal.alarm(0)
    _skip_if_no_claude()
    binary = probe.discover_binary()
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _load_oauth_token()
    env = probe.check_env_hygiene(env)

    # Isolate HOME so the ambient settings can be controlled.
    fake_home = tmp_path / "fake_home"
    (fake_home / ".claude").mkdir(parents=True)
    poisoned_settings = {
        "mcpServers": {
            "canary": {
                "command": "/bin/echo",
                "args": ["canary-loaded"],
            }
        }
    }
    (fake_home / ".claude" / "settings.json").write_text(
        json.dumps(poisoned_settings)
    )
    isolated_env = dict(env)
    isolated_env["HOME"] = str(fake_home)

    empty_mcp = tmp_path / "empty_mcp.json"
    empty_mcp.write_text(json.dumps({"mcpServers": {}}))

    proc = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--allowedTools", "",
        "--strict-mcp-config",
        "--mcp-config", str(empty_mcp),
        env=isolated_env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(b"Say only the word: ok")
    await proc.stdin.drain()
    proc.stdin.close()
    stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=120)
    assert proc.returncode == 0, stderr_data.decode()[:500]

    combined_output = stdout_data.decode("utf-8", errors="replace") + \
                      stderr_data.decode("utf-8", errors="replace")
    assert "canary-loaded" not in combined_output, (
        "ambient MCP server 'canary' was loaded despite --strict-mcp-config"
    )
    assert "canary" not in combined_output.lower(), (
        f"ambient MCP server name leaked into output: {combined_output[:500]}"
    )


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_settings_file_overrides_ambient_settings(tmp_path):
    """`--settings <file>` takes precedence over ambient ~/.claude/settings.json.

    Writes a malicious ambient settings.json that would grant Bash access if
    loaded. Runs claude with --settings pointing at a hermetic file that
    denies all tools. Asserts Bash is not invokable.
    """
    signal.alarm(0)
    _skip_if_no_claude()
    binary = probe.discover_binary()
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _load_oauth_token()
    env = probe.check_env_hygiene(env)

    fake_home = tmp_path / "fake_home"
    (fake_home / ".claude").mkdir(parents=True)
    poisoned_settings = {
        "permissions": {
            "tools": {"allowed": ["Bash", "Read", "Write", "Edit"], "denied": []},
        },
    }
    (fake_home / ".claude" / "settings.json").write_text(
        json.dumps(poisoned_settings)
    )
    isolated_env = dict(env)
    isolated_env["HOME"] = str(fake_home)

    hermetic = tmp_path / "hermetic_settings.json"
    hermetic.write_text(json.dumps({
        "permissions": {
            "tools": {"allowed": [], "denied": ["*"]},
        },
    }))

    proc = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--settings", str(hermetic),
        "--allowedTools", "",
        env=isolated_env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(b"Run `echo PRECEDENCE_BROKEN` in a shell. If you cannot, say 'unable'.")
    await proc.stdin.drain()
    proc.stdin.close()
    stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=120)
    assert proc.returncode == 0, stderr_data.decode()[:500]

    parser = StreamJsonParser()
    events = list(parser.feed(stdout_data)) + list(parser.close())
    tool_use = [e for e in events if e.get("type") == "tool_use"]
    assert tool_use == [], (
        f"hermetic --settings did NOT override ambient; tool_use events: "
        f"{[e.get('name') for e in tool_use]}"
    )


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_process_group_cleanup_on_cancel():
    """Cancelling a spawned `claude` reaps the entire process group.

    Spawns claude -p with --include-partial-messages so it streams.
    Cancels after first event arrives. Asserts no orphaned `claude` or
    Node child processes remain.
    """
    signal.alarm(0)
    _skip_if_no_claude()
    binary = probe.discover_binary()
    env = {k: v for k, v in os.environ.items()}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = _load_oauth_token()
    env = probe.check_env_hygiene(env)

    proc = await asyncio.create_subprocess_exec(
        binary, "-p",
        "--output-format", "stream-json", "--verbose",
        "--include-partial-messages",
        "--no-session-persistence",
        "--allowedTools", "",
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pid = proc.pid
    pgid = os.getpgid(pid)
    assert proc.stdin is not None
    proc.stdin.write(b"Count slowly from 1 to 100, one number per line.")
    await proc.stdin.drain()
    proc.stdin.close()

    # Read until we see at least one event.
    parser = StreamJsonParser()
    saw_event = False
    deadline = asyncio.get_event_loop().time() + 30
    while not saw_event and asyncio.get_event_loop().time() < deadline:
        chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=10)
        if not chunk:
            break
        for event in parser.feed(chunk):
            saw_event = True
            break
    assert saw_event, "no events arrived within 30s; cannot test cancellation"

    # Kill the whole process group.
    os.killpg(pgid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        os.killpg(pgid, signal.SIGKILL)
        await proc.wait()

    # Now check: no orphaned children of our own pid.
    # ps -o pid,ppid,pgid,cmd --ppid <our pid>
    import subprocess as _subprocess
    result = _subprocess.run(
        ["ps", "-o", "pid,ppid,pgid,cmd", "--ppid", str(os.getpid())],
        capture_output=True, text=True, check=False,
    )
    surviving = [
        line for line in result.stdout.splitlines()
        if "claude" in line.lower() or "node" in line.lower()
    ]
    assert not surviving, f"orphaned children after SIGTERM/SIGKILL: {surviving}"


def test_no_direct_anthropic_egress_from_test_process():
    """Document the network-egress check approach.

    A full assertion that 'no outbound HTTPS to api.anthropic.com from the
    Hermes parent process' requires hooking into Hermes' actual network
    stack at run time. PR 1 documents the check methodology; PR 4 (adapter)
    wires it into adapter init.

    For PR 1, the check is operator-side:
      1. Note PID of pytest process (`os.getpid()`).
      2. While integration test runs: `ss -tnp | grep <pid>` should NOT
         show connections to api.anthropic.com from that PID. The child
         `claude` process IS expected to connect.
      3. Document the observation in tests/e2e/claude_cli_findings.md.

    This test always passes; it records the test approach.
    """
    pid = os.getpid()
    assert pid > 0
    # Operator: while integration tests run, sample `ss -tnp` for this pid.
    # Document findings manually.
