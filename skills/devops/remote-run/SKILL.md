---
name: remote-run
description: "Execute commands on remote hosts via SSH with Paramiko."
version: 1.1.0
author: Jasper
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [SSH, Remote, Paramiko, Deployment, Sysadmin]
    related_skills: [cli-builder, host-summary]
    requires_toolsets: [ssh]
---

# Remote Run — SSH Command Execution

The `remote_run` tool lets you execute commands on remote hosts over SSH
using the `paramiko` library. Returns stdout, stderr, and exit code in a
structured JSON response.

Unlike piping commands through `ssh user@host command` via the `terminal`
tool, `remote_run` gives you proper exit code detection, structured
error output, and reliable sudo handling.

## When to Use

- Running commands on remote servers, VPS, or homelab nodes
- Multi-step SSH workflows where you need proper exit code handling
- Operations requiring sudo on a remote machine
- Automation across several hosts (deployments, config checks, health probes)
- When `terminal("ssh user@host command")` is unreliable due to TTY/mux quirks

**Don't use for:** Local commands (use `terminal`), file transfers (use
`rsync` or `scp` via `terminal`), long-lived interactive sessions (use
`terminal` with `pty=True`).

### remote_run vs the SSH Terminal Environment

Hermes also provides a persistent SSH terminal environment
(`tools/environments/ssh.py`) for interactive/long-lived sessions:

| Aspect | `remote_run` | SSH Terminal Environment |
|--------|-------------|-------------------------|
| Connection model | Fresh connection per call | Persistent ControlMaster session |
| Exit codes | ✅ Structured via Paramiko | From shell (via `$?`) |
| Sudo handling | ✅ Built-in with password via channel | Manual via `sudo` |
| Env vars | ✅ Via `env` parameter | Manual |
| Working directory | ✅ Via `workdir` parameter | Tracks CWD across commands |
| File sync | ❌ | Built-in SFTP sync |
| Use case | One-shot commands, multi-host automation | Interactive exploration, file work |

**Choose `remote_run`** for scripted, multi-host automation where you need
reliable exit codes and structured output. **Choose the SSH environment**
for interactive exploration on a single host.

## Prerequisites

- **paramiko** — installed in the Hermes venv:
  ```bash
  pip install "paramiko>=3,<4"
  ```
- **SSH access** — to the target host (key-based auth recommended)
- **ssh toolset** — the tool is in the `ssh` toolset, not the default `terminal` toolset.
  Enable it with `hermes tools` or `--toolsets ssh`.

## Quick Reference

| Scenario | Tool Call |
|----------|-----------|
| Basic command | `remote_run(host="myserver", command="whoami")` |
| With sudo | `remote_run(host="myserver", command="systemctl status nginx", sudo=True)` |
| Custom port | `remote_run(host="myserver", command="ls", port=2222)` |
| Key-based auth | `remote_run(host="myserver", command="uptime", key_file="~/.ssh/example_key")` |
| Working directory | `remote_run(host="myserver", command="./deploy.sh", workdir="/opt/app")` |
| Environment vars | `remote_run(host="myserver", command="echo $VAR", env={"VAR": "value"})` |
| Timeout | `remote_run(host="slowbox", command="sleep 30", timeout=10)` |

## Procedure

### 1. Basic Remote Execution

```python
result = remote_run(host="web-01", command="df -h /")
# Returns: {"stdout": "Filesystem ... 30% /\n", "stderr": "", "exit_code": 0}
```

The `exit_code` field is the most reliable indicator of success (0 = OK,
non-zero = error). Parse it programmatically from the JSON result.

### 2. Using Sudo

```python
# Simple sudo — works when NOPASSWD is configured
remote_run(host="server", command="journalctl -n 50", sudo=True)

# Sudo with password — password sent via channel, never in process listing
remote_run(host="server", command="apt update", sudo=True, password="<password>")
```

When `sudo=True` and a `password` is provided, the tool sends it via the SSH
channel stdin (not embedded in the command string), preventing the password
from appearing in `/proc/cmdline` or `ps aux`. When `sudo=True` without a
password, it wraps the command in `sudo bash -c '...'` which works if the
remote user has NOPASSWD sudo.

### 3. Key-Based Authentication (Recommended)

```python
# Using a specific key file
remote_run(host="db-01", command="pg_isready",
           key_file="/home/user/.ssh/example_key")

# Without key_file — uses SSH agent and default keys
remote_run(host="db-01", command="pg_isready", user="admin")
```

When no `key_file` or `password` is provided, Paramiko tries the running
SSH agent and the user's default SSH keys.

### 4. Setting Environment Variables

```python
remote_run(host="app-01", command="node app.js",
           env={"NODE_ENV": "production", "PORT": "3000"},
           workdir="/opt/app")
```

Environment variables are set via shell `export` before the command runs.
Keys are validated against `[A-Za-z_][A-Za-z0-9_]*` — invalid keys are
rejected with an error rather than silently passed to the shell.

### 5. Error Handling

The tool never raises exceptions — it always returns a JSON dict:

```python
result = remote_run(host="bad-host", command="echo x", timeout=5)
# result["error"] contains the error message
# result["exit_code"] is 1 for connection failures
# result["exit_code"] is the actual exit code for command failures
```

Check `exit_code == 0` for success. Check `"error" in result` for
connection-level failures (timeout, auth denied, host unreachable).

## Pitfalls

1. **Each call opens a new connection.** The tool creates a fresh SSHClient
   per invocation. There is no connection pooling across calls. For many
   sequential commands on the same host, this adds ~1-2s of TCP handshake
   overhead per call. Acceptable for ad-hoc operations; for bulk work,
   consider batching commands into a single call with `&&` or a script.

2. **sudo password security.** The password is sent via the SSH channel stdin
   (not embedded in the command string), so it does NOT appear in
   `/proc/cmdline` or `ps aux` on the remote host. However, it passes over
   the network in plaintext unless SSH is configured with encryption (it
   always is for modern SSH). Prefer key-based auth with NOPASSWD sudo for
   sensitive environments.

3. **Environment variable key validation.** Keys are validated against
   `[A-Za-z_][A-Za-z0-9_]*`. Invalid keys are rejected with a clear error.
   Values containing single quotes are escaped with `'\''` (end-quote,
   escaped quote, begin-quote). Values with `$`, backticks, or `$(...)` are
   NOT expanded by the shell because the export uses single quotes. This is
   intentional — it prevents command injection via env vars.

4. **PTY allocation for sudo.** When `sudo=True`, the tool allocates a PTY
   (`get_pty=True`) which is required for `sudo -S` password prompts. This
   can change output formatting compared to non-sudo commands (some tools
   produce different output on a TTY). If you see unexpected output
   formatting with sudo, it's likely the PTY effect.

5. **Default user.** If `user` is not specified, the local username
   (under which Hermes runs) is used as the SSH username. Always specify
   `user` explicitly when connecting to a host with a different username.

6. **Host key verification.** The tool uses `WarningPolicy()` which logs a
   warning for unknown or changed host keys but does not abort. This matches
   the `StrictHostKeyChecking=accept-new` pattern — convenient for ephemeral
   environments while still surfacing key changes in logs. For production use
   with strict verification, pre-configure `known_hosts` and switch to
   `RejectPolicy` in the source.

7. **Output truncation.** Stdout and stderr are read entirely into memory.
   Combined output exceeding 100,000 characters is truncated with a
   `[truncated]` marker. For commands producing very large output, use
   `terminal` with a piped SSH command instead.

8. **Toolset requirement.** The `remote_run` tool is in the `ssh` toolset
   (not the default `terminal` toolset). You must enable it:
   ```bash
   hermes tools        # Interactive toolset config
   # or
   hermes chat --toolsets ssh -q "remote_run(host=... , command=...)"
   ```

## Verification

```bash
# Check paramiko is installed
python3 -c "import paramiko; print(paramiko.__version__)"

# Quick smoke test against a known SSH host
python3 -c "
import json, sys
sys.path.insert(0, '.')
from tools.remote_run_tool import remote_run_handler
r = json.loads(remote_run_handler({'host': 'localhost', 'command': 'echo ok', 'user': '$USER'}))
assert r['exit_code'] == 0 and r['stdout'].strip() == 'ok'
print('remote_run works')
"
```
