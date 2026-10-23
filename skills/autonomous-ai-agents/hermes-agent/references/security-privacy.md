# Security & Privacy Toggles

Common "why is Hermes doing X to my output / tool calls / commands?" toggles — and the exact commands to change them. Most of these need a fresh session (`/reset` in chat, or start a new `hermes` invocation) because they're read once at startup.

### Secret redaction in tool output

Secret redaction is **on by default** — tool output (terminal stdout, `read_file`, web content, subagent summaries, etc.) is scanned for strings that look like API keys, tokens, and secrets before it enters the conversation context and logs. Leave it enabled for normal use:

```bash
hermes config set security.redact_secrets true       # keep enabled globally
```

**Restart required.** `security.redact_secrets` is snapshotted at import time — toggling it mid-session (e.g. via `export HERMES_REDACT_SECRETS=false` from a tool call) will NOT take effect for the running process. Tell the user to change it in config from a terminal, then start a new session. This is deliberate — it prevents an LLM from flipping the toggle on itself mid-task.

Disable only when you deliberately need raw credential-like strings for debugging or redactor development:
```bash
hermes config set security.redact_secrets false
```

### PII redaction in gateway messages

Separate from secret redaction. When enabled, the gateway hashes user IDs and strips phone numbers from the session context before it reaches the model:

```bash
hermes config set privacy.redact_pii true    # enable
hermes config set privacy.redact_pii false   # disable (default)
```

### Command approval prompts

By default (`approvals.mode: smart`), Hermes asks an auxiliary LLM to assess shell commands flagged as destructive (`rm -rf`, `git reset --hard`, etc.). The modes are:

- `smart` — auto-approve a low-risk command once, deny high-risk commands, and prompt when uncertain (default)
- `manual` — always prompt
- `off` — skip the interactive approval prompt for recoverable dangerous commands (equivalent to `--yolo`)

`off`/`--yolo` disables the interactive prompt, not every safety mechanism.
Two checks remain active unconditionally, even under yolo (verified in
`tools/approval.py`, which runs them "BEFORE the yolo bypass" by design):

- the **hardline floor** — commands with no recovery path (`rm -rf /`,
  `mkfs`, `dd` to a raw device, `shutdown`/`reboot`, fork bombs) are blocked
  outright, not just prompted for;
- **user-defined `approvals.deny` rules** in `config.yaml` — a deny rule is
  the user saying "never, even under yolo".

```bash
hermes config set approvals.mode smart       # recommended middle ground
hermes config set approvals.mode off         # bypass the interactive prompt (hardline/deny still apply)
```

Per-invocation bypass without changing config:
- `hermes --yolo …`
- `export HERMES_YOLO_MODE=1`

Note: YOLO / `approvals.mode: off` does NOT turn off secret redaction. They are independent.

### \"Reset permissions\" / \"make Hermes ask again\"

The user usually means: wipe the accumulated \"Always allow\" state — NOT yolo
mode, and NOT a per-edit diff prompt (which doesn't exist; file writes never
go through the interactive approval prompt the way shell commands do — but
`write_file`/`patch` still block writes to a fixed set of sensitive paths
(system-sensitive locations and the active `config.yaml`) unconditionally,
independent of `approvals.mode`; see `get_write_denied_error()` in
`tools/file_operations.py`). Two stores hold the allowlist state:

1. Shell-command allowlist: `hermes config set command_allowlist '[]'`
2. Shell-hook consent (only if present): `rm -f ~/.hermes/shell-hooks-allowlist.json`

Then sanity-check `hermes config get approvals.mode` (should not be `off`)
and confirm `--yolo` isn't baked into their launch alias or systemd unit.

### Shell hooks allowlist

Some shell-hook integrations require explicit allowlisting before they fire. Managed via `~/.hermes/shell-hooks-allowlist.json` — prompted interactively the first time a hook wants to run.

### Disabling the web/browser/image-gen tools

To keep the model away from network or media tools entirely, open `hermes tools` and toggle per-platform. Takes effect on next session (`/reset`). See `references/configuration.md` for the toolset list.

