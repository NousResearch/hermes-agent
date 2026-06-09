# Gateway status receipt (M5)

> Read-only capture. **No restart performed.**  
> Profile: `auto-coder` | HERMES_HOME: `/home/khall/.hermes/profiles/auto-coder`  
> Captured: 2026-06-09 (local session, ~4h uptime on gateway)

## Verified

| Field | Value |
|-------|--------|
| Owner profile | `auto-coder` |
| Service unit | `hermes-gateway-auto-coder.service` (user systemd) |
| Unit path | `~/.config/systemd/user/hermes-gateway-auto-coder.service` |
| State | **active (running)** |
| Main PID | `1048070` |
| Exec (live process) | `.../python -m hermes_cli.main --profile auto-coder gateway run **--replace**` |
| Exec (unit file) | `... gateway run` (**no `--replace`**) |
| WorkingDirectory | `/home/khall/.hermes/profiles/auto-coder` |
| HERMES_HOME (unit env) | `/home/khall/.hermes/profiles/auto-coder` |
| Python | `/home/khall/.hermes/hermes-agent/venv/bin/python` |
| Linger | enabled |
| MCP children | devbrain, context7, playwright, codegraph (from cgroup) |

## Config-only / drift

- **Unit vs runtime:** Running process includes `--replace`; installed unit file does not. After `systemctl --user restart`, new process may **drop** `--replace` unless unit is refreshed.
- M2b previously reported “installed gateway service definition is outdated”; this capture still shows the **ExecStart mismatch** above.

## Other gateway-capable profiles

| Profile | Gateway state (profile list) | Policy |
|---------|------------------------------|--------|
| `auto-coder` | running | **sole owner** |
| `gateway-minimal` | stopped | must stay stopped |
| all others | stopped | OK |

## Single-owner check

**Pass:** exactly one running Hermes gateway service tied to `auto-coder`.

## Evidence command

```bash
HERMES_HOME=~/.hermes/profiles/auto-coder hermes gateway status
```