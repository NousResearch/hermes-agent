# Hermes profile ↔ opencode serve integration

A Hermes profile that acts as engineering lead and delegates implementation
to a headless opencode server (`opencode serve` / `opencode web`) running on
an always-on host. The profile plans and reviews; opencode's agent loop
(read, edit, bash, git, LSP) executes on the remote host, so work survives
Hermes restarts and laptop disconnects.

## What was added

| File | Purpose |
|---|---|
| `tools/opencode_serve_tool.py` | `opencode_run` + `opencode_status` tools (HTTP client, session persistence) |
| `toolsets.py` | New `opencode` toolset entry |
| `optional-skills/opencode-serve/SKILL.md` | Profile skill: delegation workflow + guardrails |

## Prerequisites

1. An opencode server reachable from the Hermes box (e.g. the Proxmox VM with
   `opencode web --hostname <tailscale-ip> --port 4096` under systemd).
2. Basic auth configured on that server (`OPENCODE_SERVER_PASSWORD`).
3. The project repos cloned **on the server host** — `project` arguments are
   server-side paths.

## Setup

```bash
# 1. Create the profile (clone your active profile's config/env/skills)
hermes profile create coder --clone \
  --description "Engineering lead: plans coding work and delegates implementation to the opencode serve backend."

# 2. Install the skill from the repo (or copy manually:
#    optional-skills/opencode-serve → $HERMES_HOME/skills/coding/opencode-serve/)
hermes skills install opencode-serve

# 3. Add server config to the profile's .env (NOT tracked secrets, keep out of git)
#    $HERMES_HOME/.env:
#      OPENCODE_SERVER_URL=http://<vm-tailscale-ip>:4096
#      OPENCODE_SERVER_USERNAME=opencode
#      OPENCODE_SERVER_PASSWORD=<basic-auth-password>
#      OPENCODE_DEFAULT_PROJECT=/home/steve/repos/myapp   # optional convenience

# 4. Enable the toolset (or add 'opencode' under the profile's config.yaml
#    platform_toolsets list — see cli-config.yaml.example)
hermes tools enable opencode
```

## Verification

Start the profile and ask:

```
What's in /home/steve/repos/myapp? Use opencode_run to list the top-level files.
```

A blocking run returns the opencode agent's text plus per-file diff stats;
a `background=true` run must be polled with `opencode_status`.

## How it works

- `POST /session` creates a session per project; the ID persists under
  `$HERMES_HOME/data/opencode_sessions.json`, so follow-ups continue context.
- Blocking runs use `POST /session/:id/message` (waits for the reply, capped
  by `timeout_minutes`); background runs use `prompt_async` + polling.
- `GET /session/:id/diff` supplies the change summary included in run results.
- The tool is inert unless `OPENCODE_SERVER_URL` is set (`requires_env`).

## Caveats

- `project` paths are **server-side**. Local files are never touched.
- Serialize runs per project — the session store is a naive JSON file.
- `OPENCODE_SERVER_PASSWORD` is scrubbed from delegated child environments by
  design; keep it in the profile env only.
- opencode's own tool-permission prompts are server-side. Pre-approve via the
  server's config/permissions (`--auto` or explicit permission rules), or have
  the profile expect permission blocks and re-dispatch.
- Server-side sessions accumulate cost and context; `new_session=true` resets.
