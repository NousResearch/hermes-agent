# AgentCyber Standalone Runbook

This runbook defines the standalone runtime boundary for `/home/kbun/Desktop/hermes-agentcyber`.

AgentCyber is the Hermes Agent Cyber Edition for authorized defensive security work. It is not the default Hermes assistant, and it should not mutate the active default Hermes home, gateway, cron jobs, credentials, or profile config.

## Safety contract

- Use only for owned, approved, or lab-scoped security work.
- Keep sensitive cyber routing on local/open-weight models when `agent_cyber.routing.require_local_for_sensitive` is enabled.
- Keep `live_usb` disabled unless the operator explicitly approves that lane.
- Keep autonomous S5/destructive actions blocked unless an exact, scoped, unexpired break-glass approval exists.
- Do not use this runtime for public-target actions without recorded authorization and scope.
- Do not print or transfer secrets. Credentials stay in operator-approved stores only.

## Runtime boundary

The standalone boundary has two parts:

1. **Repo-local code path:** run Hermes from `/home/kbun/Desktop/hermes-agentcyber` with `uv run --frozen ...` so the Cyber Edition fork is used.
2. **Dedicated home:** set `HERMES_HOME` to an AgentCyber-only home so config, sessions, skills, memory, and cron state do not share default `~/.hermes`.

For this repair project, cron runs are constrained to write only inside the repo. The safe in-repo home for implementation and smoke checks is:

```bash
/home/kbun/Desktop/hermes-agentcyber/.agentcyber-home
```

A later human-approved install step may create an external convenience home such as `/home/kbun/.hermes-agentcyber` or a `~/bin/agentcyber` wrapper, but this cron repair run must not write those paths.

## Manual start commands

Preferred repo-local wrapper:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber setup --apply   # first run: writes only the dedicated AgentCyber home
scripts/agentcyber status --json
scripts/agentcyber                 # starts interactive AgentCyber chat from this checkout
```

The wrapper maps convenience commands as follows:

| Command | Runs inside the standalone boundary |
| --- | --- |
| `scripts/agentcyber` | `uv run --frozen hermes` |
| `scripts/agentcyber status --json` | `uv run --frozen hermes agentcyber status --json` |
| `scripts/agentcyber setup --apply` | `uv run --frozen hermes agentcyber setup --apply` |
| `scripts/agentcyber chat -q "..."` | `uv run --frozen hermes chat -q "..."` |
| `scripts/agentcyber hermes config path` | `uv run --frozen hermes config path` |

Safety checks in the wrapper:

- rejects `--profile` / `-p` so standalone runs cannot be redirected into a normal Hermes profile;
- rejects `AGENTCYBER_HOME=$HOME/.hermes` or a path under `$HOME/.hermes/` so setup cannot mutate default Hermes state;
- exports `HERMES_AGENTCYBER_STANDALONE=1`, which tells the CLI to ignore sticky `active_profile` redirects while a dedicated `HERMES_HOME` is set.

Equivalent explicit environment form:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
export HERMES_HOME=/home/kbun/Desktop/hermes-agentcyber/.agentcyber-home
export HERMES_AGENTCYBER_STANDALONE=1
uv run --frozen hermes agentcyber status
uv run --frozen hermes
```

Single-query mode:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber chat -q "Check AgentCyber status and summarize the cyber runtime boundary."
```

Status-only mode:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber status --json
```

## Required standalone config

The dedicated AgentCyber home should contain a `config.yaml` with at least:

```yaml
agent_cyber:
  routing:
    enabled: true
    require_local_for_sensitive: true
    allow_hosted_override: true
    allow_hosted_open_weight: false
    local_open_weight:
      provider: ollama
      model: qwen3-coder:30b
      base_url: http://192.168.1.120:11434/v1
      api_mode: chat_completions
      context_length: 131072
  include_builtin_bc_assets: true
  execution_gates:
    enabled: true

platform_toolsets:
  cli:
    - cyber

approvals:
  cron_mode: deny
```

Notes:

- Store secrets only in the dedicated AgentCyber `.env`, not in docs or command output.
- Keep `live_usb` disabled unless explicitly approved.
- Do not alter default `~/.hermes/config.yaml` to achieve AgentCyber behavior.

## Check status

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber status --json
```

Expected high-level fields:

- `agent_cyber.routing_enabled: true`
- `agent_cyber.require_local_for_sensitive: true`
- `agent_cyber.local_open_weight.provider: "ollama"`
- `agent_cyber.local_open_weight.model: "qwen3-coder:30b"`
- `local_runtime_health.ok: true` when the LAN Ollama host is reachable
- `toolsets.cyber_visible: true`
- `toolsets.cyber_enabled: true`
- `toolsets.live_usb_visible: true`
- `toolsets.live_usb_enabled: false`
- `assets.builtin_enabled: true`

## Verify local/open-weight model routing

Status command health check:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber status --json
```

Focused tests:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
uv run --frozen python -m pytest \
  tests/hermes_cli/test_agentcyber_wrapper.py \
  tests/agent/test_cyber_routing.py \
  tests/agent/test_agentcyber_routing_guard.py \
  tests/agent/test_cyber_breakglass.py \
  tests/hermes_cli/test_agentcyber_cmd.py \
  tests/gateway/test_cyber_audit_hook.py \
  -q -o addopts= --tb=short
```

## Verify cyber toolsets

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber hermes tools list | grep -Ei 'cyber|live_usb'
```

Expected:

```text
✓ enabled  cyber
✗ disabled live_usb
```

Exact spacing and icons may vary by terminal skin.

## Live USB operator approval

`status` and `list_usb` are read-only and do not need root or an approval token. The high-consequence actions `build`, `write`, and `provision` require both root and an operator-controlled approval token before any build script, block-device check, or USB-writing command runs.

AgentCyber `write` and `provision` tool calls also require the target to be verifiably removable via Linux block-device metadata; edge-case media that cannot be verified should be handled manually outside unattended tool control.

For an explicitly approved live USB maintenance session only, the operator can set a token in the standalone AgentCyber environment and provide the same value as `operator_approval` in the `live_usb` tool call:

```bash
export HERMES_AGENTCYBER_LIVE_USB_APPROVAL="<operator-approved one-time token>"
```

Do not print, log, or persist the token. Clear it after the approved operation. Cron repair runs must not set this token, run as root, build an ISO, or write/provision USB media.

## Asset scope

Built-in Breaking Circuits assets are expected to be enabled for the Cyber Edition runtime. Asset matching should recognize BC-owned domains and lab ranges without treating cyber terminology alone as suspicious.

Allowed-by-default examples:

- Passive analysis of BC-owned domains.
- Lab-scoped scan of an approved RFC1918 host in the BC lab range.
- Offline malware/sample triage inside the lab.

Requires clarification or refusal:

- Scanning arbitrary public IPs with no authorization statement.
- Credential retrieval or movement outside approved operator-controlled sources.
- Any external target not present in the asset registry.

## Break-glass approvals

Break-glass is operator-controlled, exact-action, scoped, expiring, and auditable. It does not grant broad autonomous permission.

See:

```text
docs/CYBER_BREAKGLASS_OPERATOR_WORKFLOW.md
```

Examples to verify CLI behavior and manage approvals from the standalone boundary:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber breakglass --help

STORE="$PWD/.agentcyber-home/agentcyber/breakglass.jsonl"
APPROVAL_JSON=$(scripts/agentcyber breakglass create \
  --tool terminal \
  --args-json '{"command":"printf '\''[DRY RUN] password reset 192.168.1.120\\n'\''"}' \
  --operator kbun \
  --reason 'owned lab recovery rehearsal' \
  --ttl-minutes 5 \
  --store "$STORE" \
  --apply \
  --json)
APPROVAL_ID=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["approval_id"])' <<<"$APPROVAL_JSON")

scripts/agentcyber breakglass list --store "$STORE"
scripts/agentcyber breakglass revoke "$APPROVAL_ID" --store "$STORE"
```

Do not create real break-glass approvals in an unattended cron run unless the user explicitly requested that exact action and scope. Prefer dry-run examples or temporary `AGENTCYBER_HOME` stores for rehearsal.

## How to keep AgentCyber separate from default Hermes

Do:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber status
```

Do not rely on:

```bash
hermes agentcyber status
```

without confirming which Hermes executable and `HERMES_HOME` are active.

Quick separation checks:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber hermes config path
scripts/agentcyber status --json
```

The config path should resolve under `.agentcyber-home`, not default `~/.hermes`.

## Stop/restart

For CLI sessions, exit with `/exit` or Ctrl-D.

Do not install, start, stop, or restart any AgentCyber gateway service from this cron repair lane. A separate service such as `hermes-agentcyber-gateway.service` can be proposed later after the CLI standalone runtime is verified and the user explicitly approves service installation.

## Current repair status

As of this repair lane, the repo-local AgentCyber command path and standalone wrapper exist. `scripts/agentcyber setup --apply` writes only the dedicated `.agentcyber-home` home, `scripts/agentcyber hermes config path` resolves under that home, and `scripts/agentcyber status --json` verifies routing, local runtime health, assets, and cyber toolset visibility from the wrapper boundary.
