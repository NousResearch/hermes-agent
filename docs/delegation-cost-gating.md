# Delegation Cost Gating

**Feature introduced in PR #64476.**

Hermes Agent supports delegating work to subagents, which can run on completely different
LLM providers than the parent agent. When subagents are configured to run on metered
API-key providers (OpenAI API, Anthropic API, custom endpoints), each subagent invocation
incurs a separate cost. In a loop-heavy or multi-agent workflow, this can multiply spend
quickly.

To prevent unbounded delegation costs, Hermes classifies delegation routes and applies
cost-gating to paid API-key routes.

## Route Classification

Delegation routes are classified into four categories:

| Route Type | Examples | Gated? | Rationale |
|------------|----------|--------|-----------|
| `local` | `http://127.0.0.1:8080/v1`, `http://localhost:11434`, custom providers on loopback | No | Free or self-hosted; delegation cost is negligible |
| `delegate` | `openai-codex` at `https://chatgpt.com/backend-api/codex` | No | Paid via ChatGPT Plus subscription; users have opted in at platform level |
| `copilot` | `copilot` at `https://api.githubcopilot.com` | No | Paid via GitHub Copilot subscription; delegation cost is negligible |
| `api_key` | OpenAI API, Anthropic API, custom endpoints with non-loopback base_url | **Yes** | Metered per-call; must be gated to prevent unbounded spend |

## Default Behavior (Fail-Closed)

**Paid API-key routes are blocked by default.** This is intentional: the cost gate is
a safety feature, and the safe default is to deny unknown delegation targets.

To enable delegation to paid API-key routes, you must either:

1. **Configure the logger script** (recommended): Set `AI_DEV_USAGE_ENFORCE=1` and
   `AI_DEV_USAGE_LOGGER` to point to an executable script that can approve/deny
   delegation requests.
2. **Bypass the gate** (not recommended): Set `AI_DEV_USAGE_ALLOW_ROUTE=1` to allow
   all api_key routes without external approval.

## Logger Script Contract

When `AI_DEV_USAGE_ENFORCE=1`, Hermes runs the logger script specified by
`AI_DEV_USAGE_LOGGER` before each delegation. The script receives one argument:
the route classification (`local`, `delegate`, `copilot`, or `api_key`).

### Exit Code

The logger script's exit code determines whether delegation proceeds:

| Exit Code | Meaning | Effect |
|-----------|---------|--------|
| `0` | Delegation approved | Gate opens; subagent spawns normally |
| Non-zero | Delegation denied | Gate stays closed; error returned to caller |

### Example Logger Scripts

#### Minimal Allow-All Script

```bash
#!/usr/bin/env bash
# Allow all delegation routes (equivalent to AI_DEV_USAGE_ALLOW_ROUTE=1)
exit 0
```

#### Rate-Limited Script (Shell)

```bash
#!/usr/bin/env bash
# Allow up to 10 api_key delegations per hour
ROUTE=$1
STATE_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/hermes-delegation-count.txt"

# Only enforce limits on api_key routes
if [[ "$ROUTE" != "api_key" ]]; then
    exit 0
fi

# Ensure state directory exists
mkdir -p "$(dirname "$STATE_FILE")"

# Read current count and timestamp
if [[ -f "$STATE_FILE" ]]; then
    IFS= read -r count timestamp < "$STATE_FILE"
else
    count=0
    timestamp=0
fi

now=$(date +%s)
one_hour_ago=$((now - 3600))

# Reset if older than an hour
if [[ "$timestamp" -lt "$one_hour_ago" ]]; then
    count=0
    timestamp="$now"
fi

# Check limit
if [[ "$count" -ge 10 ]]; then
    echo "Rate limit exceeded: $count delegations in the last hour (max 10)" >&2
    exit 1
fi

# Increment and save
echo "$((count + 1)) $now" > "$STATE_FILE"
exit 0
```

#### Python Script with Cost Cap

```python
#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timedelta

ROUTE = sys.argv[1] if len(sys.argv) > 1 else "unknown"
STATE_FILE = os.path.expanduser("~/.local/state/hermes-delegation-cost.txt")

# Only enforce on api_key routes
if ROUTE != "api_key":
    sys.exit(0)

HOURLY_COST_CAP_USD = 1.0  # $1/hour for delegation

if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        lines = f.read().strip().splitlines()
        total_cost = sum(float(line.split()[0]) for line in lines if line.strip())
else:
    total_cost = 0.0

if total_cost >= HOURLY_COST_CAP_USD:
    print(f"Cost cap exceeded: ${total_cost:.2f} in the last hour (max ${HOURLY_COST_CAP_USD:.2f})", file=sys.stderr)
    sys.exit(1)

# Approve this delegation; caller should record actual cost after completion
sys.exit(0)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_DEV_USAGE_ENFORCE` | No (default: unset) | Set to `1` to enable cost gating via logger script. |
| `AI_DEV_USAGE_LOGGER` | Yes if `AI_DEV_USAGE_ENFORCE=1` | Absolute path to the logger script. Must be executable. |
| `AI_DEV_USAGE_ALLOW_ROUTE` | No (default: unset) | Set to `1` to bypass the gate and allow all api_key routes. Not recommended. |

## Design Notes

### Why Fail-Closed by Default?

A cost gate that fails open is worse than no gate at all: users think they're protected
but aren't. The only safe default for a cost control feature is to deny unknown routes
until explicitly approved.

### Why No `allow_variable_cost` Tool Argument?

In earlier versions of PR #64476, `allow_variable_cost` was a model-controllable tool
argument. This allowed the model to approve its own paid routes by passing
`allow_variable_cost=True`, defeating the purpose of the gate.

The approval signal must come from outside the model: either a user-configured logger
script or an environment variable set by the operator.

### Why Local/OAuth Routes Are Not Gated?

* **Local routes** (loopback) are free or self-hosted; delegation cost is negligible.
* **OAuth routes** (ChatGPT Plus, GitHub Copilot) are paid via subscription at the
  platform level, not per-delegation. Users have already opted in to the provider's
  pricing model.

Gating these would add friction without meaningful cost protection.

## Troubleshooting

### "Delegation blocked for api_key route"

This error means:
1. The delegation target is a paid API-key route.
2. The logger script either doesn't exist, denied approval, or timed out.
3. `AI_DEV_USAGE_ALLOW_ROUTE` is not set.

**Fix:** Either configure the logger script or set `AI_DEV_USAGE_ALLOW_ROUTE=1`
(not recommended for production).

### Logger script times out

The logger script has a 5-second timeout. If your script is slow, it will be treated
as a denial for safety.

**Fix:** Optimize your logger script to run in under 5 seconds.