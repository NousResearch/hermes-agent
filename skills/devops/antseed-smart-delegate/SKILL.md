---
name: antseed-smart-delegate
description: "Use when delegating tasks through AntSeed P2P network or when delegate_task with provider=antseed fails. Auto-selects optimal peer+model by task type, tracks spending, provides fallback chain on failure."
version: 1.0.0
author: Hermes Agent (ryptotalent)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [antseed, p2p, delegation, smart-routing, cost-tracking, fallback, peer-selection]
    related_skills: []
---

# AntSeed Smart Delegate

## Overview

Smart routing layer for `delegate_task` through the AntSeed P2P AI services network. Instead of blindly pinning a peer and hoping for the best, this skill runs a **preflight → rank → delegate → report** pipeline that:

1. **Preflight** — checks buyer proxy health, pinned peer, wallet deposits, active channels
2. **Rank** — scans all reachable peers, scores models against task type (code/research/vision/chat/cheap), returns best match + 5 fallbacks
3. **Delegate** — calls `delegate_task` with the optimal `provider:antseed`, `model:<from-rank>`
4. **Report** — shows spending, channel state, token usage after each delegation

The three shell scripts (`scripts/preflight.sh`, `scripts/best-peer.sh`, `scripts/cost-report.sh`) output **parseable JSON on stdout** (for programmatic use) and **human-readable text on stderr** (for chat display). Exit codes: 0 = ok, 1 = warning, 2 = error.

## When to Use

- User asks to delegate a task via AntSeed (`delegate_task` with `provider: antseed`)
- Previous AntSeed delegation failed with 502/timeout/402/insufficient_deposits
- User wants to know "which peer is best right now" for a specific task type
- User asks "how much did we spend on AntSeed" after a delegation session
- Setting up AntSeed on a new machine and need to verify the full pipeline

**Don't use for:**
- Direct LLM calls through Z.AI / OpenAI / Anthropic (use normal delegate_task)
- Running an AntSeed seller node (see AntSeed docs for seller setup)
- Tasks that don't need model inference (file ops, web search, etc.)

## Prerequisites

1. **AntSeed CLI installed:** `npm install -g @antseed/cli`
2. **Buyer proxy running:** `antseed buyer start` (listens on `localhost:8377`)
3. **Peer pinned:** `antseed buyer connection set --peer <peer-id>`
4. **Wallet funded:** `antseed buyer deposit <amount>` (min ~$0.10–$1.00 depending on peer)
5. **Hermes configured** with AntSeed as custom provider in `config.yaml`:

```yaml
custom_providers:
  - name: antseed
    base_url: http://127.0.0.1:8377/v1
    api_key: antseed-p2p
    models:
      - id: auto
        name: Auto (from best-peer.sh)
        reasoning: false
        input: text
        output: text
        context_window: 200000
        max_output: 16384
```

## Quick Start

```bash
# 1. Health check before delegating
bash skills/devops/antseed-smart-delegate/scripts/preflight.sh

# 2. Find best peer+model for your task type
bash skills/devops/antseed-smart-delegate/scripts/best-peer.sh code       # coding tasks
bash skills/devops/antseed-smart-delegate/scripts/best-peer.sh research   # analysis/deep-thinking
bash skills/devops/antseed-smart-delegate/scripts/best-peer.sh vision     # image/multimodal
bash skills/devops/antseed-smart-delegate/scripts/best-peer.sh chat       # general conversation
bash skills/devops/antseed-smart-delegate/scripts/best-peer.sh cheap      # minimum cost

# 3. After delegating — check spending
bash skills/devops/antseed-smart-delegate/scripts/cost-report.sh          # human-readable
bash skills/devops/antseed-smart-delegate/scripts/cost-report.sh --json  # parseable JSON
```

## Step-by-Step Workflow

### Step 1 — Preflight Check

Always run before delegating. Returns structured JSON with system state:

```bash
bash scripts/preflight.sh
```

**JSON output (stdout):**
```json
{
  "ok": true,
  "proxy_up": true,
  "proxy_http": "200",
  "peer_pinned": "f629c1f973961917...",
  "peer_name": "Antseed ZAI Provider",
  "deposits_usdc": 1.0,
  "reserved_usdc": 0.0,
  "wallet": "0x411B...ca58",
  "active_channels": 1,
  "issues": [],
  "can_delegate": true
}
```

**Human output (stderr):**
```
Ready to delegate via AntSeed
   Peer: Antseed ZAI Provider (f629c1f97396...)
   Deposits: 1.0 USDC available (0.0 reserved)
   Channels: 1 active
```

**If `can_delegate: false`**, check `issues[]`:

| Issue | Cause | Fix |
|-------|-------|-----|
| `proxy_down` | Buyer not running | `antseed buyer start` |
| `no_peer` | No peer pinned | Run `best-peer.sh` → pick → `antseed buyer connection set --peer <id>` |
| `no_funds` | Deposits = 0 | `antseed buyer deposit <amount>` |

### Step 2 — Select Best Peer & Model

```bash
bash scripts/best-peer.sh <task_type>
```

Task types: `code`, `research`, `vision`, `chat`, `cheap`, `any`

**JSON output:**
```json
{
  "task_type": "code",
  "recommended": {
    "peer_id": "f629c1f9739619...",
    "peer_name": "Antseed ZAI Provider",
    "model": "qwen/qwen3-next-80b-a3b-thinking",
    "price_in": "$0.00701/1M",
    "price_out": "$0.01501/1M",
    "protocol": "openai-chat-completions",
    "tags": ["chat", "coding", "reasoning"],
    "free": false,
    "score": 8.5
  },
  "fallback_chain": [
    {"peer_id": "...", "model": "deepseek-v4-flash", "score": 7.2},
    {"peer_id": "...", "model": "minimax-m2.5", "score": 6.8},
    ...
  ],
  "total_candidates": 342,
  "unique_peers": 12
}
```

**Scoring logic** (in `best-peer.sh`):
1. Free models score highest (cost = 0)
2. Tag match bonus: +3 if task tags ⊂ model tags
3. Protocol preference: `openai-chat-completions` > others (avoids streaming-required protocols)
4. Price penalty: cheaper = higher score
5. Top result recommended, next 5 become fallback chain

### Step 3 — Delegate

Use the recommended model from Step 2:

```
delegate_task(
  provider="antseed",
  model="qwen/qwen3-next-80b-a3b-thinking",  # from best-peer.sh
  goal="Refactor the auth module to extract TokenValidator service"
)
```

**On failure**, iterate through `fallback_chain` from best-peer output:

| Error | Action |
|-------|--------|
| 502 / timeout | Try next peer in fallback_chain (max 3 retries) |
| 402 / insufficient_deposits | Alert user — needs more funds |
| 400 "model not found" | Re-run `best-peer.sh` (model catalog drift) |

### Step 4 — Cost Report

```bash
bash scripts/cost-report.sh         # human-readable table
bash scripts/cost-report.sh --json  # parseable JSON
```

**Text output:**
```
AntSeed Cost Report
============================
Wallet:     0x411B38A9...4bca58
Deposits:   0.95 USDC (0.05 reserved)
Available:  ~0.90 USDC for new channels

Channel(s) (Antseed ZAI Provider):
  Status:    active
  Requests:  3
  Peer:      f629c1f973961917...

Spending (session):
  Est. spent: ~0.003 USDC
  Tokens in: ~2500
```

## Complete Example Session

```
User: "отрефакторь модуль авторизации"

Agent: [loads antseed-smart-delegate skill]

Step 1 — Preflight:
  $ bash preflight.sh
  -> {ok: true, deposits: 1.0, peer: "Antseed ZAI Provider"}

Step 2 — Best peer for "code":
  $ bash best-peer.sh code
  -> {model: "qwen/qwen3-next-80b-a3b-thinking",
     price: "$0.007/$0.015", peers: 12, candidates: 342}

Step 3 — Delegate:
  $ delegate_task(provider=antseed,
                   model=qwen/qwen3-next-80b-a3b-thinking,
                   goal="Refactor auth module...")
  -> Success: extracted TokenValidator service, 15 files changed

Step 4 — Cost:
  $ bash cost-report.sh
  -> Spent ~$0.003, 2.5k tokens, channel active
```

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/preflight.sh` | Health check before delegation | none | JSON: {ok, proxy, peer, deposits, issues} |
| `scripts/best-peer.sh` | Rank peers+models by task type | task_type (code\|research\|vision\|chat\|cheap\|any) | JSON: {recommended, fallback_chain, total_candidates} |
| `scripts/cost-report.sh` | Spending & channel status | [--json] | Text table or JSON |

All scripts require `antseed` CLI in PATH and buyer proxy on `localhost:8377`.

## Common Pitfalls

### 1. Unicode Table Parsing

AntSeed CLI uses **Unicode box-drawing characters** (`│` U+2502) in table output, not ASCII pipe `|`. The scripts parse tables using Python's `str.split('\u2502')`. The value is always in `parts[2]` (after the 2nd `│`). Do not use `awk -F"|"` or `grep -oP` with `$` signs — both break on this output format.

### 2. Protocol Matters: openai-responses vs chat_completions

Some peers (e.g., Dark Signal) use `openai-responses` protocol which **requires streaming**. Hermes auxiliary calls (compression, title generation) are non-streaming and will get HTTP 400. **Fix:** `best-peer.sh` prefers `openai-chat-completions` protocol by default. For auxiliary slots, use `provider: auto`.

### 3. Reserve Ceiling ≠ Per-Token Price

A model may cost $0.001/1M tokens but the peer may require **$1.00 reserve** to open a payment channel. Your wallet must have ≥ the reserve amount. `preflight.sh` reports available deposits but cannot see the peer's reserve requirement until first delegation attempt.

### 4. Peer Pin is Session-Only

When the buyer proxy restarts, the pinned peer is reset. `preflight.sh` detects `no_peer` in this case. **Fix:** Re-pin or set up a systemd service with auto-pin on boot.

### 5. Model Catalog Drift

Peers can add/remove models at any time. The model recommended by `best-peer.sh` may disappear by the time you delegate. If you get 400 "model not found", re-run `best-peer.sh` for fresh data.

### 6. Subshell Variable Scoping in Bash

`while read` loops inside `$(...)` command substitutions run in subshells and **cannot propagate variables to the parent scope**. The scripts use temp files (`mktemp -d`) instead of pipes for collecting peer data. Do not refactor to pipe-based collection without testing.

### 7. `$` Sign in Grep/Awk Patterns

Prices contain literal `$` (e.g., `$0.007/1M`). In bash, `grep -E 'in \$[0-9]'` inside `$(...)` strips the backslash. Use `[[ "$var" == *'$'* ]]` glob matching or `awk` field extraction instead.

## Verification Checklist

- [ ] File at `skills/devops/antseed-smart-delegate/SKILL.md`
- [ ] Frontmatter starts at byte 0 with `---`, has name/description/version/author/license/platforms/metadata
- [ ] Description ≤ 1024 chars, starts with "Use when ..."
- [ ] Total SKILL.md ≤ 100,000 chars (aim: 8–15k)
- [ ] `scripts/preflight.sh` is executable, passes `bash -n`, outputs valid JSON
- [ ] `scripts/best-peer.sh` is executable, passes `bash -n`, outputs valid JSON with fallback_chain
- [ ] `scripts/cost-report.sh` is executable, passes `bash -n`, outputs text + --json mode
- [ ] All scripts handle missing antseed CLI gracefully (exit 2, not crash)
- [ ] `related_skills` only references in-repo skills (or empty)
- [ ] No hardcoded wallet addresses, server IPs, or user-specific values in SKILL.md body
