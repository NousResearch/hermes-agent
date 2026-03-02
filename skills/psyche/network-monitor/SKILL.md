---
name: network-monitor
description: Monitor Psyche decentralized AI training network — discover runs via Psyche API, look up checkpoints on HuggingFace (best-effort), check system status, and query on-chain state via CLI. No API keys required.
version: 2.0.0
author: Eren Karakus
license: MIT
metadata:
  hermes:
    tags: [Psyche, Nous Research, Decentralized AI, Solana, Training, HuggingFace, Mining Pool]
    related_skills: []
---

# Psyche Network Monitor

Monitor the Psyche decentralized AI training network built by Nous Research. Discover training runs, check system status, look up model checkpoints, and query on-chain state.

**Zero dependencies. No API keys. Uses curl + python3 + existing Hermes tools.**

## What is Psyche?

Psyche is a decentralized AI model training platform coordinated on the Solana blockchain. It distributes training across independent GPU nodes worldwide using DisTrO (distributed training optimization with ~3x bandwidth reduction) and Iroh-based P2P networking. Anyone can contribute compute or funds to training runs.

## Scope

**This skill reports runs tracked by psyche.network by default.**

The Psyche API uses an allowlist filter, so not all on-chain runs may appear (e.g., users' private runs, test runs). For full on-chain coverage of a specific run, the Psyche CLI (`run-manager`) is required with a known run-id.

Always communicate this scope to the user when reporting results.

## When to Use

This skill should be loaded when:
- User asks about Psyche Network, its training runs, or models
- User wants to check the status of a Psyche training run
- User asks about mining pool contributions or how to participate
- User wants to find or download Psyche model checkpoints

## Quick Reference

| Action | Method | Source Label |
|--------|--------|-------------|
| List tracked runs | `curl -s "https://psyche.network/api/runs"` | `[tracked]` |
| System status | `curl -s "https://psyche.network/api/status"` | `[tracked]` |
| Checkpoint search (narrow) | `curl -s "https://huggingface.co/api/models?search=<run-id>&author=PsycheFoundation"` | `[best-effort]` |
| Checkpoint search (broad) | `curl -s "https://huggingface.co/api/models?search=<run-id>"` | `[best-effort]` |
| On-chain run detail | `run-manager json-dump-run --run-id <ID>` | `[on-chain]` |

For detailed command syntax and parsing examples, see `references/commands.md`.
For run state definitions, see `references/run-states.md`.

## Source Labels

Always tag each piece of data with its source to prevent false certainty:

| Label | Source | Reliability |
|-------|--------|------------|
| `[tracked]` | psyche.network/api/runs | High — official dashboard data |
| `[tracked:not-found]` | psyche.network/api/runs (run absent from list) | High — confirmed not in tracked set; may exist on-chain |
| `[on-chain]` | Psyche CLI (json-dump-run) | High — direct blockchain data |
| `[best-effort]` | HuggingFace search | Medium — checkpoint may not be found, false matches possible |
| `[unverified]` | web_search | Low — may be outdated, unverified |

Examples:
- "moe-10b-a1b-8k-wsd-lr3e4-1t: active, 1T tokens `[tracked]`"
- "Checkpoint: PsycheFoundation/moe-10b on HuggingFace `[best-effort]`"
- "This run is not in the tracked list `[tracked:not-found]`. If you know the run-id, on-chain query via CLI is available."

## Procedure

Follow this workflow. Do not hardcode model names or run IDs — always discover dynamically.

### Step 1: Run Discovery

Query the Psyche API to list all tracked training runs:

```bash
curl -s "https://psyche.network/api/runs" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
runs = data.get('runs', [])
print(f'{\"Run ID\":40s}  {\"Status\":20s}  {\"Arch\":15s}')
print('-' * 78)
for r in runs:
    status = r.get('status', {})
    status_str = status.get('type', '?') if isinstance(status, dict) else str(status)
    print(f\"{r.get('id','?'):40s}  {status_str:20s}  {r.get('arch','?'):15s}\")
print(f'\nTotal: {len(runs)} tracked runs')
"
```

**Response shape:** `{"runs": [...], "totalTokens": "...", "totalTokensPerSecondActive": "...", "error": null}`.
Each run's `status` is an object like `{"type": "active"}` or `{"type": "completed", "at": {...}}` — extract `status.type` for the state string.

This returns runs tracked by the Psyche dashboard. **Not all on-chain runs appear here** due to the allowlist filter.

### Step 2: System Status (Optional)

Check overall Psyche network health:

```bash
curl -s "https://psyche.network/api/status" | \
  python3 -c "
import sys, json
s = json.load(sys.stdin)
for k, v in s.items():
    print(f'{k}: {v}')
"
```

### Step 3: Run Details

From the Step 1 response, extract details for a specific run of interest. The `/api/runs` response wraps runs in `{"runs": [...]}`. Per-run fields include: `id`, `name`, `description`, `status` (object: `{"type": "active"}`, `{"type": "completed", "at": {...}}`, etc.), `arch`, `totalTokens`, `size`, `lastUpdate`, `trainingStep`.

```bash
curl -s "https://psyche.network/api/runs" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
runs = data.get('runs', [])
target = '<RUN_ID>'
match = [r for r in runs if r.get('id') == target or r.get('name') == target]
if match:
    print(json.dumps(match[0], indent=2))
else:
    print(f'Run \"{target}\" not found in tracked list.')
    print('It may be a private/test run or outside the allowlist.')
    print('If you have the run-id, try: run-manager json-dump-run --run-id <ID>')
"
```

**If API returns 200 but the run is not in the list**, tell the user:
"This run is not in the tracked list `[tracked:not-found]`. It may be a private/test run or outside the allowlist. If you know the run-id, on-chain query via CLI is available."

### Step 4: Checkpoint Lookup

The `/api/runs` endpoint does **not** include checkpoint type information. Checkpoint types (Hub, P2P, Gcs, P2PGcs, Ephemeral) are only available on-chain. Use the following best-effort strategy to find checkpoints:

#### 4a. Narrow Search (PsycheFoundation org)

Search with run-id first (always present), then run-name as fallback (name may be empty):

```bash
# Search by run-id (primary key)
curl -s "https://huggingface.co/api/models?search=<run-id>&author=PsycheFoundation"

# If no results, search by run-name (secondary key)
curl -s "https://huggingface.co/api/models?search=<run-name>&author=PsycheFoundation"
```

#### 4b. Broad Search (org-unrestricted fallback)

If the narrow search returns no results, some checkpoints may be uploaded to individual users' HuggingFace accounts rather than PsycheFoundation:

```bash
# Broad search by run-id
curl -s "https://huggingface.co/api/models?search=<run-id>"

# If no results, broad search by run-name
curl -s "https://huggingface.co/api/models?search=<run-name>"
```

If results are found outside PsycheFoundation, clearly indicate this to the user:
"Checkpoint found under `<user>/<model>` (not PsycheFoundation) `[best-effort]`"

**Multiple matches:** The broad search may return multiple results. In this case:
- Prefer exact name matches
- If multiple candidates remain, list all and let the user choose
- Tag all results with `[best-effort]`

#### 4c. CLI Checkpoint Backend (if run-id is known and CLI is available)

```bash
run-manager json-dump-run \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID>
```

This reveals the checkpoint backend type (Hub, P2P, Gcs, P2PGcs, Ephemeral) from on-chain data. Tag results with `[on-chain]`.

#### 4d. Not Found

If no checkpoint is found through any method, tell the user:
"This run's checkpoints are not visible on HuggingFace. Checkpoints may be stored via P2P, GCS, or may be ephemeral. Check the Psyche CLI or dashboard for details."

**Never say "checkpoint not found = run doesn't exist or is broken."**

### Step 5: On-Chain Deep Dive (Optional)

If the user has the Psyche CLI installed and knows the run-id:

```bash
# Full on-chain state dump
run-manager json-dump-run \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID>

# User participation data
run-manager json-dump-user \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID> \
  --address <SOLANA_PUBKEY>
```

Tag results with `[on-chain]`.

**Note:** Chain Mode requires a known run-id. The CLI cannot discover runs — there is no `run-manager list` command.

## Fallback Strategy

Fallback order depends on context. The CLI requires a run-id, so it cannot help with run discovery.

**Important:** The API may return 200 but not list a specific run (due to allowlist filtering). This is not the same as "API unreachable" but the same fallback applies.

### Run Discovery (run-id unknown)

```
psyche.network/api/runs (primary, tracked runs)
    | if unreachable OR run not in list
web_search (last resort, non-deterministic) [unverified]
```

### Run Detail (run-id known)

```
psyche.network/api/runs (primary, tracked runs)
    | if unreachable OR run not in list
Psyche CLI: run-manager json-dump-run [on-chain]
    | if CLI not installed
web_search (last resort, non-deterministic) [unverified]
```

**Why not Solana RPC as a fallback?** `getProgramAccounts` and `getAccountInfo` return raw Borsh-serialized binary data. PDA derivation (run-id to coordinator account) and Borsh deserialization are not practically feasible for an AI agent. Solana RPC commands are documented in `references/commands.md` as a reference only, not as a practical fallback.

## Participation Guide

### Mining Pool (Financial Contribution)

1. Install a Solana wallet (Phantom, Solflare, or `solana-keygen new`)
2. Navigate to https://psyche.network
3. Connect wallet and deposit SOL to the training pool
4. Pool capacity is limited — if full, check back periodically

### Compute Contribution (GPU)

Requires NVIDIA GPU with sufficient VRAM:
- Minimum: RTX 4090 (24 GB)
- Recommended: A100 (40/80 GB), H100 (80 GB)

Setup: https://docs.psyche.network

### Code Contribution

- Psyche core: https://github.com/PsycheFoundation/psyche (Rust + TypeScript)
- Hermes Agent: https://github.com/NousResearch/hermes-agent
- Atropos RL: https://github.com/NousResearch/atropos

## Key Links

| Resource | URL |
|----------|-----|
| Dashboard | https://psyche.network |
| Documentation | https://docs.psyche.network |
| GitHub | https://github.com/PsycheFoundation/psyche |
| Forum | https://forum.psyche.network |
| HuggingFace Org | https://huggingface.co/PsycheFoundation |
| Discord | https://discord.gg/NousResearch |

## Pitfalls

- **Do not hardcode model names or run IDs.** Always use Step 1 to discover current runs. New runs are added regularly.
- **HuggingFace does NOT equal all runs.** Only ~33% of tracked runs have HF checkpoints. A run existing on the Psyche API but not on HuggingFace is normal — checkpoints may use P2P, GCS, or be ephemeral.
- **API may return 200 but not list a run.** The backend uses an allowlist filter. Private, test, or untracked runs will not appear. This does not mean the run doesn't exist.
- **Checkpoint type varies per run.** Hub (HuggingFace), P2P, Gcs, P2PGcs, Ephemeral. The `/api/runs` endpoint does not expose checkpoint type — only on-chain data (via CLI) reveals this.
- **Mining pool capacity.** The pool is frequently full. If `psyche.network` shows the pool as full, advise the user to check back later.
- **Public Solana RPC rate limits.** The default `https://api.mainnet-beta.solana.com` has strict rate limits. For repeated queries, recommend a private RPC provider or set `$SOLANA_RPC_URL`.
- **Checkpoint size.** Large models (40B+) have checkpoint files in the tens of GB. Warn the user about download size before initiating.
- **CLI cannot discover runs.** There is no `run-manager list` command. Most CLI query subcommands require a known run-id.

## Verification

After completing any query, verify the results against these scenarios:

1. **Run found on API, checkpoint on HF under PsycheFoundation** — Narrow search finds it. Report with `[tracked]` for run, `[best-effort]` for checkpoint.
2. **Run found on API, checkpoint on HF under another user** — Broad search finds it. Report the non-PsycheFoundation source explicitly with `[best-effort]`.
3. **Run found on API, no checkpoint on HF** — This is normal. Report the run as `[tracked]` and note: "Checkpoints may be P2P, GCS, or ephemeral. Not all runs publish to HuggingFace."
4. **Run not found on API** — Report `[tracked:not-found]` and suggest CLI if run-id is known.
