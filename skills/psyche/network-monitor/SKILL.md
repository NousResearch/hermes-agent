---
name: network-monitor
description: Monitor Psyche decentralized AI training network — discover runs, track checkpoints, check mining pool status, and query on-chain state. No API keys required.
version: 1.0.0
author: Eren Karakus
license: MIT
metadata:
  hermes:
    tags: [Psyche, Nous Research, Decentralized AI, Solana, Training, HuggingFace, Mining Pool]
    related_skills: []
---

# Psyche Network Monitor

Monitor the Psyche decentralized AI training network built by Nous Research. Dynamically discover training runs, track model checkpoints, check mining pool status, and query on-chain state.

**Zero dependencies. No API keys. Uses curl + python3 + existing Hermes tools.**

## What is Psyche?

Psyche is a decentralized AI model training platform coordinated on the Solana blockchain. It distributes training across independent GPU nodes worldwide using DisTrO (distributed training optimization with ~3x bandwidth reduction) and Iroh-based P2P networking. Anyone can contribute compute or funds to training runs.

## When to Use

This skill should be loaded when:
- User asks about Psyche Network, its training runs, or models
- User wants to check the status of a Psyche training run
- User asks about mining pool contributions or how to participate
- User wants to find or download Psyche model checkpoints from HuggingFace

## Quick Reference

| Action | Command |
|--------|---------|
| List all runs/models | `curl -s "https://huggingface.co/api/models?author=PsycheFoundation"` |
| Get model details | `curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>"` |
| List checkpoint files | `curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>/tree/main"` |
| On-chain run state | `run-manager json-dump-run --rpc <RPC_URL> --run-id <RUN_ID>` |

For detailed command syntax and parsing examples, see `references/commands.md`.
For run state definitions, see `references/run-states.md`.

## Procedure

Follow this multi-run discovery workflow. Do not hardcode model names — always discover dynamically.

### Step 1: Discover All Runs

Query the HuggingFace API to list all models under the PsycheFoundation organization:

```bash
curl -s "https://huggingface.co/api/models?author=PsycheFoundation" | \
  python3 -c "
import sys, json
models = json.load(sys.stdin)
for m in sorted(models, key=lambda x: x.get('createdAt',''), reverse=True):
    print(f\"{m['id']:55s}  created: {m.get('createdAt','N/A')[:10]}  downloads: {m.get('downloads',0)}\")
"
```

This returns all training run models sorted by creation date. To get the last modification date for a specific model, use Step 2.

### Step 2: Get Run Details

Once you identify a model of interest, fetch its full metadata:

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>" | \
  python3 -c "
import sys, json
m = json.load(sys.stdin)
print(f\"Model: {m['id']}\")
print(f\"Tags: {', '.join(m.get('tags', []))}\")
print(f\"Downloads: {m.get('downloads', 0)}\")
print(f\"Last Modified: {m.get('lastModified', 'N/A')}\")
siblings = m.get('siblings', [])
print(f\"Files: {len(siblings)}\")
for s in siblings[:15]:
    print(f\"  {s['rfilename']}\")
if len(siblings) > 15:
    print(f\"  ... and {len(siblings)-15} more files\")
"
```

### Step 3: List Checkpoint Files

To see all checkpoint files for a model (useful for downloading specific training snapshots):

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>/tree/main" | \
  python3 -c "
import sys, json
files = json.load(sys.stdin)
for f in files:
    size_mb = f.get('size', 0) / 1024 / 1024
    print(f\"{f['path']:50s}  {size_mb:>8.1f} MB\")
"
```

### Step 4: On-Chain State (Optional)

If the user has the Psyche CLI installed or wants deeper on-chain data, query Solana directly:

```bash
# Requires: psyche-solana-client (from https://github.com/PsycheFoundation/psyche)
run-manager json-dump-run --rpc "$SOLANA_RPC_URL" --run-id <RUN_ID>
```

If the CLI is not available, use the Solana JSON-RPC API as a fallback (see `references/commands.md`).

## Fallback Strategy

If the primary data source fails, follow this fallback chain:

1. **HuggingFace API** (primary) — Most reliable for model/checkpoint data
2. **Solana RPC** (secondary) — For on-chain run state and participant data
3. **Psyche CLI** (`run-manager`) — If installed locally, provides the richest run data
4. **Web search** — Last resort for general Psyche network status

If a source returns an error:
- HuggingFace 404 → The model ID may have changed. Re-run Step 1 to discover current models.
- HuggingFace timeout → Retry once, then fall back to `web_search` for cached info.
- Solana RPC timeout → Suggest the user set `$SOLANA_RPC_URL` to a private RPC endpoint.

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
| Forum | https://forum.nousresearch.com |
| HuggingFace Org | https://huggingface.co/PsycheFoundation |
| Discord | https://discord.gg/NousResearch |

## Pitfalls

- **Do not hardcode model names.** Always use Step 1 to discover current models. New runs and models are added regularly.
- **Mining pool capacity.** The pool is frequently full. If `psyche.network` shows the pool as full, advise the user to check back later.
- **Public Solana RPC rate limits.** The default `https://api.mainnet-beta.solana.com` has strict rate limits. For repeated queries, recommend a private RPC provider or set `$SOLANA_RPC_URL`.
- **Checkpoint size.** Large models (40B+) have checkpoint files in the tens of GB. Warn the user about download size before initiating.
- **Testnet vs Mainnet.** Some runs may be on Solana devnet. Check the run documentation or dashboard for the correct network.

## Verification

After completing any query, verify the results:
- Model listing returns a JSON array with `id` fields → success
- Model details contain `siblings` (file list) → model exists and has uploaded checkpoints
- If zero models returned, re-check the API URL or use `web_search "PsycheFoundation HuggingFace"` as fallback
