# Psyche Network Monitor — Command Reference

Detailed command syntax and parsing examples for querying Psyche network data.

---

## 1. Psyche API (Primary)

The Psyche dashboard API is the primary data source for run discovery. All endpoints are public, no authentication required.

### List Tracked Runs

```bash
curl -s "https://psyche.network/api/runs"
```

**Response:** JSON object `{"runs": [...], "totalTokens": "...", "totalTokensPerSecondActive": "...", "error": null}`.

Each run element contains:
- `id` — Run identifier (string)
- `name` — Human-readable run name (string, may be empty)
- `description` — Run description (string)
- `status` — Status object: `{"type": "active"}`, `{"type": "paused"}`, `{"type": "completed", "at": {...}}`, `{"type": "waitingForMembers"}`. Extract `status.type` for the state string.
- `arch` — Model architecture (HfDeepseek, HfLlama, HfAuto, Torchtitan)
- `totalTokens` — Total tokens for training
- `size` — Model size
- `lastUpdate` — Last update timestamp
- `trainingStep` — Current training step

**Note:** This endpoint uses an allowlist filter. Not all on-chain runs appear. Checkpoint type is **not** included.

**Parse to table:**

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

### System Status

```bash
curl -s "https://psyche.network/api/status"
```

**Response:** JSON object with system-level information including Coordinator Program ID and Mining Pool Program ID.

**Parse:**

```bash
curl -s "https://psyche.network/api/status" | \
  python3 -c "
import sys, json
s = json.load(sys.stdin)
for k, v in s.items():
    print(f'{k}: {v}')
"
```

### Find Specific Run

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
"
```

---

## 2. Psyche CLI (On-Chain)

The Psyche CLI binary is `psyche-solana-client`. The `run-manager` subcommand handles run queries. **Most subcommands require a known run-id** (see table below).

There is no `run-manager list` command — CLI cannot discover runs.

### Install

```bash
# Option 1: Docker
docker pull ghcr.io/psychefoundation/psyche:latest

# Option 2: Build from source
git clone https://github.com/PsycheFoundation/psyche
cd psyche
cargo build --release --bin psyche-solana-client
```

### Invocation

```bash
# If built from source:
cargo run --release --bin psyche-solana-client -- run-manager <SUBCOMMAND>

# Or if the binary is in PATH:
psyche-solana-client run-manager <SUBCOMMAND>

# Docker:
docker run ghcr.io/psychefoundation/psyche run-manager <SUBCOMMAND>
```

The examples below use the short form `run-manager` for readability. Replace with the full invocation above based on your setup.

### Available Subcommands

| Subcommand | Purpose | Requires run-id |
|-----------|---------|----------------|
| JsonDumpRun | Full on-chain run state dump | Yes |
| JsonDumpUser | User participation data | Yes + address |
| CanJoin | Check if a wallet can join a run | Yes |
| CreateRun | Create a new run (admin) | No |
| CloseRun | Close a run (admin) | Yes |
| SetPaused | Pause/resume a run (admin) | Yes |
| UpdateConfig | Update run config (admin) | Yes |
| Checkpoint | Trigger checkpoint (admin) | Yes |
| DownloadResults | Download training results | Yes |
| UploadData | Upload training data | Yes |

### Dump Run State

```bash
run-manager json-dump-run \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID>
```

Returns full JSON dump of the run's on-chain state: participants, round info, epoch data, configuration, and **checkpoint backend type** (Hub, P2P, Gcs, P2PGcs, Ephemeral).

### Dump User State

```bash
run-manager json-dump-user \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID> \
  --address <SOLANA_PUBKEY>
```

Returns the user's participation data: collateral, rewards, training history.

### Pause / Resume a Run

```bash
# Pause
run-manager set-paused \
  --rpc <RPC> \
  --run-id <RUN_ID> \
  --wallet-private-key-path <KEY_PATH>

# Resume
run-manager set-paused \
  --rpc <RPC> \
  --run-id <RUN_ID> \
  --resume \
  --wallet-private-key-path <KEY_PATH>
```

**Note:** Pause/resume requires the run coordinator's wallet. This is an admin operation.

---

## 3. HuggingFace API (Checkpoint Search — Best Effort)

HuggingFace is used **only for checkpoint lookup**, not for run discovery. All endpoints are public, no authentication required.

**Important:** Not all runs publish checkpoints to HuggingFace. Checkpoint types include Hub (HF), P2P, Gcs, P2PGcs, and Ephemeral. Only Hub-type checkpoints appear on HuggingFace.

### Narrow Search (PsycheFoundation org)

Search by run-id first (always present), then run-name (may be empty):

```bash
# By run-id (primary)
curl -s "https://huggingface.co/api/models?search=<run-id>&author=PsycheFoundation"

# By run-name (secondary, if run-id search returns no results)
curl -s "https://huggingface.co/api/models?search=<run-name>&author=PsycheFoundation"
```

### Broad Search (org-unrestricted fallback)

If narrow search returns no results, checkpoints may be under individual users' HuggingFace accounts:

```bash
# By run-id (primary)
curl -s "https://huggingface.co/api/models?search=<run-id>"

# By run-name (secondary)
curl -s "https://huggingface.co/api/models?search=<run-name>"
```

**Multiple matches:** Broad search may return unrelated models. Disambiguation rules:
- Prefer exact name match with run-id or run-name
- If multiple candidates, list all and let the user choose
- Tag all results with `[best-effort]`
- If found outside PsycheFoundation, note the actual org/user

### Parse Model List

```bash
curl -s "https://huggingface.co/api/models?search=<run-id>&author=PsycheFoundation" | \
  python3 -c "
import sys, json
models = json.load(sys.stdin)
if not models:
    print('No models found for this search query.')
else:
    for m in sorted(models, key=lambda x: x.get('createdAt',''), reverse=True):
        print(f\"{m['id']:55s}  created: {m.get('createdAt','N/A')[:10]}  downloads: {m.get('downloads',0)}\")
    print(f'\nTotal: {len(models)} models')
"
```

### Get Model Details

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

### List Checkpoint Files (Tree)

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>/tree/main" | \
  python3 -c "
import sys, json
files = json.load(sys.stdin)
total = 0
for f in sorted(files, key=lambda x: x.get('size',0), reverse=True):
    size = f.get('size', 0)
    total += size
    if size > 1024*1024:
        print(f\"  {size/1024/1024:>10.1f} MB  {f['path']}\")
    else:
        print(f\"  {size/1024:>10.1f} KB  {f['path']}\")
print(f'\nTotal: {total/1024/1024/1024:.1f} GB across {len(files)} files')
"
```

---

## 4. Solana RPC (Reference Only)

**These commands are documented for reference only.** Solana RPC returns raw Borsh-serialized binary data that requires PDA derivation and schema-specific deserialization. This is not practically feasible for an AI agent. **Prefer the Psyche CLI for on-chain queries.**

For production use, set `$SOLANA_RPC_URL` to a private RPC provider. The default public endpoint (`https://api.mainnet-beta.solana.com`) has strict rate limits.

### Check Account Balance

```bash
curl -s -X POST "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getBalance",
    "params": ["<WALLET_ADDRESS>"]
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
lamports = r.get('result', {}).get('value', 0)
print(f'Balance: {lamports / 1e9:.4f} SOL')
"
```

### Get Account Info

```bash
curl -s -X POST "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAccountInfo",
    "params": ["<COORDINATOR_ACCOUNT>", {"encoding": "base64"}]
  }'
```

**Note:** The account data is Borsh-serialized binary. PDA derivation (run-id to coordinator account address) requires Solana cryptographic operations. For human-readable output, use the Psyche CLI `json-dump-run` command instead.

### Known Program IDs

| Program | Address |
|---------|---------|
| Coordinator | `4SHugWqSXwKE5fqDchkJcPEqnoZE22VYKtSTVm7axbT7` |
| Mining Pool | `PsyMP8fXEEMo2C6C84s8eXuRUrvzQnZyquyjipDRohf` |
