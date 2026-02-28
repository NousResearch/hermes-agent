# Psyche Network Monitor — Command Reference

Detailed command syntax and parsing examples for querying Psyche network data.

## HuggingFace API

All endpoints are public, no authentication required.

### List All Models

```bash
curl -s "https://huggingface.co/api/models?author=PsycheFoundation"
```

**Response:** JSON array. Each element contains:
- `id` — Full model ID (e.g., `PsycheFoundation/consilience-40b-CqX3FUm4`)
- `modelId` — Same as `id`
- `lastModified` — ISO 8601 timestamp
- `downloads` — Total download count
- `tags` — List of tags (architecture, library, etc.)
- `pipeline_tag` — Model pipeline type

**Parse to table:**

```bash
curl -s "https://huggingface.co/api/models?author=PsycheFoundation" | \
  python3 -c "
import sys, json
models = json.load(sys.stdin)
print(f'{'Model ID':55s}  {'Modified':12s}  {'Downloads':>10s}')
print('-' * 82)
for m in sorted(models, key=lambda x: x.get('lastModified',''), reverse=True):
    print(f\"{m['id']:55s}  {m.get('lastModified','N/A')[:10]:12s}  {m.get('downloads',0):>10d}\")
print(f'\\nTotal: {len(models)} models')
"
```

### Get Model Details

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>"
```

**Response:** JSON object with full metadata including `siblings` (file list), `cardData`, `config`, etc.

### List Files (Tree)

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>/tree/main"
```

**Response:** JSON array. Each element contains:
- `path` — File path relative to repo root
- `size` — File size in bytes
- `type` — `file` or `directory`
- `oid` — Git object ID (SHA)

**Parse with size:**

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
print(f'\\nTotal: {total/1024/1024/1024:.1f} GB across {len(files)} files')
"
```

### List Commits (Version History)

```bash
curl -s "https://huggingface.co/api/models/PsycheFoundation/<model-id>/commits/main" | \
  python3 -c "
import sys, json
commits = json.load(sys.stdin)
for c in commits[:10]:
    print(f\"{c['date'][:16]}  {c['title']}\")
"
```

---

## Solana RPC

For on-chain queries. The default public endpoint has rate limits; for production use, set `$SOLANA_RPC_URL` to a private RPC provider.

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

### Get Account Info (Run State)

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

**Note:** The account data is binary (Borsh-serialized). For human-readable output, prefer the Psyche CLI `json-dump-run` command below.

---

## Psyche CLI (run-manager)

Requires the Psyche client. Install via Docker or build from source.

### Install

```bash
# Option 1: Docker
docker pull ghcr.io/psychefoundation/psyche:latest

# Option 2: Build from source
git clone https://github.com/PsycheFoundation/psyche
cd psyche
cargo build --release --bin psyche-solana-client
```

### Dump Run State

```bash
run-manager json-dump-run \
  --rpc "${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}" \
  --run-id <RUN_ID>
```

Returns full JSON dump of the run's on-chain state: participants, round info, epoch data, configuration.

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
