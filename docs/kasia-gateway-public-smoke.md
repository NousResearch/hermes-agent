# Kasia Gateway Public Smoke

This is the current repeatable baseline smoke for the Hermes Kasia gateway.

Use this runbook before and after each Kasia phase so we compare behavior against the same public-infra setup.

## Current Baseline

Today the Hermes Kasia integration supports:

- Dedicated Hermes Kasia identity derived from `KASIA_SEED_PHRASE`
- Direct-message only flow
- Text-only delivery in Hermes
- Inbound handshake detection
- Automatic handshake response for authorized peers
- Active-conversation outbound sends
- Send jobs with delivery phases:
  - `queued`
  - `submitting`
  - `submitted`
  - `waiting_for_indexer`
  - `processed`
  - `failed`
  - `rejected`
- `priority` fee policy as the default bridge behavior

Current public-infra reality:

- Hermes submission is fast
- public acceptance and indexer visibility are often the slowest part
- live delivery time varies with mempool pressure and public indexer lag

## Public Infra Baseline

The current working public baseline is:

- Indexer: `https://indexer.kasia.fyi`
- Node: `wss://wrpc.kasia.fyi`
- Network: `mainnet`
- Fee policy: `priority`

This runbook intentionally uses the public path because that is what we have validated repeatedly.

## Prerequisites

- Two funded Kasia/Kaspa seed phrases:
  - one for Hermes
  - one for the peer
- Node.js installed
- Hermes Python environment ready:

```bash
source .venv/bin/activate
```

- Enough balance for:
  - handshake transfer
  - message fees
  - retries during public-infra testing

## Recommended Repeatable Setup

Use a local second Kasia bridge as the peer. This is more repeatable than manually driving KaChat for every baseline test.

Ports:

- Hermes bridge: `3010`
- Peer bridge: `3011`

State dirs:

- Hermes home: `/tmp/hermes-kasia-home`
- Peer bridge state: `/tmp/hermes-kasia-peer`

## Environment

Set Hermes gateway env:

```bash
export KASIA_ENABLED=true
export KASIA_SEED_PHRASE="YOUR_HERMES_SEED"
export KASIA_INDEXER_URL="https://indexer.kasia.fyi"
export KASIA_NODE_WBORSH_URL="wss://wrpc.kasia.fyi"
export KASIA_NETWORK="mainnet"
export KASIA_FEE_POLICY="priority"
export KASIA_ALLOW_ALL_USERS=true
```

Set peer bridge env:

```bash
export KASIA_SEED_PHRASE="YOUR_PEER_SEED"
export KASIA_INDEXER_URL="https://indexer.kasia.fyi"
export KASIA_NODE_WBORSH_URL="wss://wrpc.kasia.fyi"
export KASIA_NETWORK="mainnet"
export KASIA_FEE_POLICY="priority"
```

## Start The Peer Bridge

```bash
node scripts/kasia-bridge/bridge.js --port 3011 --state-dir /tmp/hermes-kasia-peer
```

Verify:

```bash
curl http://127.0.0.1:3011/health
```

## Start The Hermes Gateway

Run Hermes with an isolated home so smoke state is easy to inspect:

```bash
source .venv/bin/activate
HERMES_HOME=/tmp/hermes-kasia-home python -m hermes_cli.main gateway run
```

Verify:

```bash
curl http://127.0.0.1:3010/health
```

Confirm:

- `status`
- `walletAddress`
- `feePolicy`
- `lastSyncMs`

## Smoke Flow

### 1. Peer Sends The Initial Handshake

Current baseline still assumes the peer initiates first contact.

Do this from a Kasia-compatible peer:

- KaChat / Kasia app
- or a local helper that sends the handshake through the peer wallet path

Expected result:

- Hermes sees the handshake
- Hermes auto-responds
- both sides converge to an active conversation

### 2. Confirm Active Conversation State

Inspect state files:

```bash
python -m json.tool /tmp/hermes-kasia-home/kasia/state.json | sed -n '1,220p'
python -m json.tool /tmp/hermes-kasia-peer/state.json | sed -n '1,220p'
```

Expected result:

- both sides show the peer conversation
- conversation status is `active`
- aliases are populated

### 3. Peer Sends A Short Message To Hermes

Send a small message like:

```text
/new
```

Expected result:

- Hermes receives it
- Hermes replies
- the peer eventually receives the reply through the public indexer path

### 4. Hermes Sends A Direct Outbound Message

Use the Hermes bridge directly:

```bash
curl -sS -X POST http://127.0.0.1:3010/send \
  -H 'Content-Type: application/json' \
  -d '{"chatId":"PEER_KASPA_ADDRESS","message":"hello from hermes","waitMs":5000}'
```

Expected result:

- response contains `jobId`
- early status is usually `submitted` or `waiting_for_indexer`
- final status becomes `processed`

### 5. Poll The Send Job

```bash
curl -sS http://127.0.0.1:3010/send/JOB_ID
```

Expected result:

- job eventually reaches `processed`
- `indexedParts` matches `completedParts`

### 6. Optional Multipart Check

Use a longer message that still stays inside the configured multipart cap.

Expected result:

- Hermes submits all parts
- peer eventually processes all tx ids
- final job status becomes `processed`

## Helper Commands

Bridge health:

```bash
curl http://127.0.0.1:3010/health
curl http://127.0.0.1:3011/health
```

Queued inbound events:

```bash
curl http://127.0.0.1:3010/messages
curl http://127.0.0.1:3011/messages
```

Chat info:

```bash
curl http://127.0.0.1:3010/chat/PEER_KASPA_ADDRESS
curl http://127.0.0.1:3011/chat/HERMES_KASPA_ADDRESS
```

Send job status:

```bash
curl http://127.0.0.1:3010/send/JOB_ID
```

Readable state output:

```bash
python -m json.tool /tmp/hermes-kasia-home/kasia/state.json | sed -n '1,260p'
python -m json.tool /tmp/hermes-kasia-peer/state.json | sed -n '1,260p'
```

Stop leftover listeners:

```bash
lsof -iTCP:3010 -sTCP:LISTEN
lsof -iTCP:3011 -sTCP:LISTEN
```

## Success Criteria

The baseline smoke is considered good when:

- Hermes bridge starts cleanly
- peer bridge starts cleanly
- peer initiates handshake
- Hermes auto-responds
- conversation becomes active
- peer message reaches Hermes
- Hermes reply reaches peer
- Hermes outbound send reaches `processed`

## Notes For Future Phases

- Keep this smoke constant as much as possible
- If we add new capability, test the old baseline first, then test the new feature
- If public infra is congested, compare timing in terms of:
  - submitted
  - indexed
  - processed
  rather than assuming Hermes is the bottleneck
