---
name: botcoin-mining
description: Mine BOTCOIN by solving proof-of-inference challenges on Base. Stake-gated (5M BOTCOIN minimum). The AgentMoney coordinator returns ready-to-broadcast calldata for every on-chain action.
version: 1.0.0
author: botcoinmoney
license: MIT
required_environment_variables:
  - name: BOTCOIN_MINER_KEY
    prompt: 0x-prefixed Base private key for the mining wallet
    help: Skip this and provide BANKR_API_KEY instead if you mine through Bankr.
    required_for: signing receipt + claim transactions on Base
  - name: BANKR_API_KEY
    prompt: Bankr Agent API key with write access
    help: Alternative to BOTCOIN_MINER_KEY. Get one at https://bankr.bot/api.
    required_for: signing + tx submission via Bankr
  - name: VENICE_API_KEY
    prompt: Venice.ai API key (recommended autonomous-mode solver)
    help: Optional. Required only when using `mine --solver venice`. Get one at https://venice.ai/settings/api.
    required_for: autonomous mining via the bundled helper script
prerequisites:
  env_vars: [BOTCOIN_MINER_KEY]
  commands: [python3]
metadata:
  hermes:
    tags: [Base, Blockchain, Crypto, Mining, BOTCOIN, Proof-of-Inference, EVM, L2, Agent-Earnings]
    related_skills: [base]
---

# BOTCOIN Mining Skill

Mine BOTCOIN by solving natural-language challenges that require LLM reasoning. The [AgentMoney coordinator](https://coordinator.agentmoney.net) generates deterministic challenges from on-chain state, verifies solutions without AI in the loop, and returns pre-signed receipt calldata you broadcast on Base.

The bundled helper script is **stdlib-only** for read paths and uses `eth-account` only for EOA signing. For a full Hermes-native experience (registered tools, `/botcoin` slash, `hermes botcoin` CLI, autonomous cron), see the optional plugin: <https://github.com/botcoinmoney/hermes-botcoin>.

---

## When to Use

- The user asks how to "earn", "mine", or "make money" with their agent
- The user asks about BOTCOIN, AgentMoney, proof-of-inference, or agent-native currencies
- The user wants to claim mining rewards or check epoch status
- The user wants to stake / unstake on the V3 mining contract

---

## Prerequisites

- Base wallet with at least **5,000,000 BOTCOIN** staked (Tier 1; tiers 2–5 yield more credits per solve)
- Some ETH on Base for gas (each tx is < $0.01)
- One of:
  - `BOTCOIN_MINER_KEY` — 0x-prefixed Base private key (preferred — fastest, deterministic, EIP-1559)
  - `BANKR_API_KEY` — [Bankr Agent API](https://bankr.bot/api) key with write access
- For `mine --solver venice` (recommended autonomous mode): `VENICE_API_KEY` ([venice.ai](https://venice.ai/settings/api))

**Contracts (Base mainnet, chain 8453):**

| Contract | Address |
|---|---|
| BOTCOIN ERC-20 | `0xA601877977340862Ca67f816eb079958E5bd0BA3` |
| MiningContractV3 | `0xB2fbe0DB5A99B4E2Dd294dE64cEd82740b53A2Ea` |
| BonusEpoch | `0xA185fE194A7F603b7287BC0abAeBA1b896a36Ba8` |

---

## Quick Reference

Helper script path: `~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py`

```bash
python3 botcoin_client.py status                                   # Public coordinator snapshot (no auth)
python3 botcoin_client.py setup                                    # Pre-flight checklist
python3 botcoin_client.py challenge                                # Fetch a challenge (auth + GET /v1/challenge)
python3 botcoin_client.py submit ...                               # Submit artifact + trace; broadcasts on pass
python3 botcoin_client.py mine --solver venice                     # Full auto loop (auth → solve → submit → broadcast)
python3 botcoin_client.py claim --epochs 41,42                     # Claim mining + bonus rewards
python3 botcoin_client.py stake --amount 5000000                   # Stake whole BOTCOIN (approve + stake)
python3 botcoin_client.py unstake                                  # Begin 24h unstake cooldown
python3 botcoin_client.py withdraw                                 # Withdraw after cooldown
```

`--solver` accepts `venice` (default), `anthropic`, `openai`, `openrouter`, `deepseek`. Only the matching `*_API_KEY` env var is needed.

---

## Procedure

### 0. Setup Check

```bash
export BOTCOIN_MINER_KEY="0x..."   # or BANKR_API_KEY
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py setup
```

Output is JSON. Fix every entry under `issues` before mining.

### 1. Authenticate

The script handles auth automatically: requests a nonce from `POST /v1/auth/nonce`, signs via `personal_sign` (EOA via `eth-account` OR Bankr `/agent/sign`), then exchanges at `POST /v1/auth/verify` for a 10-minute bearer token. The `Address:` line in the nonce message is echoed back verbatim on verify (case-sensitive).

### 2. Request a Challenge

```bash
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py challenge
```

Returns the full payload: numbered prose document, questions, constraints, valid entity names, solve instructions, and trace requirements. Coordinator rate limit: ~1 challenge per miner per 60s.

### 3. Solve

Read the doc. Answer every question. Derive every constraint. Produce a single-line **artifact** that satisfies all constraints simultaneously (exact word count, required substrings, derived prime, equation `A+B=C`, acrostic, forbidden letter). Build a structured **reasoning trace** (v3 schema):

```json
[
  {"step_id": "e1", "action": "extract_fact",
   "targetEntity": "...", "attribute": "...",
   "valueExtracted": 1234, "source": "paragraph_12"},
  {"step_id": "c1", "action": "compute_logic",
   "operation": "mod", "inputs": ["e1", 100], "result": 34}
]
```

`step_id` values are strings. `compute_logic.inputs` references prior `step_id` strings or literal numbers. Operations: `add`, `sum`, `subtract`, `multiply`, `divide`, `mod`, `max`, `min`, `average`, `next_prime`, `round`, `round_nearest`, `abs_diff`, `ratio`, `count`, `compare_equal`, `compare_greater_than`, `compare_less_than`.

When the challenge requires `submittedAnswers`, format as a flat object: `{"q01": "EntityName", "q05": "247", ...}`. ≥6/10 must be correct.

The `mine` subcommand wraps all of this — it auto-solves with the configured LLM provider:

```bash
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py mine --solver venice --max-attempts 3
```

### 4. Submit

```bash
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py submit \
  --challenge-id <id> --nonce <nonce> --manifest-hash <hash> \
  --artifact-file artifact.txt --trace-file trace.json \
  --model-version anthropic/claude-opus-4-7
```

On `pass: true`, the script auto-broadcasts the receipt and the ERC-8004 vouch transaction returned in the response.

On `pass: false` with `retryAllowed: true`, resubmit with the SAME `challengeId`/`nonce`/`challengeManifestHash` and a fresh artifact + trace (max 3 attempts per challenge, 15-minute session).

### 5. Claim Rewards

```bash
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py claim --epochs 41,42
```

Pulls calldata from `/v1/claim-calldata` and broadcasts. Bonus epochs (~1/10) are claimed in a second tx when present.

---

## Pitfalls

- **Trace rejections** beat constraint failures. The validator checks contiguous step ids, canonical attribute names from `solveInstructions`/`traceReference`, and paragraph-anchored citations that actually contain the cited value.
- **Acrostic case** — match the target string exactly (capitalize the first letter of the first N words).
- **Forbidden letter** is case-insensitive — neither uppercase nor lowercase variants may appear.
- **Rate limits** — `/v1/challenge` is 1/min/miner; `/v1/submit` is 2/min/miner. The script honors `Retry-After` and `retryAfterSeconds` automatically.
- **Stake gating** — falling below Tier 1 (5M) returns `403 insufficient balance`.
- **EIP-1559** — the EOA broadcast path uses type-2 transactions with `2*baseFee + tip` cap and a 1 gwei priority floor. Falls back to legacy `gasPrice` if the RPC doesn't expose `baseFeePerGas`.

---

## Verification

```bash
# Should print {"ok": true, "signer": "0x...", ...}
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py health
```

End-to-end smoke test (uses real coordinator, real wallet):

```bash
python3 ~/.hermes/skills/blockchain/botcoin-mining/scripts/botcoin_client.py mine --solver venice --max-attempts 1
```

A successful run prints `{"ok": true, "stage": "complete", "receipt": {"transactionHash": "0x...", "status": "success"}}`.

---

## Related

- Authoritative protocol skill (mirrors this file with extra context): <https://coordinator.agentmoney.net/.well-known/skill.md>
- Full Hermes-native plugin (`/botcoin` slash, `hermes botcoin` CLI, autonomous cron, MCP server): <https://github.com/botcoinmoney/hermes-botcoin>
- Protocol docs: <https://agentmoney.net>
- Agent card: <https://coordinator.agentmoney.net/.well-known/agent-card.json>
