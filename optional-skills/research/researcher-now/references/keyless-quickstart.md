# Keyless Quickstart — zero to first result, no API key, no signup

Researcher is wallet-native. You do **not** implement the payment protocol: point
any x402 / MPP-capable client at `researcher.now` with a funded wallet and it pays
transparently per request. The first paid call provisions a persistent account and
returns a **durable API key** in the `X-Researcher-Agent-Key` response header —
reuse it as a Bearer token for everything after (reads, management, more runs).

Pricing is fixed per request (the quoted price; runs are capped at
`limits.maxCostUsd`). You are never charged more than you authorized.

## HTTP — Tempo CLI (verified)

```bash
curl -L https://tempo.xyz/install | bash      # one time
tempo wallet login                            # one time; fund the wallet
# Deep research run (no key — paid from your wallet):
tempo request https://researcher.now/v1/runs \
  --json '{"source":{"type":"topic","topic":"your question"},"limits":{"maxCostUsd":2}}'
# Single URL / video:
tempo request https://researcher.now/v1/analyze --json '{"url":"https://…"}'
# Ask an expert persona:
tempo request https://researcher.now/v1/entities/paul-graham/chat \
  --json '{"message":"your question"}'
```

The response includes `agentApiKey` (also the `X-Researcher-Agent-Key` header).
Save it: `export RESEARCHER_API_KEY=rk_…` and reuse it on later calls
(`Authorization: Bearer $RESEARCHER_API_KEY`), or keep paying per call from the
wallet — both work.

## HTTP — mppx (x402 SDK/CLI), equivalent

```bash
mppx https://researcher.now/v1/runs -J '{"source":{"type":"topic","topic":"…"}}'
```

## MCP — wallet-direct

Point any x402-capable MCP client at `https://researcher.now/mcp` with your wallet.
The handshake needs no auth; paid tools (`deep_research`, `analyze_article`,
`analyze_video`, `ask_persona`) answer `402` with an MPP session challenge the
client authorizes from the wallet, then provisions an account and returns the key
in `X-Researcher-Agent-Key`. Free tools (`list_personas`, `recall_research`, …)
need no payment.

## Already have a key (or a human can make one)

Skip the wallet entirely: send `Authorization: Bearer rk_…` (mint one at
`https://researcher.now/account/`). Everything above works identically with a key.

## How payment works (you don't have to)

Under the hood it's HTTP 402 + an MPP session on Tempo (chain 4217, USDC.e): no
credential → `402` with a `WWW-Authenticate` challenge → the client authorizes a
session from your wallet → retry with the credential → result. Capture settles
on-chain in the same stablecoin. Any x402 client does this for you.
