---
name: token-research
description: |
  Due-diligence research on a crypto token before you interact with it —
  contract checks, liquidity, holder concentration, common rug-pull red flags,
  and social signals. Research and education only, NOT financial advice.
version: 0.1.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
category: web3
triggers:
  - "research this token [address/symbol]"
  - "is this token safe"
  - "check this contract [address]"
  - "due diligence on [token]"
  - "red flags for [token]"
  - "analyze this token [address]"
  - "is this a rug"
toolsets:
  - terminal
  - web
  - file
metadata:
  hermes:
    tags: [Web3, Crypto, Token, Due-Diligence, Security, DeFi, Research]
    related_skills: [evm, solana, osint-investigation]
---

# Token Research

Structured due-diligence on a crypto token: pull on-chain facts, check the
contract, assess liquidity and holder distribution, and surface the common
red flags associated with scams and rug pulls. The output is a clear risk
summary so the user can make their own informed decision.

> ⚠️ **This skill is research and education only. It is NOT financial advice
> and NOT a buy/sell signal.** On-chain data can be incomplete or manipulated,
> and a "clean" report does not guarantee safety. Never invest more than you
> can afford to lose. Always do your own research.

Uses `web` to query public block explorers and aggregators. No private keys,
no transactions — this skill never moves funds.

---

## When to Use

- User wants to vet a token before interacting with it
- User pastes a contract address and asks "is this safe?"
- User wants a red-flag scan on a project they're considering

Do NOT use for:
- Executing trades or swaps — this skill is read-only by design
- Price predictions — that's speculation, not due diligence
- Tax/accounting on holdings — different domain

---

## Hard Guardrails

1. **No financial advice.** Every report ends with the not-financial-advice
   disclaimer. Never say "you should buy/sell" — present facts and risks, let
   the user decide.
2. **Read-only.** This skill never signs a transaction, never asks for a
   private key or seed phrase, never connects a wallet. If the user pastes a
   seed phrase or private key, STOP and warn them to never share it.
3. **No guarantees.** A report with no red flags is "no major red flags found
   in public data" — never "this is safe." Sophisticated scams pass shallow
   checks.
4. **Verify the address.** Token symbols are not unique — scammers clone
   names. Always work from the contract address, and confirm the chain.

---

## Prerequisites

- `web` toolset for querying public explorers (Etherscan, Solscan,
  BscScan, etc.) and aggregators (DexScreener, CoinGecko).
- No API keys required for the public endpoints used here. If the user has
  explorer API keys set, use them for higher rate limits.

---

## Procedure

### Step 1 — Identify the token and chain

Establish exactly what's being researched:
- **Contract address** (preferred — unambiguous)
- **Chain** (Ethereum, Solana, BSC, Base, Arbitrum, …) — required, since the
  same address format can exist on multiple EVM chains
- If only a symbol is given, search a aggregator and confirm the address with
  the user before proceeding (symbols are routinely cloned by scams).

### Step 2 — Contract verification

Check the contract on the chain's explorer:
- **Is the source code verified?** Unverified contracts are a yellow flag —
  you can't audit what you can't read.
- **Is it a proxy / upgradeable?** Upgradeable contracts mean the logic can
  change after launch — note who controls upgrades.
- **Mint function** — can the owner mint unlimited new tokens? Major red flag.
- **Blacklist / pause functions** — can the owner freeze transfers or blacklist
  wallets? Common in honeypots.
- **Ownership** — is ownership renounced, or does one address retain control?
  Retained ownership isn't automatically bad, but it's a centralization risk.

### Step 3 — Liquidity analysis

- **Total liquidity** — thin liquidity = high slippage and easy price
  manipulation.
- **Is liquidity locked?** Unlocked LP tokens are the classic rug setup —
  the deployer can pull liquidity at any moment. Check for a lock (Unicrypt,
  Team.Finance) and its expiry.
- **LP holder concentration** — if one wallet holds most of the LP, that's a
  single point of failure.

### Step 4 — Holder distribution

Pull the top holders from the explorer:
- **Top-10 concentration** — if the top 10 wallets hold a large share of
  supply (excluding burns, locks, and the LP pair), a few actors can dump and
  crash the price.
- **Deployer holdings** — how much does the deployer wallet still hold?
- **Suspicious patterns** — many wallets funded from the same source at the
  same time often signal a single entity faking distribution.

### Step 5 — Honeypot and tax checks

- **Can you sell?** Honeypots let you buy but block selling. Check a honeypot
  detector or simulate a sell on a DEX aggregator (read-only quote).
- **Buy/sell tax** — high or asymmetric taxes (e.g. 0% buy, 90% sell) are a
  trap. Note the tax rates.
- **Transfer restrictions** — max-wallet / max-transaction limits that only
  the owner can change.

### Step 6 — Social and project signals

Using `web` and optionally the `osint-investigation` skill:
- **Age** — how old is the contract/project? Brand-new + heavy hype = caution.
- **Team** — anonymous vs doxxed. Anonymity isn't disqualifying but raises the
  bar on other checks.
- **Community** — real engagement vs bot-inflated follower counts.
- **Audits** — any third-party audit? By whom? (An "audit" by an unknown
  outfit is worth little.)
- **Website/docs** — present, coherent, or copy-pasted template?

### Step 7 — Risk summary

Produce a structured report:

```
Token Research: $SYMBOL
Chain:    Ethereum
Address:  0x...

── On-chain ──────────────────────────────
Contract verified:    Yes
Upgradeable proxy:    No
Mint function:        None found
Ownership:            Renounced
Liquidity:            $1.2M, locked until 2027-01 (Unicrypt)
Top-10 holders:       34% of supply (excl. LP + burns)
Sellable:             Yes (sell simulation passed)
Buy / sell tax:       2% / 2%

── Project ───────────────────────────────
Contract age:         8 months
Team:                 Doxxed (verifiable)
Audit:                CertiK (2025-11)
Community:            Organic engagement, ~20k holders

── Red flags ─────────────────────────────
- None critical found in public data
- Yellow: top-10 concentration moderately high (34%)

── Risk level: MODERATE ──────────────────
[2-3 sentence plain-English summary]

⚠️ Research only — NOT financial advice. Public on-chain data can be
incomplete or manipulated. A clean report is not a guarantee of safety.
DYOR and never invest more than you can afford to lose.
```

---

## Red Flag Quick Reference

| Critical (likely scam) | Yellow (caution) |
|---|---|
| Liquidity not locked | Ownership not renounced |
| Cannot sell (honeypot) | Unverified contract |
| Owner can mint unlimited | High top-10 concentration |
| Sell tax > 25% | Very new contract (< 2 weeks) |
| Hidden blacklist/pause | Anonymous team |
| LP held by single wallet | No audit |

Any single critical flag warrants stopping and warning the user prominently.

---

## Edge Cases

**Symbol with no address:** confirm the exact contract before researching —
clones are rampant. Show the user candidate addresses and let them pick.

**Brand-new token (hours old):** most checks are unreliable this early
(liquidity just added, no holder history). Say so — early = inherently higher
risk, regardless of what the surface data shows.

**Chain not supported by common explorers:** report what you can find and be
explicit about the data gaps.

**User pastes a seed phrase or private key:** STOP immediately. Warn them never
to share it with anyone, including AI, and to consider that wallet compromised.

---

## What This Skill Does NOT Cover

- Executing buys/sells/swaps — read-only by design
- Price predictions or "is it going up" — speculation, not research
- Smart-contract source-code audit at the bytecode level — recommend a
  professional audit for anything you're seriously considering
- Portfolio management or tax — different skills/tools
