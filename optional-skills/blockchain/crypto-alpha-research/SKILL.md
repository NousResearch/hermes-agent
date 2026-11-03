---
name: crypto-alpha-research
description: Token and project due diligence with risk-flagged verdicts.
version: 0.2.0
author: catfrommarss, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: blockchain
    tags: [Crypto, Blockchain, Research, Due-Diligence, Tokens, Web3]
    related_skills: [hyperliquid]
---

# Crypto Alpha Research Skill

Turn a raw crypto lead — an X/Twitter handle, a token contract address, or a
project name — into a decision: a comparison table first, then a one-line
verdict from a fixed template set. Covers authenticity and brand-squatting
checks, on-chain facts, team background, narrative timing, competitor position,
and the "real project ≠ good token" value-capture test. It researches and
surfaces risks; it does not place trades or auto-conclude a project is dead.

## When to Use

- A message contains an X handle, a contract address, or a project name and
  asks to research / vet / look into / track it.
- Vetting a new token or project before buying or tracking it.
- Refreshing a project already tracked (delta-focused re-run).
- NOT for a bare price check with no project context, and NOT for generic
  market commentary unrelated to a specific project.

## Prerequisites

- No API key: the bundled helper calls the public DexScreener API
  (`api.dexscreener.com`, read-only GET).
- Profile and social lookups use whatever the session provides: an X/Twitter
  data MCP server if one is configured, otherwise `web_extract` /
  `browser_navigate` on public pages.
- Optional Notion backend: if the user tracks projects in Notion through a
  Notion MCP server, read the database ID from the user's own config (for
  example `ALPHA_DB_ID`); if unset, fall back to Markdown or CSV output. Write
  rules: use the user's own enums for Stage/Tags, FDV in USD, append/update
  only (never delete), and show a delta table first on refresh. Never hardcode
  database IDs, API keys, tokens, or endpoints in the skill.

## How to Run

The on-chain helper (Procedure step 4) runs via `terminal`:

```bash
python scripts/price.py <token_address>
```

```bash
python scripts/price.py --search "<name or symbol>"
```

The first form prints price, FDV, market cap, liquidity, and 24h volume/change
for the token's highest-liquidity pair. The second lists candidate tokens with
an `exactMatch` flag per entry: `true` means the normalized name or symbol
equals the query (same-name / brand-squatting evidence); `false` means a fuzzy
hit only. Holders count is not provided by this source — label it "not found"
or use an explorer. Output is JSON; on failure the helper prints
`{"error": ...}` so the run records "not found" instead of breaking.

## Quick Reference

Comparison table columns (rename or trim to fit any tracker):

| Column | Meaning |
|---|---|
| Name | Project name |
| One-liner | One-sentence positioning |
| Stage | concept / testnet / mainnet / token-live / dead (never auto-set "dead") |
| Tags | from your own tag set |
| FDV | fully-diluted valuation, USD |
| URL | X / site / contract |
| Team | vetted / anonymous / not-found / red-flag |
| Narrative | which meta + timing (early / peak / fading) |
| Competitor position | leading / following / clone |
| KOL quality | institutional vs shill vs engagement pod |
| Token utility | has sink / missing (real ≠ good token) |
| Risk signals | red flags, listed |

Example tag set (adapt): AI · L1 · L2 · DeFi · DePIN · GameFi · RWA · meme ·
Infra · Stablecoin · Perp · Prediction · Privacy · Consumer · Social · ZK ·
Cross-chain · Restaking · BTCFi · DeSci.

Decision templates — pick exactly one per target (pairs allowed, never "TBD"):
**Worth tracking** · **High priority** · **Impostor risk** · **Team red flag** ·
**Dead candidate** · **Stage upgrade** · **Watch first** · **Real project ≠
good token** · **Narrative fading** · **Valuation stretched** · **Slideware
(low delivery)**.

Death signals (need ≥2 for "Dead candidate"; list the specific ones found):
account 30 days silent / suspended / bio reset · liquidity collapsing /
holders falling / deployer dumped · site 404 / Discord disbanded / GitHub ~6
months without commits.

## Procedure

Hard rules:

- Comparison table first, even when the data looks great.
- Real project ≠ good token: if the product runs fine without the token, flag
  value capture as missing — do not soften it to "worth watching".
- Never treat a user-supplied handle or contract as genuine; brand-squatting
  and impostor tokens are the default assumption.
- State on-chain numbers as facts, not death verdicts; surface risks and let
  the user decide.
- A data point that cannot be retrieved is "not found" — never fabricate.

Steps (parallelize independent lookups where the runtime allows; with ≥2
targets, run every target and produce one combined table):

1. **Profile** — handle, bio, creation date, followers/following, tweet count,
   verified status.
2. **Activity & narrative** — read tweet content, not just timestamps: what
   the project ships, posting cadence, pinned positioning, market reaction
   (quotes/retweets), deleted tweets (red flag), notable followers.
3. **Authenticity / brand-squatting** — suspicious signals: `_ai` / `_base` /
   `_official`-style suffixes; <50 followers with a polished bio; <5 tweets; a
   copy of a real account (incl. typos); <30 days old with no notable
   followers. ≥1 signal → find the real account and compare; ≥2 → label
   "likely impostor". For tokens, assume multiple same-name contracts exist
   (incl. impostors on other chains): run the `--search` helper and confirm
   the official contract via the project's own channels.
4. **On-chain facts** (only when a contract is in scope) — price, FDV / market
   cap, liquidity, holders, circulating supply, deployer history via the
   bundled helper, a public explorer, or an MCP. List the numbers as-is.
5. **Team** — per member: name, title, LinkedIn, X handle, web2 résumé, past
   crypto projects, rug/scam history. Red flags: unverifiable identity,
   refusal to disclose past work, LinkedIn mismatch, prior rug, serial
   launcher. Anonymity alone is NOT a red flag when backed by public VCs,
   credible KOL vouching, active GitHub, or verifiable past work — otherwise
   "watch first". Cross-check reputable crypto media (real reporting vs PR).
6. **Ecosystem & external signals** — the narrative/meta and its timing
   (early / peak / fading); competitors (leading / following / clone); KOL
   tiers (institutional or ecosystem backing vs small-cap shill callers vs
   engagement pods); the value-capture test ("remove the token — does the
   product still run?"); when applicable, otherwise "not found": GitHub
   activity, VC backing and funding rounds, media coverage, valuation vs last
   round, unlock/vesting schedule (note an upcoming catalyst if found).
7. **Synthesize** — build the comparison table, assign a verdict template,
   offer track/insert options. For batches a lighter first pass works:
   handle | one-liner | followers | tweets | authenticity | on-chain | team |
   narrative | take — sorted suspicious → team red flag → high quality →
   stable, every row carrying a take.

## Pitfalls

- Dumping raw data instead of a judgment — lead with the table + verdict.
- Trusting a handle or contract because the user provided it.
- Calling a token "dead" from thin liquidity or low holders alone.
- Treating every `--search` hit as same-name evidence — only `exactMatch`
  entries are; the rest are fuzzy hits.
- Letting a real, impressive project mask a token with no value capture.
- Omitting the team section for anonymous teams — write "anonymous + signals".
- Writing "TBD" — always commit to a decision template.

## Verification

- [ ] Comparison table present with all columns.
- [ ] Authenticity / brand-squatting explicitly checked (especially contracts).
- [ ] Team section present (named, or "anonymous + signals").
- [ ] Value-capture test applied; "real ≠ good token" flagged if relevant.
- [ ] Every claim sourced or labeled "not found" — nothing fabricated.
- [ ] Verdict uses a decision template, not "TBD".
