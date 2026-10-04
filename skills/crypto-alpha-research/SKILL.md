---
name: crypto-alpha-research
description: >-
  Structured due-diligence on a crypto project from an X/Twitter handle, a token
  contract address, or a project name. Produces a decision-first comparison table
  and a risk-flagged verdict — authenticity / brand-squatting checks, on-chain
  facts, team background, narrative timing, competitor position, and the "real
  project vs good token" value-capture test. Use when vetting or tracking a crypto
  token/project before investing, or when asked to "research / look into / add" a
  crypto project, handle, or contract.
license: MIT
version: 0.1.0
metadata:
  author: catfrommarss
  category: research
  tags: [crypto, due-diligence, research, web3, tokens]
---

# Crypto Alpha Research

Turn a raw lead (an X handle, a contract address, or a project name) into a
**decision**. Always output a comparison table first, then a one-line verdict
from a fixed set of decision templates — never "to be confirmed".

## When to use
- Vetting a new token / project before tracking or buying.
- A message contains an X/Twitter handle, a contract address, or a project name
  and asks to "research / look into / vet / add" it.
- Refreshing a project you already track (delta-focused).

## When NOT to use
- A pure price check with no project context (just query a price source).
- Generic market commentary unrelated to a specific project.

## Inputs
Accept any of: X/Twitter handle (`@name`), contract address (chain may be missing
— infer it), project name, or several targets in one message. With ≥2 targets,
run the batch path and produce one comparison table for all.

## Required mindset (hard rules)
- **Comparison table first**, even when the data looks great.
- **Real project ≠ good token.** If the product runs fine without the token, flag
  value-capture as missing — do not soften it to "worth watching".
- **Verify before trusting.** Never treat a user-supplied handle/contract as
  genuine; brand-squatting and impostor tokens are the default assumption.
- **Surface risks, don't decide for the user.** List death / red-flag signals and
  let the user call it. Never auto-conclude "dead".
- **State on-chain numbers as facts**, not death verdicts.
- Every verdict uses a decision template below — never "TBD".
- If a data point can't be retrieved, label it "not found" — never fabricate.

## Procedure
Run steps 1–6 to gather, then step 7 to synthesize. Parallelize independent
lookups where the runtime allows.

1. **Profile** — handle, bio, creation date, followers/following, tweet count,
   verified status. (Use an X/Twitter data tool or MCP if available.)
2. **Activity & narrative** — read tweet *content*, not just timestamps: what the
   project ships, posting cadence, pinned/positioning, market reaction (quotes /
   retweets), deleted tweets (= red flag), and who follows it.
3. **Authenticity / brand-squatting** — suspicious signals: `_ai / _base /
   _official`-style suffixes; <50 followers but a polished bio; <5 tweets; a copy
   of a real account (incl. typos); <30 days old with no notable followers. ≥1 →
   find the real account and compare. For tokens: assume multiple same-name
   contracts exist (incl. impostors on other chains); confirm the official one via
   the project's own channels. ≥2 signals → label "likely impostor".
4. **On-chain facts** (only if a contract is in scope) — price, FDV / market cap,
   liquidity, holders, circulating supply, deployer history. **List the numbers
   as-is**; do not call it "dead". Use a price/explorer tool, a public read-only
   API, or an MCP — never bundle credentials.
5. **Team** — for each member: name, title, LinkedIn, X handle, web2 résumé, past
   crypto projects, rug/scam history. Red flags: unverifiable identity; refuses to
   disclose past work; LinkedIn mismatch; prior rug; serial launcher. Anonymity is
   NOT itself a red flag if backed by public VCs, credible KOL vouching, active
   GitHub, or verifiable past work — otherwise "watch first". Cross-check
   reputable crypto media (real reporting vs PR).
6. **Ecosystem & external signals**
   - *Narrative / meta + timing*: which meta, and is it early / peak / fading?
   - *Competitors*: who else does this; is the project leading / following / a clone?
   - *KOL tiers*: institutional / ecosystem backing (high quality) vs small-cap
     shill callers (discount heavily) vs engagement pods (yellow flag).
   - *Tokenomics / value capture*: run the "remove the token — does the product
     still run?" test. Missing sink/utility → flag "real project ≠ good token".
   - *Triggered (only if applicable, else "not found")*: GitHub activity; VC
     backing / funding rounds; media coverage; valuation vs last round; unlock /
     vesting schedule (→ note an upcoming catalyst if found).
7. **Synthesize** — build the comparison table, assign a verdict, and offer the
   user the track/insert options.

## Output: comparison table (first, always)
Columns (rename/trim to fit any tracker):

| Column | Meaning |
|---|---|
| Name | Project name |
| One-liner | One-sentence positioning |
| Stage | concept / testnet / mainnet / token-live / dead (never auto-set "dead") |
| Tags | from your own tag set |
| FDV | fully-diluted valuation, USD |
| URL | X / site / contract |
| Team | vetted / anonymous / not-found / red-flag |
| Narrative / meta | which meta + timing (early / peak / fading) |
| Competitor position | leading / following / clone |
| KOL quality | institutional vs shill vs engagement pod |
| Token utility | has sink / missing (real ≠ good token) |
| Risk signals | red flags, listed |

For batch (≥2 targets), a lighter first pass also works: `handle | one-liner |
followers | tweets | authenticity | on-chain | team | narrative | take`. Sort:
suspicious → team red flag → high quality → stable. Every row carries a take.

Example tag set (adapt): AI · L1 · L2 · DeFi · DePIN · GameFi · RWA · meme ·
Infra · Stablecoin · Perp · Prediction · Privacy · Consumer · Social · ZK ·
Cross-chain · Cosmos · Restaking · BTCFi · DeSci · MOVE.

## Output: decision templates (pick exactly one — never "TBD")
Pairs are allowed (e.g. "High priority" + "Real project ≠ good token").
- **Worth tracking** — solid; monitor.
- **High priority** — strong signal; watch closely.
- **Impostor risk** — likely brand-squat / fake token.
- **Team red flag** — unresolved team concern.
- **Dead candidate** — ≥2 death signals; surface, don't auto-conclude.
- **Stage upgrade** — moved concept → testnet → mainnet → token-live.
- **Watch first** — anonymous / early; insufficient signal.
- **Real project ≠ good token** — genuine project, value capture missing.
- **Narrative fading** — the meta is rotating out.
- **Valuation stretched** — FDV ahead of fundamentals / last round.
- **Slideware (low delivery)** — far more promised than shipped + slippage + non-trivial FDV.

Death signals (need ≥2 for "Dead candidate"): account 30 days silent / suspended
/ bio reset · liquidity collapsing / holders falling / deployer dumped · site 404
/ Discord disbanded / GitHub ~6 months no commits. List the specific ones found.

## Optional: Notion backend
If the user tracks projects in Notion, results can be written via the Notion MCP.
The user supplies their own database; never hardcode IDs in the skill. Read IDs
(e.g. `ALPHA_DB_ID`) from the user's config at runtime; if unset, fall back to
Markdown or CSV. Write rules: `Stage`/`Tags` use the user's enums; `FDV` in USD;
relations append/update only (never delete); on refresh, show a delta table first.
Never commit real database IDs, API keys, tokens, or endpoints.

## On-chain helper (optional)
For step 4, a bundled read-only helper is available — no API key required:

```bash
python scripts/price.py <token_address>        # price, FDV, market cap, liquidity, 24h volume/change
python scripts/price.py --search "<name>"      # list same-name tokens across chains (brand-squatting check)
```

It calls the public DexScreener API (read-only, GET only; `api.dexscreener.com`)
and prints JSON; no credentials are bundled. Holders count is not provided by this
source — label it "not found" or use an explorer if you have one. Treat the
numbers as facts (step 4), not as a death verdict.

## Pitfalls
- Dumping raw data instead of a judgment — lead with the table + verdict.
- Trusting a handle/contract because the user provided it.
- Calling a token "dead" from thin liquidity / low holders alone.
- Letting a real, impressive project mask a token with no value capture.
- Omitting the team section for anonymous teams — still write "anonymous + signals".
- Writing "TBD" — always commit to a decision template.

## Verification (before declaring done)
- [ ] Comparison table present with all columns.
- [ ] Authenticity / brand-squatting explicitly checked (especially for contracts).
- [ ] Team section present (named, or "anonymous + signals").
- [ ] Value-capture test applied; "real ≠ good token" flagged if relevant.
- [ ] Every claim sourced or labeled "not found" — nothing fabricated.
- [ ] Verdict uses a decision template, not "TBD".
