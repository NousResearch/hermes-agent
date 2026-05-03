---
name: token-scam-analysis
description: Deep on-chain scam / rug / soft-rug analysis for EVM tokens (especially Clanker, Doppler, Bankr-style single-admin ERC-20s). Use when the user asks to "analyze this token for scam", "is this a rug", "should I trust this migration", "do a deep dive on holders and deployer", or provides one or more token addresses and wants a risk verdict backed by on-chain facts. Especially useful for migration narratives where a team claims they are "redeploying to fix tokenomics".
emoji: 🔍
tags: [scam, rug, analysis, forensics, clanker, doppler, evm, security]
visibility: public
---

# Token Scam Analysis Skill

You are performing a forensic on-chain analysis to determine whether a token (or set of tokens) is a scam, rug-pull, or soft-rug. The output is a written report saved to the user's file storage plus a verdict in chat.

## When to use this skill
- User gives one or more token contract addresses and asks for a risk assessment.
- User mentions a "migration" / "redeploy" / "new contract" narrative.
- User names a Twitter/X handle of the project and wants to cross-reference claims vs on-chain reality.
- Comparing old vs new contracts from the same team.

## Core principle
Narrative is noise. On-chain state is signal. Every claim the team makes should be checked against what the contract and the deployer's wallet actually did. If the two conflict, the chain wins.

**BUT: on-chain cleanliness ≠ not a scam.** A team can deploy a perfectly clean LayerZero OFT or ERC-20, hand it to a multisig, and still run a textbook insider pump-and-dump via CEX coordination and concentrated supply. You MUST always run the off-chain intel pass (Step Final-1) before issuing a verdict, or you will under-call real manipulation cases.

---

## Step 0 — Read the platform's deploy docs BEFORE judging tokenomics claims
Before making any claim about what a team "could not" or "should have" configured, read the deployment docs for the launch platform. Otherwise you will miss capabilities and bait on the team's narrative. Identify the platform by looking at the `allData()` `context` field (e.g., `"interface":"clanker.world"`), the deploy factory address, or the token ABI (`admin/originalAdmin/allData/isVerified` is Clanker-style).

Priority reads by platform:

- **Clanker v4** (`allData()` context = `clanker.world`, factory `0xe85a59c628f7d27878aceb4bf3b35733630083a9`):
  - https://clanker.gitbook.io/clanker-documentation/general/token-deployments  — overview: 100B ERC-20, extensions up to 90% of supply.
  - https://clanker.gitbook.io/clanker-documentation/authenticated/deploy-token-v4.0.0  — full deploy payload (vault, airdrop, fees, up to 7 reward recipients, pool config).
  - https://clanker.gitbook.io/clanker-documentation/references/core-contracts/v4  — ClankerVault / Airdrop extension internals.
  - What you MUST know before judging:
    - Tokenomics (allocations, lockups, vesting, multiple reward splits, custom paired token, initial market cap, static/dynamic fees) are set in the **deploy payload**. A team cannot say "we had to redeploy because we couldn't configure tokenomics" — Clanker v4 supports all of that in one transaction.
    - Min vault lockup 7 days, min airdrop lockup 1 day.
    - Max 90% of supply across all extensions (rest goes to LP).
    - Token bytecode being identical across deploys is **expected** (it's a factory template) — do not use bytecode-sameness as a red flag by itself. Configuration differences live in the deploy payload, not in the token runtime code.
- **Bankr / Doppler / Scheduled Multicurve**: deployer beneficiary shares, no migration, locked pool, 95/5 fee split.
- **Zora / Virtuals / Pump.fun**: platform-specific, look for official docs via `search_tool`.
- **LayerZero OFT multichain tokens** (source has `import "@layerzerolabs/oft-evm/contracts/OFT.sol"` and `peers(uint32)`, `setPeer`, `send`, `setEnforcedOptions`): standard multichain pattern. `setPeer` by owner is the main live admin power — if signers collude they can add a malicious peer chain and mint via `_credit`. Existing peers matching across chain explorers = legit bridge config. LZ V2 endpoint on Base: `0x1a44076050125825900e736c501f859c50fE728c`.

**Save this step to your report.** The "Claim vs reality" section needs to quote a specific docs capability the team *could* have used and chose not to.

---

## Tool map (roughly in order of use)

1. **`token_search`** (identifier_type=address, chain=...) — baseline market data: price, mcap, volume, 24h change, holder count, security scan flag. Set include_chart=false, include_market_data_image=false to keep context lean.
2. **`get_token_launch_info`** — if the token was deployed via Bankr/Doppler, you get the deployer wallet + twitter handle + tweet URL for free. Always try this first even if you think it's Clanker; the tool returns cleanly on non-match.
3. **`get_contract_abi`** (chain=...) — confirm the address is a real contract and enumerate read/write functions. Flag dangerous functions: `mint`, `crosschainMint`, `setOwner`, `updateAdmin`, `blacklist`, `setFee`, `pause`, `updateImage`, `updateMetadata`. For OFTs, note `setPeer` / `setEnforcedOptions` — these are owner-gated but not exit-scam primitives by themselves.
4. **`read_contract`** for on-chain state. For Clanker v4 tokens:
   - `totalSupply() view returns (uint256)` — expect 100_000_000_000 * 10^18.
   - `allData() view returns (address originalAdmin, address admin, string image, string metadata, string context)` — the single richest read. Tells you deployer admin, current admin (if different = admin handoff happened), whether metadata/socials/audits are populated, and which interface launched it (`clanker.world`, Farcaster, etc.).
   - `isVerified() view returns (bool)` — Clanker's own verification flag.
   - `balanceOf(address) view returns (uint256)` — for any wallet you want to check (admin, top holders, pool manager).
   For OFTs: `owner()`, `peers(uint32)` for each known eid (Ethereum=30101, BSC=30102, Arbitrum=30110, Base=30184, Polygon=30109, Optimism=30111), `endpoint()`, `msgInspector()`, `preCrime()`.
5. **`get_clanker_reward_ownership`** (tokenAddress, chain) — returns every reward recipient/admin slot for the token. **This is the single best signal for "did the team actually allocate ecosystem/marketing/CEX rewards like they claim"** — if it shows only one `{admin, recipient}` both equal to the deployer, the team did NOT configure multi-recipient tokenomics.
6. **Direct viem reads via `execute_cli`** when you need to batch-check balances, detect contract vs EOA (top holders with 48-byte bytecode are AA smart-wallet sniper proxies, not whales), or compute concentration. Install `viem@2.21.55` and run a small `analyze.mjs`.

   **RPC configuration (custom env var is OPTIONAL):**
   - First, call `get_env_vars` to check if the user has set a custom RPC var for the chain you're analyzing (`BASE_RPC_URL`, `MAINNET_RPC_URL`, `ARBITRUM_RPC_URL`, etc.). If present, use it — it's almost always higher-rate-limit.
   - If no custom var is set, **fall back to viem's built-in public RPC** by creating the client with just the chain (no custom `transport` URL). Example:
     ```js
     import { createPublicClient, http } from 'viem';
     import { base } from 'viem/chains';
     const rpcUrl = process.env.BASE_RPC_URL; // may be undefined
     const client = createPublicClient({
       chain: base,
       transport: rpcUrl ? http(rpcUrl) : http(), // http() with no arg uses chain.rpcUrls.default
     });
     ```
     This works for `base`, `mainnet`, `arbitrum`, `optimism`, `polygon`, `bnb`, `unichain`, etc. out of the box.
   - The public RPCs are rate-limited and can be flaky under load. If you see `429` / timeouts when batching many reads, either (a) add small delays between calls, (b) chunk the reads, or (c) tell the user at the end of the report that a custom RPC env var would make re-runs faster and more reliable — don't block the analysis on it.

   Always use this viem step to:
   - Cross-check `totalSupply`, `admin`, `originalAdmin`, `isVerified`, `allData` from the RPC directly (don't trust only indexer outputs).
   - `getBalance` + `getTransactionCount` for each deployer and the claimed Twitter wallet.
   - `getBytecode` on each top non-pool holder — 48 bytes = EIP-7702 / minimal-proxy smart wallet (sniper pattern); >100 bytes = real contract (could be Clanker vault/airdrop extension, or a Gnosis Safe); no code = regular EOA.
   - For Safe multisigs: Safe contracts have a known bytecode prefix and a `VERSION()` / `getOwners()` function — you can detect them specifically.
   - For cross-chain tokens: do balance reads on the OTHER chains too (e.g., if it's on Base but deployed as an OFT, read the Ethereum and BSC mirrors via their RPCs — again, custom env var if set, else viem public default).
   - Concentration math: sum top-20, subtract pool, compute non-pool concentration.
7. **`market_intelligence` with `query_type="holders"`** (chain, token_id=contract address, limit up to 20) — top holder distribution. The #1 holder is almost always the Uniswap v4 PoolManager (`0x498581ff718922c3f8e6a244956af099b2652b2b` on Base). Always identify pool addresses first and exclude them from concentration math.
8. **`get_chain_activity_for_wallet`** on each interesting wallet:
   - The deployer/admin for each contract — look for the lifecycle: funded → deploy → extract → (sometimes) claim fees.
   - Top 3 non-pool holders — were they fresh wallets that sniped at genesis? How were they funded?
   - The claimed Twitter wallet — does it have any footprint on the deploy chain at all?
   - For OFT tokens: check if top holders are sending TO exchange deposit wallets (Coinbase, Binance hot wallets) — this is the distribution-into-bid signal.
9. **`get_evm_address_from_twitter_username`** / `get_evm_address_from_farcaster_username` — resolve the claimed social handle to a wallet. Then check with `get_chain_activity_for_wallet` and viem `getTransactionCount` whether that wallet is the admin of any contract or has any footprint on the deploy chain. Most rugs have social wallets with ZERO activity on the chain they're scamming.
10. **`browse_url`** on basescan/etherscan/polygonscan (`https://basescan.org/address/{addr}`) only as fallback when `get_contract_abi` fails or to confirm an address is a DEX/pool/infra contract. Many docs sites (gitbook-hosted root pages) return empty via firecrawl — go directly to the specific article URLs. After extracting the fact you need, immediately `prune_messages(tool_names=["browse_url"], replacement="<one-line fact>")`.
11. **`perform_technical_analysis`** (only if user is Bankr Club — will error otherwise, don't retry).

---

## Step Final-1 (MANDATORY) — Off-chain intel pass before verdict

A contract can be on-chain-clean and still be a running manipulation case. Before writing the TL;DR verdict, you MUST run a parallel off-chain pass. Run these three calls IN PARALLEL in one turn:

1. **`search_tool`** for investigator / press coverage:
   - Query 1: `"<TOKEN_SYMBOL>" <token_name> scam rug investigation twitter`
   - Query 2: `ZachXBT <TOKEN_SYMBOL> manipulation` (ZachXBT is the most-trusted on-chain investigator in crypto — always check for direct callouts by name)
   - Query 3: `<TOKEN_SYMBOL> pump dump exchange investigation <current_year>` (to catch Bitget/Binance/OKX/Gate investigation announcements)
   - Query 4 (if a team name surfaces): `"<founder name>" <token_name> founder` (to identify the team and any prior-project baggage)
2. **`get_social_sentiment_for_ticker`** — pass the ticker and `additionalContext` describing the token, chain, and whether it's in a parabolic move. Returns community split, funding-rate context, and surfaced allegations.
3. **`browse_url`** on the project's official X/Twitter account if known (format: `https://x.com/<handle>`) — look for self-warnings from the project itself. A team publicly "warning" its own community about "extreme volatility" during a parabolic move is often a soft acknowledgment of insider distribution about to begin.

### Verdict-adjustment rules based on off-chain intel

- **Tier-1 investigator (ZachXBT) has publicly flagged the token as manipulation** → bump verdict up one tier (e.g., LOW-to-MEDIUM → MEDIUM-to-HIGH), even if contract is clean. Note the X post ID and the specific claims.
- **A named CEX (Bitget / Binance / OKX / Gate / Kraken) has confirmed a formal investigation** → bump verdict up one tier AND explicitly flag "CEX delisting / forced-unwind risk" in the TL;DR. This is material.
- **A public whistleblower bounty** ($10k+) has been posted naming the token → treat as a strong manipulation signal; at minimum mention in the TL;DR.
- **Team identified as experienced crypto quant / market-making desk founders** (ZX Squared, Wintermute alumni, known MM firms) combined with insider supply concentration → note "plausibly self-funded MM" structural fit. Not proof, but the correct profile for the tactic.
- **Founder has prior rug/scam/abandoned-project history** → bump verdict up one tier.
- **Insider concentration claim from investigators matches what you measured on-chain** (e.g., investigator says "90%+ in insiders", your sum of top treasury Safes + CEX-bound EOAs ≈ that number) → treat the allegation as corroborated on-chain.
- **Funding rates flipped sharply negative during the pump** + insider CEX deposits beforehand = mechanical short-squeeze tell. Mention in market-structure section.
- **No negative signals and coverage is only standard "listing pump" takes** → leave verdict unchanged but note the cleaner signal in the TL;DR.

If you skip this step and issue a verdict based only on on-chain data, you will systematically under-call insider manipulation cases. This step is not optional.

---

## Red flag checklist (score each; >=5 = high risk)

Contract / configuration-level (AFTER reading platform docs):
- [ ] `isVerified()` = false on the platform's own verification flag (Clanker).
- [ ] `allData()` metadata, socialMediaUrls, auditUrls are all empty.
- [ ] `get_clanker_reward_ownership` shows only ONE recipient = the admin itself (no ecosystem / marketing / multisig recipients), ESPECIALLY when the team's narrative claims those allocations exist.
- [ ] No Vault extension (no tokens held by a Clanker vault contract as a top holder).
- [ ] No Airdrop extension (no Clanker airdrop contract as a top holder).
- [ ] Mutable admin (`updateAdmin`) with no multisig / timelock / renounce.
- [ ] Admin can update image/metadata post-launch (metadata hijack risk).
- [ ] Fee / blacklist / pause / mint functions callable by admin.
- [ ] For OFTs: `setPeer` can still be called by owner and there's no public timelock or DAO process around it.

Deployer-level:
- [ ] Deployer wallet was funded via bridge minutes-to-hours before deploy.
- [ ] Deployer wallet extracts value (bridges ETH/USDC/WETH out) immediately after launch.
- [ ] Deployer wallet has no prior history tying it to a real team.
- [ ] Admin wallet is a fresh EOA (single-digit nonce) rather than a Safe multisig.
- [ ] For a "migration": new contract deployed from a DIFFERENT wallet than the old one, with no on-chain continuity (no direct transfers between the two, no shared funding path).

Holder-level:
- [ ] Top 10 non-pool holders control >10% of supply and bought at genesis with fresh wallets.
- [ ] Several top non-pool holders are 48-byte smart-wallet proxies (AA / 7702 sniper bots).
- [ ] "Pump" on new contract is driven by <5 wallets that each bought >1% of supply in the first blocks.
- [ ] Old contract still has more holders than new — most of the "loyal community" did not actually migrate.
- [ ] For multichain OFTs: a single treasury Safe holds >50% of global supply on the mint chain (even if Base/target-chain concentration looks healthy).
- [ ] Top non-pool holders are actively sending to CEX hot wallets (Coinbase, Binance deposit addresses) during the pump or immediately after.

Narrative vs reality:
- [ ] Claimed social account (Twitter/X) has NO on-chain presence on the deploy chain.
- [ ] Claimed social account is NOT the admin of any contract.
- [ ] Claimed "tokenomics fix" is a capability the platform already supports at deploy time — i.e., a redeploy is technically unnecessary to implement it. (This is where reading Step 0 docs matters.)
- [ ] Migration narrative lacks: snapshot, 1:1 claim contract, old-token LP burn sink. Team just says "re-buy".
- [ ] New contract has the SAME vanilla configuration as the old one — no vault, no airdrop, no multi-recipient rewards — despite the narrative being "new tokenomics".

Off-chain / market-integrity (from Step Final-1):
- [ ] ZachXBT or another tier-1 investigator has publicly flagged the token.
- [ ] A CEX has confirmed a formal investigation.
- [ ] A public bounty has been posted for manipulation evidence.
- [ ] Team has prior rug / abandoned-project history.
- [ ] Team includes a known market-making / quant firm AND insider concentration is high.
- [ ] Project is self-warning its community about "volatility" during a parabolic move.
- [ ] Mainstream crypto press (Coindesk, The Block, Cointelegraph) has pivoted coverage from positive to investigative.

---

## Output structure (write to `/reports/<token_or_project>_scam_analysis.md`)

1. **TL;DR** — Verdict (LOW / MEDIUM / HIGH / EXTREME risk) + one-line reasoning + confidence %. MUST incorporate off-chain intel from Step Final-1 if any triggers hit.
2. **Contracts under analysis** — table with address, deployer, admin, supply, isVerified, metadata completeness, reward-recipient count, vault/airdrop presence, market stats, pool %. Include sibling contracts on other chains for multichain OFTs.
3. **Off-chain intel / social & press coverage** — dedicated section with: key incidents (investigator callouts, CEX investigations, bounties with dates), team identification (founders, prior projects, LinkedIn/RootData links), community sentiment summary, mainstream press coverage links, and net effect on verdict. Always add this section even if findings are neutral.
4. **Claim vs reality** — enumerate each of the team's public claims; under each, state (a) what the chain shows and (b) what the platform's deploy docs say the team *could* have done. Cite the doc URL.
5. **Deployer wallet forensics** — for each deployer: funding source, nonce-by-nonce lifecycle, funding pre-deploy / extraction post-deploy / fee claims post-launch.
6. **Holder distribution** — explicitly identify the pool/DEX holder and exclude it from concentration math; list the top 5–10 real holders with how they acquired their bag (tx hash, ETH paid, which block), and mark each as EOA vs smart-contract. For multichain: include cross-chain treasury holdings.
7. **Contract-level red flags** — pulled directly from the ABI + allData + reward ownership, with function signatures and why each matters.
8. **Economic irrationality test** — what SHOULD a legit team have done? Name the specific platform-native tools (for Clanker: `vault`, `airdrop`, multiple `rewards` recipients, Safe multisig admin, snapshot + claim contract, LP burn). Contrast with what the team actually did.
9. **Pattern match** — does this fit a known grift template (e.g., "Clanker redeploy dump", "sniper-fronted relaunch", "CEX-listing reflex pump with dominant treasury", "self-funded MM short-squeeze")?
10. **What would change the verdict** — list the specific on-chain facts that would lower the risk score. Forces the team to put up or shut up. Also list what would raise it.
11. **Sources** — direct block-explorer links for every contract and wallet referenced, plus the platform docs URLs you read in Step 0 and the investigator/press URLs from Step Final-1.
12. **Appendix** — full raw data pulled (totalSupply, admin, allData, key balances, viem script output, block height of reads).

---

## After saving the file
Return to the user a 4–6 bullet summary with:
- the verdict + confidence (informed by both on-chain AND off-chain intel)
- the 2–3 worst red flags (phrased as facts, not accusations)
- any named tier-1 investigator / CEX investigation / founder identification
- a pointer to the saved report path

Do NOT dump the whole report inline.

## Context management
These analyses pull a lot of data. After you have extracted the facts you need, call `prune_messages` on noisy tools like `browse_url`, `get_contract_abi` (if the ABI was huge), and `get_chain_activity_for_wallet` (these return long tx histories). Pass a short replacement summary so the facts aren't lost.

## Common pitfalls
- **Do not treat the Uniswap v4 PoolManager (or v3 position NFT, or v2 pair) as a whale.** It's the pool. Always identify it first.
- **Do not flag "same bytecode" on a redeploy as a red flag by itself.** Clanker (and most factories) produce identical token runtime code on every deploy. Configuration differences live in the deploy-time payload (vault, airdrop, reward splits, initial market cap, fees). Compare **configuration**, not bytecode.
- **Do not confuse admin balance = 0 with "team has no tokens".** On Clanker/Doppler the whole supply is sold into the pool at launch unless the team explicitly configured a Vault or Airdrop extension. Empty admin wallet is expected; presence of a Clanker Vault/Airdrop contract as a top holder is what would indicate real allocations.
- **Do not trust the Twitter handle without resolving it to a wallet.** A project can claim any handle. What matters is whether that handle's wallet is the admin.
- **Do not read `browse_url` on gitbook docs root URLs** — they often return empty via firecrawl. Go to specific article URLs (e.g., `.../general/token-deployments`, not `.../`). Use `search_tool` to find the article URL first.
- **Do not hard-block on a missing custom RPC env var.** Viem's `http()` transport with no argument falls back to the chain's default public RPC (rate-limited but functional). Only require a custom `*_RPC_URL` if the user explicitly wants high-volume reads or you hit rate limits mid-analysis — then suggest setting one for next time.
- **If the user is not in Bankr Club, `perform_technical_analysis` will error.** Don't retry — use `browse_url`, `market_intelligence`, and contract reads instead.
- **Do not issue a verdict based only on on-chain data.** Always run the Step Final-1 off-chain intel pass first. A contract can be clean and the token still be an active manipulation case. RAVE (April 2026) is the canonical example: clean LayerZero OFT, clean deployer, multisig owner, and simultaneously ZachXBT-flagged + Bitget-investigated as a coordinated pump-and-dump.
- **For multichain OFTs, always check the mint chain's treasury, not just the chain the user asked about.** A "healthy" Base holder distribution can be irrelevant if 75% of global supply sits in one Safe on Ethereum.
