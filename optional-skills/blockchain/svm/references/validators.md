# Validator Economics

Validators are the backbone of Solana's security — they stake SOL, vote on blocks, and earn rewards for honest participation. The economics are designed to incentivize both validators (via commission) and delegators (via staking APY), funded through inflation that decreases over time toward a 1.5% floor. Slashing (penalizing misbehavior by destroying stake) exists in proposals but is not yet live on mainnet.

## Key Concepts

- **Validator** — a full node that holds the full ledger, participates in consensus voting, and optionally produces blocks as a leader
- **Stake** — SOL locked in a stake account delegated to a validator; stake-weighting determines vote influence and reward share
- **Epoch rewards** — distributed at each epoch boundary; calculated as: `vote_credits × delegated_stake × epoch_inflation_rate × (1 - validator_commission)`
- **Vote credits** — earned by validators for timely, correct votes; a proxy for uptime and network participation quality
- **Commission** — the percentage of epoch rewards a validator keeps before passing the rest to delegators; set by the validator (0-100%)
- **Inflation schedule** — starts at 8% annual issuance, decreases 15% per year, floors at 1.5%; new SOL minted each epoch to fund rewards
- **Slashing** — proposed mechanism to destroy a portion of stake for provable misbehavior (duplicate voting, equivocation); currently not active; SIMD-0085 defines the initial design
- **RPC nodes** — non-voting nodes that maintain full ledger state and serve JSON-RPC requests; economically different from validators (no rewards, significant hardware cost)
- **Nakamoto coefficient** — minimum number of validators (by stake) needed to control 33% of stake (halt consensus); a decentralization metric; Solana's is ≈19-25
- **SIMD governance** — Solana protocol changes go through the SIMD (Solana Improvement Document) process; community discussion + validator vote via mainnet feature flags

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `solana-validator-economics-a-primer` — Complete validator economics: rewards, commissions, vote costs, inflation, and the economics of running a validator
- `solana-nodes-a-primer-on-solana-rpcs-validators-and-rpc-providers` — Distinction between validators, RPC nodes, and RPC providers; hardware requirements and economics for each
- `solana-issuance-inflation-schedule` — Inflation schedule mechanics: current rate, annual decrease, floor, and how new SOL is minted
- `bringing-slashing-to-solana` — Slashing proposal: why it matters for security, what misbehavior would be penalized, current status
- `solana-decentralization-facts-and-figures` — Nakamoto coefficient, geographic distribution, client diversity, and how Solana compares to other L1s
- `solana-governance--a-comprehensive-analysis` — How Solana governance works: SIMDs, feature flags, the role of validators in activating changes
- `simd-228` — SIMD-0228 analysis: validator revenue sharing and its implications for economics

## Relevant SIMDs

Use `getSIMD` for:

- SIMD-0228 — Validator revenue sharing: distributes a portion of MEV/priority fees to all consensus voters
- SIMD-0085 — Slashing: initial design for penalizing duplicate block production

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `programs/vote/src/vote_processor.rs` — Vote program: how validators cast votes and accumulate credits
- `runtime/src/stakes.rs` — Stake tracking: delegation, activation, deactivation, and reward distribution

## Solana Docs

Try `searchSolanaDocs` with: "validator rewards", "inflation schedule", "stake delegation", "slashing", "vote credits"

## See Also

- `references/consensus.md` — Tower BFT and how votes are used in fork choice
- `references/data.md` — RPC nodes: the data access layer validators and users rely on
