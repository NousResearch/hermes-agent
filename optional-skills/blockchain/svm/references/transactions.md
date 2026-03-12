# Transactions & Local Fee Markets

Solana's transaction design contains its most consequential architectural decision: all accounts must be declared upfront before execution begins. This single constraint enables conflict detection without locking, which enables Sealevel's parallel execution across CPU cores. It also enables local fee markets — fee pressure is scoped to the accounts a transaction touches, so a congested token swap doesn't raise fees for an unrelated NFT mint.

## Key Concepts

- **Upfront account declarations** — every transaction lists all accounts (read and write) before any instruction runs; the runtime uses this to detect conflicts without locks; the defining design choice that enables parallelism
- **Sealevel** — Solana's parallel transaction processing engine; schedules non-conflicting transactions across all available CPU cores simultaneously; named after the sea (parallel, vs EVM's sequential "EVM" → Ethereum Virtual Machine)
- **Local fee markets** — fee pressure is per-account (or per-program); high contention on one hot account raises priority fees only for transactions touching that account; unrelated transactions are unaffected
- **Transaction structure** — signatures array, message (header, account keys, recent blockhash, instructions); account keys list determines parallelism; max 1232 bytes per transaction
- **TPU pipeline** — Fetch Stage (receive txs via QUIC) → SigVerify Stage (parallel signature verification on GPU) → Banking Stage (parallel execution via Sealevel) → Broadcast (send block to network)
- **Base fee** — 5,000 lamports per signature; burned (50%) + validator reward (50%) pre-SIMD-0096; 100% to validators post-SIMD-0096
- **Priority fee** — optional `ComputeBudgetProgram.setComputeUnitPrice` in micro-lamports per compute unit; validators prioritize higher-fee transactions within a slot
- **Compute unit limit** — set via `SetComputeUnitLimit`; default 200,000; max 1.4M per transaction
- **SWQoS (Stake-Weighted Quality of Service)** — validators reserve QUIC connection slots proportional to stake; transactions forwarded by staked validators get preferential treatment
- **Blockhash expiry** — recent blockhash expires after ~150 slots (~60-90 seconds); prevents replay; use durable nonces for longer-lived transactions
- **MEV on Solana** — block producers (leaders) can reorder transactions within a slot; Jito's block engine enables MEV extraction via bundles and tip auctions

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `solana-transactions` — Transaction structure anatomy: message format, account keys, instructions, and how serialization works
- `priority-fees-understanding-solanas-transaction-fee-mechanics` — Priority fee mechanics: how `setComputeUnitPrice` works, fee calculation, and validator incentives
- `solana-fees-in-theory-and-practice` — Complete fee model: base fees, priority fees, rent, and what users actually pay
- `solana-local-fee-markets` — Local fee markets deep dive: why contention is per-account, how the scheduler works, and implications for dApp design
- `how-to-land-transactions-on-solana` — Practical guide to transaction landing: priority fees, retries, confirmation strategies
- `how-to-deal-with-blockhash-errors-on-solana` — Blockhash expiry, prefetching, and durable nonces
- `solana-congestion-how-to-best-send-solana-transactions` — Congestion periods: why transactions fail and how to improve landing rates
- `solana-mev-an-introduction` — MEV on Solana: how Jito bundles work, tip accounts, and the MEV landscape
- `stake-weighted-quality-of-service-everything-you-need-to-know` — SWQoS: how stake affects transaction routing and why it matters for landing rates
- `block-assembly-marketplace-bam` — BAM (Block Assembly Marketplace): the next evolution of Solana's fee market design (in-progress proposal)

## Relevant SIMDs

Use `getSIMD` for:

- SIMD-0096 — Priority fees 100% to validators (removes 50% burn); changes validator incentives
- SIMD-0123 — Block revenue sharing: distributes fees across validators who participate in consensus

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `runtime/src/bank.rs` — Banking Stage: where transactions are scheduled and executed in parallel
- `fee/src/lib.rs` — Fee calculation logic: base fee, priority fee, compute unit pricing

## Solana Docs

Try `searchSolanaDocs` with: "sealevel parallel", "local fee markets", "priority fees", "transaction structure", "compute budget"

## See Also

- `references/accounts.md` — account model and ownership (the foundation that upfront declarations build on)
- `references/consensus.md` — how the TPU output becomes a confirmed block via Tower BFT
