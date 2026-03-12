# Data Layer

Solana's data layer covers how account state and transaction data are stored, propagated, and streamed to external consumers. RPC nodes maintain full account state and serve JSON-RPC queries; Geyser plugins stream updates from inside the validator as they happen; shreds are the primitive unit of block propagation; and compression (state compression + ZK compression) makes storing large datasets on-chain economically viable.

## Key Concepts

- **RPC node** — a full replay node that maintains complete account state; serves `getAccountInfo`, `getTransaction`, `getProgramAccounts`, and other JSON-RPC methods; not a voting validator
- **Geyser plugin** — a shared library loaded by a validator at startup; receives real-time callbacks for account updates, transaction notifications, slot changes, and block completions — before they're finalized; powers Helius webhooks and Laserstream
- **AccountsDB** — Solana's account storage system: accounts stored in append-only "account files" with background compaction; hot accounts cached in memory
- **Shred** — the atomic unit of block data: ≈1.2 KB fragments of a serialized block, Reed-Solomon erasure coded (data shreds + code shreds); sent via Turbine; validators can reconstruct blocks even with significant packet loss
- **Ledger** — the complete history of all blocks and transactions; RPC nodes maintain this; pruned for most nodes after a configurable number of slots; Bigtable archives historical data at Solana Foundation
- **State compression** — stores Merkle tree account hashes on-chain (cheap) with off-chain leaf data; enables millions of compressed NFTs for fractions of a cent each; used by cNFTs
- **ZK compression** — zero-knowledge proofs compress arbitrary state to a constant on-chain footprint; enables scalable token balances and other state without per-account rent
- **DoubleZero** — network infrastructure project to provide dedicated low-latency links between validators; reduces inter-validator latency and improves block propagation
- **Zero Slot** — block explorer and monitoring tooling focused on Solana slot-level data

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `how-solana-rpcs-work` — RPC node internals: how they replay transactions, maintain state, serve queries, and differ from validators
- `solana-rpc` — Practical RPC guide: endpoints, methods, commitment levels, and how Helius extends standard RPC
- `solana-geyser-plugins-streaming-data-at-the-speed-of-light` — Geyser plugin architecture: plugin interface, what data is available, latency characteristics, and use cases
- `solana-data-streaming` — Data streaming overview: Geyser vs webhooks vs WebSockets and when to use each
- `solana-shreds` — Shreds deep dive: structure, erasure coding, how Turbine uses them, and why shred-level data is valuable for low-latency applications
- `all-you-need-to-know-about-compression-on-solana` — State compression: how Merkle trees work on-chain, cNFTs, and the cost comparison
- `zk-compression-keynote-breakpoint-2024` — ZK compression: how zero-knowledge proofs enable scalable compressed state
- `doublezero-a-faster-internet` — DoubleZero network: dedicated validator infrastructure and its impact on propagation latency
- `solana-post-quantum-cryptography` — Post-quantum cryptography considerations for Solana's long-term security
- `zero-slot` — Zero Slot explorer: slot-level data access and what makes it useful for analytics

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `geyser-plugin-manager/src/geyser_plugin_manager.rs` — Plugin manager: how Geyser plugins are loaded and dispatched
- `ledger/src/shred.rs` — Shred structure and erasure coding logic
- `accounts-db/src/accounts_db.rs` — AccountsDB: storage, compaction, and cache management

## Solana Docs

Try `searchSolanaDocs` with: "geyser plugin", "state compression", "rpc methods", "shreds", "accountsdb"

## See Also

- `references/consensus.md` — Turbine uses shreds for block propagation
- `references/validators.md` — RPC nodes vs validators: different roles, same data
