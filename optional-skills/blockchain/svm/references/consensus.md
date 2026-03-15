# Consensus

Solana's consensus stack combines several novel protocols: Proof of History (PoH) as a verifiable clock, Tower BFT for fork choice with exponential lockout, Turbine for block propagation, and Gulf Stream for mempool-less transaction forwarding. The network currently has two independent validator clients (Agave and Firedancer), with Alpenglow proposed as a next-generation consensus replacement that eliminates PoH as a consensus input.

## Key Concepts

- **Proof of History (PoH)** — a sequential SHA-256 hash chain that acts as a verifiable delay function (VDF); creates a cryptographic timestamp for every event; enables validators to agree on time ordering without communication
- **Tower BFT** — a PBFT variant designed around PoH; validators lock votes on forks with exponentially increasing lockout (2^n slots); once locked in, switching forks costs proportional stake loss via slashing (future)
- **Turbine** — block propagation protocol using erasure-coded shreds (≈1.2 KB fragments) distributed through a tree topology; each validator receives shreds from a neighborhood and re-broadcasts; tolerates up to 1/3 packet loss via erasure coding
- **Gulf Stream** — transaction forwarding: clients send transactions directly to the expected leader (known via the published leader schedule) rather than a mempool; reduces confirmation latency and buffering
- **QUIC** — the network transport layer (replaced raw UDP); provides congestion control, connection multiplexing, and stream prioritization; underpins SWQoS connection allocation
- **Leader schedule** — a deterministic rotation of which validator produces blocks in each slot; published one epoch in advance; stake-weighted
- **Fork** — when two validators produce competing blocks for the same slot; Tower BFT resolves forks via supermajority vote on the heaviest fork
- **Supermajority** — 2/3 of stake-weighted votes required for confirmation and finality
- **Firedancer** — Jump Crypto's independent, high-performance validator client written in C; targets 1M TPS; currently on testnet (Frankendancer, a hybrid, is on mainnet)
- **Alpenglow** — proposed consensus overhaul (SIMD-0232): replaces Tower BFT with Votor (fast voting) + Rotor (block propagation); eliminates PoH from consensus path; targets ~150ms finality

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `consensus-on-solana` — Complete consensus overview: PoH, Tower BFT, leader schedule, forks, and how finality is achieved
- `proof-of-history-proof-of-stake-proof-of-work-explained` — Conceptual comparison of PoH vs PoS vs PoW and what role each plays
- `turbine-block-propagation-on-solana` — Turbine deep dive: shreds, erasure coding, tree topology, and how large blocks propagate efficiently
- `solana-gulf-stream` — Gulf Stream: why eliminating the mempool reduces latency and how transaction forwarding works
- `all-you-need-to-know-about-solana-and-quic` — QUIC adoption: why raw UDP was replaced, how QUIC improves reliability and SWQoS
- `cryptographic-tools-101-hash-functions-and-merkle-trees-explained` — Cryptographic foundations: hash functions and Merkle trees as used in PoH and Turbine
- `what-is-firedancer` — Firedancer overview: Jump Crypto's client, its architecture, current status, and impact on network diversity
- `alpenglow` — Alpenglow proposal: Votor voting protocol, Rotor propagation, how it differs from Tower BFT, and expected timeline

## Relevant SIMDs

Use `getSIMD` for:

- SIMD-0083 — Relax leader schedule entry requirements
- SIMD-0232 — Alpenglow consensus protocol (Votor + Rotor); replaces Tower BFT + PoH for consensus

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `core/src/consensus.rs` — Tower BFT policy layer: vote recording (`record_bank_vote_and_update_lockouts`), stake threshold checks (`check_vote_stake_threshold`), and fork switching logic
- `core/src/consensus/tower_vote_state.rs` — Raw lockout state machine: exponential doubling (`double_lockouts`), vote stack management (`process_next_vote_slot`, `pop_expired_votes`)
- `core/src/consensus/tower_storage.rs` — Tower persistence: how vote state is serialized and saved to disk (not lockout mechanics)
- `ledger/src/shred.rs` — Shred structure: how blocks are split for Turbine propagation

## Solana Docs

`searchSolanaDocs` covers developer-facing API content only (accounts, transactions, programs, PDAs, RPC methods). It does **not** index consensus/protocol content — queries like "proof of history", "tower bft", "turbine", and "gulf stream" return no results. Skip this tool for consensus topics and rely on `fetchHeliusBlog` and `readSolanaSourceFile` instead.

Useful queries for adjacent topics: "transaction fees", "account model", "program deployment", "RPC methods"

## See Also

- `references/validators.md` — validator economics, stake weighting, and governance
- `references/transactions.md` — TPU pipeline and how transactions flow into blocks
- `references/data.md` — shreds from the data propagation perspective
