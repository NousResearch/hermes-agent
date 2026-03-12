# Program Development

The Solana program development ecosystem has matured significantly, with multiple competing frameworks offering different tradeoffs between safety, performance, and developer experience. Anchor remains the dominant choice for most projects, while leaner alternatives (Steel, Pinocchio) target performance-critical programs. On the client side, the web3.js 2.0 SDK introduces a functional, tree-shakeable architecture.

## Key Concepts

- **Anchor** — the most widely-used Solana framework; uses procedural macros to auto-generate account validation, serialization (Borsh), and a TypeScript IDL; opinionated but safe; best for most production programs
- **Steel** — lightweight Anchor alternative; minimal macros, low overhead, still provides account validation helpers; targets developers who want less "magic"
- **Pinocchio** — zero-dependency, maximum-performance framework; no Anchor, no alloc; used for programs where every CU counts (lending protocols, DEXs); requires manual account parsing
- **Gill** — TypeScript client library for Solana programs; functional API, tree-shakeable; similar philosophy to web3.js 2.0
- **web3.js 2.0** — the new official Solana TypeScript SDK: functional (no classes), tree-shakeable, composable; `@solana/web3.js` v2; replaces the old Connection/PublicKey/Transaction class-based API
- **Borsh** — the serialization format used by most Solana programs and Anchor; deterministic binary encoding; IDL-based for cross-language compatibility
- **IDL (Interface Definition Language)** — Anchor's JSON description of a program's accounts, instructions, and types; enables auto-generated clients and type-safe interactions
- **Compute unit optimization** — techniques: minimize account reads, use lookup tables (ALTs) to compress account lists, avoid dynamic dispatch, prefer u64 over u128, use zero-copy deserialization
- **Address Lookup Tables (ALTs)** — on-chain tables of account addresses; allow a transaction to reference up to 256 accounts using 1-byte indices instead of 32-byte addresses; critical for complex multi-account transactions
- **sBPF version** — programs are compiled targeting a specific sBPF version (v1, v2, v3); validators must support the target version; newer versions add opcodes but require validator adoption

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `optimizing-solana-programs` — CU optimization guide: profiling, reducing account lookups, efficient serialization, and common bottlenecks
- `steel` — Steel framework: how it differs from Anchor, when to use it, and migration considerations
- `pinocchio` — Pinocchio: zero-dependency approach, raw account parsing, and the performance gains at the cost of safety abstractions
- `gill` — Gill TypeScript SDK: API design, comparison to web3.js 2.0, and use cases
- `an-introduction-to-anchor-a-beginners-guide-to-building-solana-programs` — Anchor introduction: account constraints, instruction handlers, and the derive macro system
- `how-to-start-building-with-the-solana-web3-js-2-0-sdk` — web3.js 2.0 guide: functional API, RPC client setup, transaction building, and signing

## Relevant SIMDs

Use `getSIMD` for:

- SIMD-0161 — sBPF v2 instruction set: new opcodes available to programs compiled for v2
- SIMD-0178 — Static syscall IDs: stable ABI for program-to-runtime syscall dispatch

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `programs/system/src/system_processor.rs` — System program: create accounts, transfer SOL, assign ownership
- `svm/src/transaction_processor.rs` — core SVM orchestration: transaction scheduling, account loading, and parallel execution dispatch

## Solana Docs

Try `searchSolanaDocs` with: "anchor framework", "program development", "compute units optimization", "address lookup tables", "web3.js 2"

## See Also

- `references/compilation.md` — how programs are compiled to sBPF before development tools matter
- `references/programs.md` — deploying and upgrading the programs you build
- `references/execution.md` — understanding compute units and memory at the runtime level
