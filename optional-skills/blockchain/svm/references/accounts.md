# Account Model & Programming Model

Solana's account model is the foundation of its programming paradigm: all state lives in accounts, programs are stateless, and data ownership is enforced by the runtime. This model — combined with upfront account declarations — is what enables Sealevel's parallel execution. Key abstractions built on top of it (PDAs, CPIs, syscalls) give programs composability and the ability to sign without private keys.

## Key Concepts

- **Account structure** — 32-byte address (pubkey), lamport balance, arbitrary data buffer, owner program ID, `executable` flag, `rent_epoch`
- **Ownership** — only the owner program can write to an account's data or debit lamports; any program can read any account; ownership transfers are possible
- **Rent** — accounts must maintain a minimum lamport balance (rent-exempt threshold ≈ 0.00089 SOL per byte); below threshold, accounts are purged from state
- **PDA (Program Derived Address)** — deterministic address derived from `program_id + seeds`; has no private key so only the program can sign for it via `invoke_signed`; used for vaults, mint authorities, config accounts
- **CPI (Cross-Program Invocation)** — a program calling another program's instruction; same transaction, same atomicity; max CPI depth = 4; the callee sees its own account subset
- **Syscalls** — the boundary between program and runtime: `sol_log_`, `sol_sha256`, `sol_invoke_signed_`, `sol_get_clock_sysvar`, etc.; each is a stable ABI callable from sBPF
- **Sysvars** — special read-only accounts with runtime data: `Clock` (slot, epoch, unix timestamp), `Rent` (rent parameters), `EpochSchedule`, `RecentBlockhashes`
- **Slot** — the smallest time unit: ~400ms average; a leader produces one block per slot (or skips); slots group into epochs (~2.5 days, ~432,000 slots)
- **Commitment levels** — `processed` (leader received), `confirmed` (supermajority voted), `finalized` (32 confirmed blocks on top; irreversible)
- **Asynchronous execution** — Solana processes transactions without global ordering; programs must be designed for concurrent, non-sequential state access

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `solana-pda` — PDAs: derivation algorithm, canonical bumps, use cases (vaults, authorities, indexed accounts), and `find_program_address` vs `create_program_address`
- `the-solana-programming-model-an-introduction-to-developing-on-solana` — Full programming model overview: accounts, instructions, ownership, and how programs interact with state
- `solana-slots-blocks-and-epochs` — Time model: slots, blocks, epochs, leader schedules, and how they relate
- `solana-commitment-levels` — Processed vs confirmed vs finalized: when to use each and the tradeoffs
- `asynchronous-program-execution` — Why Solana's concurrency model is fundamentally different from sequential blockchains
- `solana-vs-sui-transaction-lifecycle` — Compares Solana and Sui's execution models; illuminates what makes Solana's account declaration approach unique

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `runtime/src/bank.rs` — the Bank: processes transactions, manages account state, applies rent, distributes rewards

**Note**: `sdk/program/src/account_info.rs` (AccountInfo struct) and `sdk/program/src/program.rs` (`invoke`/`invoke_signed` CPI primitives) live in `solana-labs/solana`, not agave — `readSolanaSourceFile` cannot fetch them.

## Solana Docs

Try `searchSolanaDocs` with: "program derived address", "cross program invocation", "account model", "rent exempt", "commitment"

## See Also

- `references/transactions.md` — how upfront account declarations enable Sealevel parallel execution
- `references/execution.md` — how the runtime enforces ownership and memory isolation during execution
