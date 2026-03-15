# Program Deployment

After a Solana program is compiled to sBPF bytecode, it must be deployed to the blockchain before it can be called. Deployment is a multi-transaction process that uploads the ELF binary into on-chain accounts and marks the program as executable. Solana has evolved through four BPF loader versions, each with different account models, upgrade capabilities, and security tradeoffs.

## Key Concepts

- **BPF Loader (Legacy / V1)** — original loader; immutable programs, single account; still exists but deprecated
- **BPF Loader V2** — added upgradability; rarely used directly
- **BPF Upgradeable Loader (V3)** — current standard; two-account model: `Program` account (executable, stores ProgramData address) + `ProgramData` account (stores bytecode + upgrade authority)
- **BPF Loader V4** — simplified back to single account; uses "retract" instead of close; not yet default
- **Upgrade authority** — the keypair permitted to replace program bytecode; can be set to null to make immutable
- **ELF format** — the binary format used; contains code sections, relocation tables, and symbol metadata
- **Deploy process** — bytecode is chunked into ~1KB write transactions (due to tx size limits), then a final `finalize` call marks it executable
- **Program account** — the publicly known address users invoke; always marked `executable = true`
- **ProgramData account** — stores the actual bytecode; derived from the Program address; owned by the loader
- **Closing programs** — recover lamports by closing ProgramData and the Program account; requires authority

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `solana-virtual-machine` — Covers the program deployment model, BPF loader evolution (V1→V4), the two-account model, ELF structure, and how static verification works before a program is marked executable

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `programs/bpf_loader/src/lib.rs` — the upgradeable BPF loader implementation: deploy, upgrade, close instructions

**Note**: `sdk/program/src/bpf_loader_upgradeable.rs` (client-side instruction builders) lives in `solana-labs/solana`, not agave — `readSolanaSourceFile` cannot fetch it.

## Solana Docs

Try `searchSolanaDocs` with: "bpf loader upgradeable", "program deployment", "upgrade authority", "program account"

## See Also

- `references/compilation.md` — how Rust source becomes the sBPF ELF binary that gets deployed
- `references/execution.md` — what happens when a deployed program is invoked and executed
