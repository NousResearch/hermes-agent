# Execution Engine

When a transaction invokes a program, the validator loads the program's sBPF bytecode and executes it inside an isolated sandbox. The execution engine JIT-compiles sBPF to native machine code for near-native performance, enforces strict memory isolation across distinct regions, and meters every instruction against a compute unit budget. Determinism is guaranteed through the sBPF ISA's constraints: no floating point, no undefined behavior, bounded memory access.

## Key Concepts

- **sBPF ISA** — 11 64-bit registers (r0=return, r1-r5=args, r6-r9=callee-saved, r10=frame pointer); 64-bit and 32-bit opcodes; register-based (not stack-based like JVM)
- **JIT compilation** — the validator translates sBPF opcodes to native x86-64 (or AArch64) at load time; cached per program; eliminates interpretation overhead
- **Compute units (CU)** — each instruction costs CUs; default budget 200,000 CU/tx; `SetComputeUnitLimit` can increase up to 1.4M; metering prevents denial-of-service
- **Memory regions** — four isolated regions per invocation: program code (read-only), call stack (4 KB frames, max 64 frames), heap (32 KB, zero-initialized), input (serialized account data, read/write per account permissions)
- **Memory bounds checking** — every load/store is validated against region boundaries at JIT time; out-of-bounds = program error
- **Call stack limit** — max 64 frames deep (includes CPI chains); prevents unbounded recursion
- **Syscalls** — privileged operations available to programs via stable IDs: `sol_log_`, `sol_invoke_signed_`, crypto primitives (SHA-256, ed25519, secp256k1), sysvar access
- **Determinism** — no floating point instructions, no randomness syscalls, hash-stable dispatch IDs (Murmur3) for syscalls; same bytecode + same accounts → same result on every validator

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `solana-virtual-machine` — Full pipeline overview: JIT compilation, memory region isolation, compute unit metering, and sBPF register design; use this for general execution questions
- `sbpf-assembly` — sBPF ISA deep dive: opcodes, calling conventions, memory layout, and low-level optimization; use this for JIT/ISA/assembly-level questions
- `solana-arithmetic` — Numeric types in Solana programs: integer overflow, checked math, and why floating point is forbidden

## Relevant SIMDs

Use `getSIMD` for:

- SIMD-0161 — sBPF v2: new arithmetic opcodes, 32-bit sign extension, improved code density
- SIMD-0178 — Static syscall IDs: deterministic Murmur3 hashes replace positional dispatch

## Source Code Entry Points

Use `readSolanaSourceFile` (repo: agave) to explore:

- `svm/src/transaction_processor.rs` — core SVM orchestration: transaction scheduling, account loading, and parallel execution dispatch
- `program-runtime/src/invoke_context.rs` — the execution context: account access, CPI dispatch, compute metering; search for `pub struct InvokeContext` to jump to the key struct (file is ~3,900 lines)

## Solana Docs

Try `searchSolanaDocs` with: "compute units", "sbpf registers", "memory regions", "jit compilation solana"

## See Also

- `references/compilation.md` — how sBPF bytecode is produced from source
- `references/programs.md` — how bytecode is deployed and loaded before execution
- `references/accounts.md` — CPIs and syscalls from the programming model perspective
