# Live Trust-Kernel Upgrade and Activation Pattern

Use this reference when a core storage/governance invariant changes across a library, MCP adapter, hooks, installed binaries, and running services.

## Why source-green is insufficient

A core invariant can pass every library test yet break adapters that still construct legacy inputs or call obsolete mutation paths. Typical examples:

- canonical writes begin requiring immutable origin labels, while MCP/HTTP adapters still create unlabeled permits;
- immutable provenance prevents legacy physical deletion;
- a full-profile admin tool uses its own service principal against a resource created by another adapter;
- the current agent session retains an old MCP process after a new binary is installed.

Treat these as integration defects, not reasons to weaken the core invariant.

## Required rollout sequence

1. **Core RED/GREEN**
   - Add focused tests for the new invariant.
   - Run dependent authority, temporal, retrieval, forgetting, and hostile suites.

2. **Adapter contract audit**
   - Search every adapter for constructors and mutation calls affected by the invariant.
   - Add adapter-level tests, not only core tests.
   - Preserve distinctions such as external evidence versus operator override.

3. **Deletion-semantics audit**
   - If provenance/origin rows are immutable, legacy hard-delete APIs are no longer valid for governed records.
   - Route user-facing deletion through dependency-closure forgetting.
   - Return a content-free tombstone/closure receipt.
   - Keep raw compatibility APIs explicitly named and outside governed paths.

4. **Cross-adapter principal test**
   - Create through adapter A and forget through admin adapter B.
   - Bind authorization to the immutable resource-origin principal while retaining adapter B as the audited caller.
   - Do not allow an adapter to substitute its own principal for the resource principal.

5. **Build and install verification**
   - Resolve the actual Cargo `target_directory`; do not assume parent-workspace output paths.
   - Hash the built binary and every installed copy.
   - Stop services only after the build artifact is verified.

6. **Fresh-process verification**
   - Existing MCP tool handles may remain connected to an old process after installation.
   - Verify with a newly spawned MCP client/process, not the current session handle alone.
   - Distinguish: source proof, installed-binary proof, fresh-process proof, and current-session proof.

7. **Live canary lifecycle**
   - Governed write → receipt/origin/epoch checks.
   - Namespace-scoped witnessed retrieval → exact canary found.
   - Governed forgetting → closure receipt and all surfaces checked.
   - Post-forget search → absent; raw tombstone and revocation present.
   - Clean up canaries through the same governed path being certified.

8. **Hook smoke**
   - Confirm injected context is provenance-framed DATA ONLY.
   - Confirm retrieval receipt references are present.
   - Confirm unsafe/background evidence is filtered rather than merely labeled unsafe.

## Regression design lessons

- Test both classification and admission. A classifier can correctly return `safe=false` while a downstream gate still admits the item.
- Test resource creation and deletion across different adapters/principals.
- Test the freshly installed executable directly; current-session MCP failures may be stale-connection failures.
- When a compatibility tool conflicts with a stronger invariant, update the tool semantics instead of adding a bypass.

## Claim boundary

Do not say “active” from source tests or a successful build. Require matching installed hashes, healthy restarted services, fresh MCP discovery, and a complete live write/retrieve/forget cycle.
