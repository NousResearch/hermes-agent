# Small-Model Variant Pattern

How to create a compact, deterministic variant of an existing LAH Stack workflow skill optimized for small/fast reasoning models (Ling 3 Flash, etc.).

## When to Use

When the canonical workflow skill is known to work with stronger models but a small/fast model variant is needed for:
- InclusionAI Ling 3 Flash or similar small models
- Hermes + LCM long-running sessions where execution efficiency matters
- Scenarios where the canonical skill's discovery/exploration steps cause excessive tool calls

## Design Principles

1. **Same role, different execution style** — the variant tells Hermes the same thing as the canonical skill but with tighter bounds.
2. **No new conceptual layer** — the variant has the same role as the canonical skill.
3. **Explicit over implicit** — every instruction is explicit, bounded, and deterministic.
4. **Defer to canonical for details** — mission-type branching, pitfall resolution, and detailed reference patterns are delegated to the canonical skill.

## Creation Process

1. **Audit the canonical skill** — read the full SKILL.md, classify instructions into hard invariants, strong-model judgment, discovery/exploration, duplication, explanatory material, and conditional detail.
2. **Define execution modes** — classify every mission into exactly one of EXECUTE/DIAGNOSTIC/REPAIR/CERTIFY. No silent mode switching.
3. **Define authority order** — create a deterministic 7-level hierarchy (operator > constraints > runbook > workflow > domain > tool > historical).
4. **Add certified runbook fast path** — when a certified runbook exists, load it verbatim and execute its steps. No reconstruction from history.
5. **Add search budget** — explicit bounded discovery policy. EXECUTE: max 1 narrow lookup. CERTIFY: max 2 narrow evidence-resolution actions. DIAGNOSTIC/REPAIR: bounded but allowed.
6. **Add evidence contract** — compact evidence classes with strict hierarchy (runtime state > provider state > authorization > safety > deployment > certified history).
7. **Add stop contract** — explicit STOP conditions with prescribed actions. No improvisation after STOP.
8. **Add LCM context discipline** — compact working state format, no full historical mission reloads, no repeated reconstruction of certified facts.
9. **Preserve essential governance** — authority order, gate sequence, STOP semantics, safety constants, sub-agent discipline, commit conventions.
10. **Validate** — verify the original canonical skill is byte-for-byte unchanged, the new skill is discoverable, and behavioral tests pass.

## Key Differences from Canonical

| Aspect | Canonical | Small-Model Variant |
|--------|-----------|---------------------|
| Discovery | Unbounded | Bounded (1-2 narrow lookups) |
| Mode switching | Implicit | Explicit with STOP |
| Evidence | Broad | Contract with strict hierarchy |
| Runbook | Reconstructed | Loaded verbatim (fast path) |
| Context | Full historical | Compact working state |
| Pitfall encyclopedia | Inlined | Deferred to canonical references |
| Mission-type branching | Inlined | Deferred to canonical skill |

## Pitfalls

- Don't duplicate the canonical skill's pitfall encyclopedia — defer to it.
- Don't create a separate "small model operator" abstraction — the variant has the same role as the canonical skill.
- Don't modify the canonical skill — the variant is a separate, independent skill.
- Don't include T03-specific business logic in the variant — use T03 only as an example of the certified-runbook fast path.