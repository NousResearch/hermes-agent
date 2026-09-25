# Engineering Discipline for HDA

This reference distills the build discipline used to implement Hermes Digital Assistant. It is intentionally smaller than the behavioral specification.

## 1. Current reality outranks the plan

Inspect the actual running Hermes before designing changes. Documentation, prior sessions, remembered paths, and older HDA implementations are leads, not truth. Record exact versions, state, and capabilities that materially affect the design.

## 2. Design a deep module with a small interface

Prefer a small number of stable seams that each carry substantial behavior. Avoid scattering equivalent rules across prompts, adapters, wrappers, and scripts. If one native hook/state mechanism can own the invariant, put the invariant there.

## 3. Put contracts at boundaries

Treat these as explicit boundaries:

- user preference versus user grant,
- memory versus current evidence,
- internal reasoning versus external action,
- reversible versus irreversible action,
- shared-chat context versus owner authority,
- built files versus running process,
- installer success versus behavioral success.

Validate at the boundary instead of hoping a long prompt keeps every distinction intact.

## 4. Build vertical, verifiable slices

Implement one behavior from input to observable outcome, test it, then continue. Start with the smallest high-leverage slice from the method's build order. A vertical test beats a wide pile of unexercised plumbing.

## 5. Preserve before improving

Before touching a live install, inventory and protect unrelated local state. Back up the exact things being changed. Do not use destructive reset/clean operations to make the target resemble an assumed baseline.

## 6. Gather evidence, do not self-certify

For every completion claim, identify the observable evidence that could prove it false. Source inspection proves code/state exists; behavior tests prove it works; restart/live-process checks prove it is active. Use the strongest available oracle.

## 7. Attack the premise and subtract first

Before adding a subsystem, ask whether Hermes already has a native mechanism that can be configured or composed to satisfy the requirement. Remove duplicate old mechanisms when migrating to a better one.

## 8. Encode recurring lessons structurally

When a failure repeats, prefer a gate, state field, lifecycle hook, test, schema, or config invariant over another paragraph of prompt instructions. Prompts are appropriate for semantic judgment; enforcement belongs in code/state when the platform exposes the boundary.

## 9. Independent falsification

Separate builder and reviewer roles when practical. The reviewer receives the specification, requirement map, changed-state inventory, tests, and activation evidence, and tries to disprove completion. Review findings are work items, not commentary.

## 10. Evidence-bound status

Use `DONE`, `PARTIAL`, `BLOCKED`, or `FAILED` based on observed scope, not confidence or effort. Nothing material disappears from the final accounting.

## Provenance

This discipline combines patterns drawn from the user's Keeltrace workflow, Matt Pocock's agent-engineering skill structures, and the Patina Project agent-engineering skills: reality-first review, narrow/deep design, contract boundaries, vertical tests, preservation, evidence gathering, and independent falsification.
