# Scalar Supremacy via Private Client Viewports

## Thesis

Hermes-surface scalar supremacy means: **the most scalable formal implementation supersedes**.
It is not a preference; it is a runtime obligation enforced at the conductor boundary.

In practice, supremacy is achieved through **private client viewports globally**:
- **Bounded:** each viewport is a sovereign local-first surface, not a shared monolith
- **Scalable:** `pc://mesh/global/<id>` + `pc://mesh/user/<user_id>/<id>` address any node
- **Supersedes:** thinner runtimes (Tauri, WebView) replace heavier ones by policy, not by port

## Enforcement Points

1. `hermes_cli/conductor.py` — `_enforce_promote_scalar_supremacy()` blocks bare
   scalar authority tails before dispatchers mutate state.
2. `templates/æ.html` — sovereign shell bootstraps Monaco, terminal, and QR as
   local-first surfaces under the operator grammar.
3. `apps/reachy/hermes-monaco.js` — bridges viewport edits through `/conductor`,
   making every in-viewport change a bounded mesh action.
4. `skills/æ/SKILL.md` — installs the sovereign mesh primitive on any Hermes host.

## Metastability Rule

Merge writes are gated by the current executor resource bound.
Bare `status`, `run`, `default`, `latest`, `self` tails are rejected unless they
are inside a deterministic policy path (e.g. `sim/...` or `/sim/...`).

Result: infinite surface expansion is impossible without explicit resource binding.

## Victory Condition

More bounded private client viewports globally than monolithic controllers.
