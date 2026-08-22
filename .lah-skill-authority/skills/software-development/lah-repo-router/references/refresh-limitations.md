# Inventory and drift limits — schema v4

`repo_mappings.json` is the canonical ontology, not an automatically generated filesystem dump.

`detect-routing-drift.cjs` may discover new repositories and worktrees, but it cannot infer ownership roles, write scope, or semantic signals. Review discovered evidence before updating the canonical mapping.

Use `validate-routing-drift.cjs` to verify schema v4, required role authorities, and Codex/Hermes documentation handoff.

Never overwrite the canonical mapping from discovery output. Never use historical prefixes or receipts as current ownership proof.
