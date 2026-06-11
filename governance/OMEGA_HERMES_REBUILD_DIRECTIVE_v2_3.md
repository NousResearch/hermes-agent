# OMEGA / HERMES REBUILD DIRECTIVE v2.3 — Code Artifact

Status: implemented as a local-first governance substrate in this repository.

This artifact captures the execution boundary for the directive as implemented in
code.  It is not a claim that every live Omega/Hermes runtime service is ready or
that strict enforcement has been flipped on in production.

## Implemented gates

- Source-of-truth registry: `governance.source_registry`.
- Policy Gate classifier/evaluator: `governance.policy`.
- Hash-chained decision/evidence logs: `governance.evidence`.
- Verified backup helper and manifests: `governance.backup`.
- Memory candidate/admission schema: `governance.memory_governor`.
- Export mode and secret scan scaffolding: `governance.export_safety`.
- Dispatch seam integration: `model_tools.handle_function_call` classifies and
  logs every non-bridge tool call; hard denials block execution.
- Memory write integration: `tools.memory_tool.MemoryStore` records successful
  and rejected durable-memory admission decisions.

## Enforcement boundary

The Policy Gate always logs decisions and always blocks hard denials such as
known destructive commands.  Other gated states (`require_approval` and
`allow_after_backup`) are audit-mode by default and become hard blocks when
`HERMES_POLICY_GATE_MODE=strict` or `HERMES_POLICY_GATE_MODE=enforce` is set.

This staged boundary prevents accidental live-runtime breakage while making the
strict gate one environment flip away after operational rollout testing.

## Stop conditions

- Missing capability classification defaults to fail-closed at the policy API.
- Known destructive shell semantics are denied.
- Confirmed-active source registry entries require evidence references.
- Reversible edits are marked `allow_after_backup` unless a current verified
  backup exists.
- Shareable export mode requires secret redaction.

## Rollback

For the implementation session that created this artifact, verified backups for
modified existing files are recorded under `/home/eric/Downloads/hermes-governance-backups/`.
Restoring those files is a live mutation and requires explicit approval.
