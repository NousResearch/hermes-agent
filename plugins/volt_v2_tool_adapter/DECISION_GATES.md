# Volt V2 Tool Adapter Decision Gates

Status: local done-done proof gate, no remote push.

## Transform value gate

Decision: transform mode is valuable only for explicit local verification and downstream canonicalization experiments.

Allowed in this proof:

- `mode: transform` for allowlisted read-only/file-inspection calls in an isolated profile.
- Result mutation limited to adding `_volt_v2_adapter` metadata to string JSON-object results.
- Non-string results, denied tools, denied paths, sensitive args, and non-allowlisted tools return `None` from the hook and leave the original tool result untouched.

Not allowed without a new approval gate:

- transforming mutation-tool results;
- transforming results that contain credentials or private payloads;
- enabling transform globally in the production Doni profile;
- using transform as a routing or policy-enforcement substitute.

Acceptance evidence:

- runtime transform smoke passed in a temporary `HERMES_HOME`;
- plugin manager integration test validates transformed marker and audit rows;
- `test_transform_value_gate_does_not_change_non_string_or_denied_results` validates no-op behavior for denied/non-string cases.

## Route / override decision gate

Decision: route and override are intentionally not shipped as active runtime behavior in v0.

Reasons:

- `tools.registry` already has explicit `override=True`, and shadowing must remain auditable.
- Route/override can affect mutation tools and therefore needs stronger approval, rollback, and regression coverage.
- Current proof objective is observer + transform, not dispatch replacement.

Current v0 behavior:

- no tool override is registered by `plugins/volt_v2_tool_adapter/__init__.py`;
- `handle_adapter_exception()` documents fail-closed behavior for mutation tools in route/override modes;
- route/override stay decision-gated and should remain disabled for Doni rollout.

Promotion criteria for future route/override work:

1. explicit Goran approval;
2. separate branch/commit;
3. mutation-tool fixtures for `write_file`, `patch`, and `terminal` with rollback paths;
4. proof that `override=True` is never used accidentally;
5. production config backup + rollback drill;
6. no credential, memory, network, or public side effects during tests.
