# Volt V2 Tool Adapter PR Gate

Local artefact only. Remote push/PR is blocked until Goran explicitly approves it.

## Branch / commit state

- Target branch: `main` locally.
- Existing proof commit before done-done hardening: `5a4df2345 feat: add opt-in Volt V2 tool adapter proof`.
- This gate artefact is for the follow-up done-done hardening commit.

## Proposed PR title

`feat: harden Volt V2 tool adapter proof`

## Proposed PR body

### Summary

- hardens the opt-in Volt V2 tool adapter config parser and malformed-config behavior;
- formalizes audit JSONL schema v1 with validation and redaction tests;
- adds plugin manager integration coverage for bundled plugin discovery, hook registration, runtime transform, and audit output;
- adds runtime smoke evidence for observe/transform and controlled disabled rollout using temporary `HERMES_HOME` paths only;
- documents security/privacy boundaries, transform value gate, and route/override decision gate.

### Safety

- disabled by default;
- no production Doni config modified;
- no credentials read or written;
- no memory-layer writes;
- no network/public side effects;
- no `tools.registry` override in v0;
- route/override remain decision-gated and inactive.

### Verification

Run locally before opening PR:

```bash
python3 /home/goran/.hermes-doni-clean/runtime/tool-compaction-v0/test_hermes_tool_compact_v0.py
python3 -m pytest tests/test_model_tools.py tests/test_model_tools_async_bridge.py tests/hermes_cli/test_plugins.py tests/plugins/test_disk_cleanup_plugin.py plugins/volt_v2_tool_adapter/tests -q
```

### Remote gate

Do not run any of these without explicit approval:

```bash
git push origin main
# or
# gh pr create --fill
```
