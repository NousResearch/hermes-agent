# Volt V2 Tool Adapter Proof

Local, opt-in Hermes plugin proof for the Volt V2 tool adapter lane.

## Status

- Default: disabled (`volt_v2.tool_adapter.enabled: false`).
- Hook mode only: registers `post_tool_call` and `transform_tool_result`.
- No registry override, no new global dispatch path, no network send, no memory write.

## Config shape

```yaml
plugins:
  enabled:
    - volt_v2_tool_adapter

volt_v2:
  tool_adapter:
    enabled: false
    mode: observe        # observe | transform | route | override
    fail_policy: open    # open | closed_for_allowlisted
    artifact_root: /mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/08-OPERATIONS/ACTIVE-WORK
    audit_path: /home/goran/.hermes-doni-clean/logs/volt-v2-tool-adapter.jsonl
    allowlist:
      tools:
        - read_file
        - search_files
        - write_file
      toolsets: []
      paths:
        - /mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/
    denylist:
      tools:
        - terminal
        - send_message
        - cronjob
        - memory
      path_prefixes:
        - /home/goran/.ssh/
        - /home/goran/.hermes-doni-clean/.env
    verification:
      emit_events: true
      write_audit_jsonl: true
      require_result_marker: true
```

## Behavior

- Disabled config: hook callbacks return without side effects.
- Observe mode: allowlisted calls write redacted JSONL audit events and keep original results untouched.
- Transform mode: allowlisted string JSON object results receive `_volt_v2_adapter` marker.
- Denylisted tools, denylisted paths, non-allowlisted tools, and sensitive args are ignored by observe/transform mode.
- Adapter exceptions fail open in observe/transform mode.
- Route/override exception helper fails closed for mutation tools and falls back for read-only tools.

## Audit safety

Audit stores argument shape only:

- path-like values are SHA-256 shaped (`sha256:<prefix>`), not raw paths;
- token/secret/password/credential-like keys are `<redacted>`;
- result payload is represented as character count only.
