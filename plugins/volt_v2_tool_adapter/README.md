# Volt V2 Tool Adapter Proof

Local, opt-in Hermes plugin proof for the Volt V2 tool adapter lane.

## Status

- Default: disabled (`volt_v2.tool_adapter.enabled: false`).
- Hook mode only: registers `post_tool_call` and `transform_tool_result`.
- No registry override, no new global dispatch path, no network send, no memory write.
- Runtime observe and transform smokes verified with temporary `HERMES_HOME` directories; production Doni config was not modified.

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

Malformed config is coerced to safe defaults:

- unknown `mode` -> `observe`;
- unknown `fail_policy` -> `open`;
- scalar list fields become single-item tuples only when safe;
- non-list/non-scalar allowlist and path fields become empty tuples;
- verification booleans accept normal YAML booleans and safe strings like `true`, `false`, `yes`, `no`, `on`, `off`.

## Behavior

- Disabled config: hook callbacks return without side effects.
- Observe mode: allowlisted calls write redacted JSONL audit events and keep original results untouched.
- Transform mode: allowlisted string JSON object results receive `_volt_v2_adapter` marker.
- Denylisted tools, denylisted paths, non-allowlisted tools, and sensitive args are ignored by observe/transform mode.
- Adapter exceptions fail open in observe/transform mode.
- Route/override exception helper fails closed for mutation tools and falls back for read-only tools.

## Audit safety

Audit format is schema v1. Each JSONL row is validated before write/read and contains:

- `schema_version`, `ts`, `session_id`, `task_id`, `tool_call_id`;
- `tool_name`, `mode`, `decision`, `allowlisted`, `duration_ms`, `reason`;
- `args_shape` with redacted/hash-shaped arguments only;
- `result_chars`, never raw result payload.

Audit stores argument shape only:

- path-like values are SHA-256 shaped (`sha256:<prefix>`), not raw paths;
- token/secret/password/credential-like keys are `<redacted>`;
- result payload is represented as character count only.

## Security / privacy review

Verified boundaries:

- no credential read/write requirement;
- no production config mutation during tests or runtime smokes;
- no memory-layer writes;
- no network delivery or public side effects;
- no `tools.registry` override in v0;
- allowlist/denylist blocks sensitive args, denylisted paths, denied tools, and non-allowlisted tools.

Route/override are decision-gated only. They are not enabled as runtime modes for Doni rollout until a separate explicit approval and stronger acceptance test set exist.
