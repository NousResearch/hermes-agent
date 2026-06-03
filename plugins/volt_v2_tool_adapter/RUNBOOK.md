# Volt V2 Tool Adapter RUNBOOK

Local runbook for Hermes Volt V2 Tool Adapter v0.

## Scope

This adapter is an opt-in local plugin for observer/transform proofing. It must remain disabled by default.

## Safe enablement for a test profile

Use a temporary or dedicated `HERMES_HOME`. Do not edit `/home/goran/.hermes-doni-clean/config.yaml` directly without backup and rollback.

```bash
export HERMES_HOME=/tmp/volt-v2-hermes-home
mkdir -p "$HERMES_HOME"
cat > "$HERMES_HOME/config.yaml" <<'YAML'
plugins:
  enabled:
    - volt_v2_tool_adapter
volt_v2:
  tool_adapter:
    enabled: true
    mode: observe
    audit_path: /tmp/volt-v2-audit/events.jsonl
    allowlist:
      tools: [read_file]
      paths: [/tmp/volt-v2-vault]
    denylist:
      tools: [terminal, send_message, cronjob, memory]
      path_prefixes: [/tmp/volt-v2-vault/.env]
YAML
```

## Production-profile safety steps

Only after explicit approval:

1. backup current Doni config;
2. add plugin entry with `enabled: false` first;
3. run a disabled smoke and confirm no audit output and no result marker;
4. only then test `enabled: true` in observe mode with a narrow allowlist;
5. never enable route/override in production profile in v0.

Example backup:

```bash
cp /home/goran/.hermes-doni-clean/config.yaml \
  /home/goran/.hermes-doni-clean/config.yaml.before-volt-v2-$(date +%Y%m%d-%H%M%S)
```

Rollback:

```bash
cp /home/goran/.hermes-doni-clean/config.yaml.before-volt-v2-YYYYMMDD-HHMMSS \
  /home/goran/.hermes-doni-clean/config.yaml
```

## Verification commands

```bash
python3 /home/goran/.hermes-doni-clean/runtime/tool-compaction-v0/test_hermes_tool_compact_v0.py
cd /mnt/d/HermesAgent/app
python3 -m pytest tests/test_model_tools.py tests/test_model_tools_async_bridge.py tests/hermes_cli/test_plugins.py tests/plugins/test_disk_cleanup_plugin.py plugins/volt_v2_tool_adapter/tests -q
```

## Expected behavior

- Disabled: no audit write, no result marker.
- Observe: redacted audit JSONL row for allowlisted calls; original result untouched.
- Transform: redacted audit JSONL rows and `_volt_v2_adapter` marker on allowlisted string JSON-object result.
- Denied/sensitive cases: no audit write, no transform.

## Stop conditions

Stop and rollback if:

- audit contains raw path, token, password, secret, credential, or raw result payload;
- any mutation tool is transformed or routed;
- route/override becomes active without a separate approval;
- production config differs from planned diff;
- tests fail.
