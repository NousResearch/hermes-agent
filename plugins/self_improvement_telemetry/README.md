# Self-Improvement Telemetry Plugin

Opt-in Hermes plugin that writes local, structured tool-call metrics for
self-improvement reviews.

It records:

- hook timestamp;
- session/task id;
- tool name;
- argument key names only;
- result character count;
- risk flags such as `large_tool_output`, `duplicate_skill_view`, and
  `repeated_cronjob_list`.

It does **not** record raw prompts, transcripts, argument values, tool output, or
secrets.

## Enable

Add the plugin to `plugins.enabled` in `~/.hermes/config.yaml`, then restart the
Hermes process or gateway:

```bash
hermes config set plugins.enabled '["self_improvement_telemetry"]'
```

For tests or custom deployments, override the output directory:

```bash
export HERMES_SELF_IMPROVEMENT_TELEMETRY_DIR=/tmp/hermes-self-improvement
```

Default output:

```text
~/.hermes/ops/self-improvement-log/context_metrics.jsonl
```
