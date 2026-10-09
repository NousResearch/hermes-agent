# Otto bounded counterpoint plugin

This plugin adds one admission point to Hermes model turns. It is **not enabled
by default** and it does not create providers, credentials, workers, bots, or
external integrations.

## Rollout

Enable it explicitly in the active Hermes profile. Keep `mode: shadow` while
validating ledger events and route smoke tests. Move to `canary` only with a
non-zero `canary_percent`, and use `blocking` only after the shadow evidence
has been reviewed.

```yaml
plugins:
  enabled:
    - otto-counterpoint
  entries:
    otto-counterpoint:
      settings:
        mode: shadow
        project_id: otto-hermes-gateway
        generator_reasoning_effort: high
        counterpoint_route:
          vendor: anthropic
          family: anthropic
          provider: nous
          model: anthropic/claude-sonnet-5
          reasoning_effort: high
          authenticated: true
          accessible: true
          smoke_tested: true
          relative_load: 1.0
        sample_percent: 0
        canary_percent: 0
        max_corrections: 1
        timeout_seconds: 180
        run_budget_seconds: 120
```

A counterpoint route is eligible only when its family differs from the
runtime generator family and `authenticated`, `accessible`, and
`smoke_tested` are all true. These flags are assertions made by the operator
only after the route has been verified; the plugin never treats a catalog entry
as authorization.

The plugin records request hashes, route metadata, deterministic gate results,
decisions, and lifecycle states in the profile plugin state directory. Prompt,
response, and private model reasoning text are not written to the ledger.

In `shadow`, a successful counterpoint observation leaves tool calls and the
original response unchanged. Admission reviews, blocked decisions, callback
failures, and failed/unresolved workflows remain fail-closed in every mode:
tools are vetoed and the response becomes a human-review message. In `canary`,
non-selected counterpoint turns remain explicit non-enforced observations; they
are never recorded as direct acceptance. In `blocking`, non-direct turns block
tools until the bounded workflow accepts locally; missing independence, failed
gates, callback errors, and unresolved divergence become human review. A human
review message never authorizes an external effect.

This rollout does not restart the gateway or activate a durable worker. Those
operations require a separate operational approval.
