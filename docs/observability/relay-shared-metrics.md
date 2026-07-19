# NeMo Relay Shared Metrics

Hermes includes NeMo Relay as a normal runtime dependency on platforms for
which Relay publishes a native wheel. The shared-metrics integration is built
into Hermes and does not require `hermes plugins enable
observability/nemo_relay`. Hermes remains importable without Relay on other
native targets. Those targets use an explicit reduced-capability no-op host:
Hermes execution remains available, while Relay scopes, middleware, plugins,
and subscribers are unavailable. The `hermes-agent[nemo-relay]` extra remains
as a no-op compatibility alias for existing installation commands.

Collection remains off unless Hermes policy enables it:

```yaml
telemetry:
  shared_metrics:
    enabled: true
```

The existing `observability/nemo_relay` plugin remains separate. Enable that
plugin only for its opt-in rich observability exporters, adaptive execution,
or dynamic Relay plugins.

Hermes core owns one Relay host and one isolated Relay session scope per Hermes
session. Core lifecycle producers use
`hermes_cli.observability.relay_runtime` to obtain the shared session handle or
run Relay scope, LLM, tool, and mark APIs in that session context. New product
marks do not require Hermes plugin registration. Shared-metrics marks must
still contain only fields approved by the versioned allowlist; the hard
dependency does not change the collection or privacy policy.

## Current Slices

The current vertical slices record logical model calls and top-level task runs:

```text
Hermes turn, API, and tool hooks
  -> Relay session, task, and LLM lifecycle
  -> Hermes shared-metrics subscriber
  -> SQLite counters
  -> immutable JSON delta package
```

Hermes sends an empty `LLMRequest` into this metrics lifecycle. The terminal
event contains only bounded model family, provider family, locality, call role,
and outcome values. Prompts, responses, exact model IDs, endpoints, errors,
session IDs, task IDs, and request IDs are not included in the metrics event or
package.

Each task run is a Relay `Function` scope named `hermes.task_run`, parented to
the owning Hermes session. The start counter contains only bounded execution
surface and entrypoint values. The terminal counter contains bounded outcome,
end reason, termination status, duration, logical model-call count, terminal
tool-call count, and provider-retry count buckets. Retries are additional
provider attempts for the same Hermes API request ID; they do not inflate the
logical model-call count. Tool calls are deduplicated by their Hermes tool-call
ID after a terminal tool result is observed. The outer `AIAgent` execution
boundary closes the task for normal returns, early returns, exceptions, and
cancellations. Active task ownership follows the task ID if Hermes rotates its
conversation session during context compression.

Local state is written under:

```text
$HERMES_HOME/telemetry/shared_metrics/metrics.sqlite3
$HERMES_HOME/telemetry/shared_metrics/outbox/*.json
```

The database keeps transactional aggregate and package-outbox state. Package
files are immutable delta documents that conform to a closed JSON schema and
are written with atomic replacement.

## Smoke Test

Run a real Hermes CLI turn against the deterministic local model server:

```bash
./.venv/bin/python scripts/smoke_nemo_relay_shared_metrics.py
```

The script uses the installed `nemo-relay` dependency by default. Pass
`--relay-python ../nemo-relay/python` only when testing a locally built Relay
binding.

The smoke verifies the model request reached the local server, model and task
counters were stored, one package was exported, and prompt, response, and
exact-model canaries are absent from the package.
