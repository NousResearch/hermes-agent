# NeMo Relay Shared Metrics

Hermes includes NeMo Relay as a normal runtime dependency on platforms for
which Relay publishes a native wheel. The shared-metrics integration is built
into Hermes and does not require `hermes plugins enable
observability/nemo_relay`. Hermes remains importable without Relay on other
native targets. Those targets use an explicit reduced-capability no-op host:
Hermes execution remains available, while Relay scopes, middleware, plugins,
and subscribers are unavailable. The `hermes-agent[nemo-relay]` extra remains
as a no-op compatibility alias for existing installation commands.

Hermes requires NeMo Relay 0.6 RC 3 or later. That release establishes the
lossless provider-codec contract used for Anthropic Messages, OpenAI Chat
Completions, and OpenAI Responses requests.

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

The current vertical slices record anonymous install activity, setup and
first-use milestones, logical model calls, top-level task runs, tool and
approval outcomes, and skill lifecycle and reuse:

```text
Hermes turn, API, tool, and approval hooks
  -> Relay session, task, LLM, tool, and mark lifecycle
  -> Hermes shared-metrics subscriber
  -> SQLite counters
  -> immutable JSON delta package
```

Hermes sends an empty `LLMRequest` into this metrics lifecycle. The terminal
event contains only bounded model family, provider family, locality, call role,
outcome, total latency, input/output token, physical retry, and estimated-cost
values. Exact latency, token counts, and cost are bucketed before Relay event
emission. Prompts, responses, exact model IDs, endpoints, errors, session IDs,
task IDs, and request IDs are not included in the metrics event or package.

The first consented session start emits an empty `hermes.client.active` Relay
mark. The profile-scoped subscriber creates a random UUID install identity and
uses a transactional compare-and-set to record at most one active-install
counter in any rolling 24-hour window. The metric has no dimensions; Hermes
version, OS family, architecture, and install method remain bounded package
resources. Concurrent Hermes processes share the SQLite latch, so simultaneous
starts cannot double-count one install. A later session or task can attempt the
mark again, but the subscriber suppresses it until the rolling window expires.

An opted-in `hermes setup` run, including the equivalent `hermes portal`
onboarding entry point, emits `hermes.setup.started` and
`hermes.setup.finished` marks through a short-lived Relay scope. The marks
contain only a bounded setup mode, outcome, and failure stage; provider names,
credentials, answers, and error text are never included. An unflagged setup
invocation is classified as `interactive` because a new user chooses quick,
full, or blank-slate setup after the lifecycle begins. Setup that begins before
shared-metrics consent is available is not recorded retroactively. This
preserves the rule that no telemetry identity or local state exists before the
profile has explicitly enabled collection.

The first consented session or task that reaches Hermes's normal runtime
boundary records `hermes.client.first_usable` once. The first accepted task
terminal with the bounded `success` outcome records
`hermes.client.first_successful_task` once. Both use transactional SQLite
latches, survive process restarts, and remain single-counted when concurrent
Hermes processes reach the milestone together. The successful-task latch is
committed in the same transaction as its terminal task counter.

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

Each tool invocation is represented by a Relay tool lifecycle named
`hermes.tool_call`. The terminal counter contains only bounded tool category,
outcome, approval outcome, latency, and explicit retry-count buckets. Hermes
does not infer retries from repeated tool names or adjacent calls; when the
hook does not provide an explicit retry relationship, the retry bucket is
`unknown`. Approval decisions are emitted as `hermes.tool_approval` marks and
recorded as attributed to a tool call or explicitly `unattributed`. Tool names,
call IDs, arguments, results, commands, descriptions, and error text are not
included in shared-metrics events or packages. A started tool that is still
open when its task terminates is closed as failed or cancelled and remains in
the task's tool-count bucket.

Successful skill mutations emit `hermes.skill.lifecycle` marks with only a
bounded action and provenance. Successful loads emit `hermes.skill.load`
marks with bounded provenance, first-use or reuse state, reuse-after-patch
state, and a use-count bucket. Hermes derives reuse and patch-generation
continuity transactionally in its existing `skills/.usage.json` state; skill
names and exact counts or generations never enter Relay metrics events,
SQLite dimensions, or packages. A use after a new patch is counted once as
`reused_after_patch`; later uses remain ordinary reuse until another patch.
Task-outcome attribution after a patch remains deferred until its window and
multi-skill semantics are defined.

Local state is written under:

```text
$HERMES_HOME/telemetry/shared_metrics/metrics.sqlite3
$HERMES_HOME/telemetry/shared_metrics/outbox/*.json
```

The database keeps transactional aggregate and package-outbox state. Package
files are immutable delta documents that conform to a closed JSON schema and
are written with atomic replacement. Each package records the Hermes version,
OS family, architecture, and install method as bounded client resources.
Unrecognized platform or installation values are exported as `unknown`; raw
platform strings, hostnames, and paths are never included.

The install identity is scoped to one `HERMES_HOME`. To reset it, stop Hermes
processes and remove `$HERMES_HOME/telemetry/shared_metrics`. This deliberately
removes the old identity, aggregate database, and queued local packages
together; the next consented session creates a new identity. Disabling shared
metrics stops new collection but does not silently delete previously collected
local state.

## Smoke Test

Run a real Hermes CLI turn against the deterministic local model server:

```bash
./.venv/bin/python scripts/smoke_nemo_relay_shared_metrics.py
```

The script uses the installed `nemo-relay` dependency by default. Pass
`--relay-python ../nemo-relay/python` only when testing a locally built Relay
binding.

The smoke first emits an opted-in setup lifecycle, then has the local model
request a real `read_file` tool call before its final response. It also drives
create, load, reuse, patch, edit, stale, archive, restore, and install skill
transitions through the installed Relay binding. It verifies bounded setup,
first-use, model, task, tool, and skill counters in SQLite, validates all
exported delta packages against the closed schema, verifies the anonymous
active-install counter, and checks that prompt, response, exact-model,
tool-call ID, tool-result, and skill-name canaries are absent from the packages.
