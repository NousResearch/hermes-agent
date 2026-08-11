# NeMo Relay Shared Metrics

Hermes includes NeMo Relay as a normal runtime dependency on platforms for
which Relay publishes a native wheel. The shared-metrics integration is built
into Hermes and does not require a Hermes observability plugin. Hermes remains
importable without Relay on other native targets. Those targets use an
explicit reduced-capability no-op host:
Hermes execution remains available, while Relay scopes, middleware, plugins,
and subscribers are unavailable. The `hermes-agent[nemo-relay]` extra remains
as a no-op compatibility alias for existing installation commands.

Hermes requires NeMo Relay 0.7.1 or later within the 0.7 release line. That
release establishes the lossless provider-codec contract used for Anthropic
Messages, OpenAI Chat Completions, and OpenAI Responses requests.

## Runtime Dependency and Data Boundary

Hermes installs the platform-specific `nemo-relay` native wheel from the
bounded `>=0.7.1,<0.8` dependency range. The published package is built from
the [NVIDIA NeMo Relay repository](https://github.com/NVIDIA/NeMo-Relay).
Unsupported platforms use the explicit no-op runtime described above rather
than downloading a different implementation.

When Relay managed execution is active, the provider request and response pass
through that native module in the Hermes process so configured interceptors can
operate on the real call. This is separate from the shared-metrics data
contract. Shared-metrics mode installs no network exporter and its subscriber
accepts only the versioned, allowlisted projection described below. Enabling a
separately configured rich-observability or dynamic plugin can create a
different data path and requires its own policy review.

Collection remains off unless Hermes policy enables it:

```yaml
telemetry:
  shared_metrics:
    enabled: true
```

This choice is read from the profile's own `config.yaml`. A machine-managed
configuration overlay cannot enable or disable shared metrics on the profile's
behalf.

Relay plugin activation is owned by the native runtime and remains explicitly
opt-in. Set `HERMES_NEMO_RELAY_PLUGINS_TOML` to a selected `plugins.toml` to
activate configured middleware, exporters, or dynamic plugins. When it is
unset, Hermes does not invoke Relay's plugin initializer or trigger Relay
plugin configuration discovery. Invalid explicit configuration is reported
and Hermes continues without native plugin activation.

Hermes core owns one Relay host and one isolated Relay session scope per Hermes
session. Core lifecycle producers use
`hermes_cli.observability.relay_runtime` to obtain the shared session handle or
run Relay scope, LLM, tool, and mark APIs in that session context. New product
marks do not require Hermes plugin registration. Shared-metrics marks must
still contain only fields approved by the versioned allowlist; the hard
dependency does not change the collection or privacy policy.

## Current Slices

The current vertical slices record pseudonymous profile activity, logical
model calls, top-level task runs, tool and approval outcomes, and skill
lifecycle and reuse:

```text
Hermes turn, API, tool, and approval hooks
  -> Relay session, task, LLM, tool, and mark lifecycle
  -> Hermes shared-metrics subscriber
  -> SQLite counters
  -> immutable JSON delta package
```

Hermes sends an empty `LLMRequest` into the metrics-owned lifecycle. This does
not describe the separate managed-execution call through the native runtime
documented above. The terminal metrics event contains the model identifier and
provider route that Hermes used for the logical call, such as
`nvidia/nemotron-3-ultra` through `openrouter`. These identifiers are
lowercased and structurally bounded, but they are not normalized through a
checked-in model catalog. Pricing and model-family classification belong to
the metrics backend. Prompts, responses, endpoints, errors, session IDs, task
IDs, and request IDs are not included in the metrics event or package.
New calls use `hermes.model_route.count`. The previous
`hermes.model_call.count` contract remains readable only so pending local
counters created by older builds can be exported without losing data.

The first consented session start emits an empty `hermes.client.active` Relay
mark. The profile-scoped subscriber creates a random UUID install identity and
uses a transactional compare-and-set to record at most one client-active
counter in any rolling 24-hour window. The metric has no dimensions; Hermes
version, OS family, architecture, and install method remain bounded package
resources. Concurrent Hermes processes share the SQLite latch, so simultaneous
starts cannot double-count one install. A later session or task can attempt the
mark again, but the subscriber suppresses it until the rolling window expires.

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
derives the category from the toolset already declared in its runtime registry;
custom and unrecognized toolsets collapse to `other` rather than exporting
tool or plugin names. Hermes does not infer retries from repeated tool names or
adjacent calls; when the
hook does not provide an explicit retry relationship, the retry bucket is
`unknown`. Approval decisions are emitted as `hermes.tool_approval` marks and
recorded as attributed to a tool call or explicitly `unattributed`. Tool names,
call IDs, arguments, results, commands, descriptions, and error text are not
included in shared-metrics events or packages. A started tool that is still
open when its task terminates is closed as failed, timed out, or cancelled and
remains in the task's tool-count bucket.

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
platform strings, hostnames, and paths are never included. Fully packaged
aggregate rows and successfully exported package rows and files are retained
locally for 30 days. Pending package rows and counters with unexported deltas
are never pruned.
Package schema v1 remains unchanged for existing outbox files. New packages
use v2, which accepts both the retired model-call contract and the current
model-route contract so upgrades can drain pending counters safely.

Each package contains an `install_id` generated as a random UUID. Despite the
schema field name, its current scope is one `HERMES_HOME`, so it is more
precisely a persistent pseudonymous profile identifier. It is not derived from
hardware, account, host, path, or credential data. It remains stable across
packages from that profile and can therefore link those local packages.
Deleting `$HERMES_HOME/telemetry/shared_metrics` resets the identifier together
with all aggregates and package files.

This slice has no remote-delivery path. A future remote exporter must not reuse
the persistent local identifier by default. It requires a separate product and
privacy decision covering consent, identity scope, rotation or keyed
pseudonymization, reset behavior, retention, and deletion.

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

The smoke has the local model request a real `read_file` tool call before its
final response, then drives create, load, reuse, patch, edit, stale, archive,
restore, and install skill transitions through the installed Relay binding. It
verifies model, provider, task, tool, and skill counters in SQLite, validates
all exported delta packages against the closed schema, verifies the
pseudonymous client-active counter, and checks that prompt, response, tool-call
ID, tool-result, and skill-name canaries are absent from the packages.
