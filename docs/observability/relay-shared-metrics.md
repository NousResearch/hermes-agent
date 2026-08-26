# NeMo Relay Shared Metrics

Hermes includes NeMo Relay as a normal runtime dependency on platforms for
which Relay publishes a native wheel. The shared-metrics integration is built
into Hermes and does not require a Hermes observability plugin. Hermes remains
importable without Relay on other native targets. Those targets use an
explicit reduced-capability no-op host:
Hermes execution remains available, while Relay scopes, middleware, plugins,
and subscribers are unavailable. The `hermes-agent[nemo-relay]` extra remains
as a no-op compatibility alias for existing installation commands.

> [!WARNING]
> This removes the Hermes `observability/nemo_relay` plugin. Existing users
> must remove `observability/nemo_relay` (or its legacy `nemo_relay` alias)
> from `plugins.enabled` and move exporter configuration into a Relay
> `plugins.toml` selected with `HERMES_NEMO_RELAY_PLUGINS_TOML`. The legacy
> `HERMES_NEMO_RELAY_ATOF_*` and `HERMES_NEMO_RELAY_ATIF_*` variables no
> longer activate exporters. Without the new variable, Hermes does not run
> Relay plugin discovery, configuration layering, middleware, or exporters.

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
contract. Shared-metrics mode installs no rich-observability network exporter,
and its subscriber
accepts only the versioned, allowlisted projection described below. The
opt-in package sender described in Appendix A is the only outbound path, it
transmits nothing unless the user enables both `enabled` and `send`, and it
sends whole packages rather than live spans. Enabling a
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
activate configured middleware, exporters, or dynamic plugins. When the
variable is unset, Hermes does not invoke Relay's plugin initializer, so Relay
does not perform plugin configuration discovery or layering. When it is set
and the selected file loads successfully, Relay performs its normal static
`plugins.toml` discovery and layers the selected static configuration over the
discovered configuration. Dynamic `[[plugins.dynamic]]` records are loaded
from the selected file only. If the selected file cannot be loaded, Hermes
reports the error and does not invoke Relay initialization or fall back to
ambient discovery.

## Session-Span Segmentation for Continuous Sessions

Relay exports a span when its scope closes. A continuous gateway session can
remain open for days, so its session span remains open even though each turn
span is exported normally. Optional segmentation rotates only the session
scope at a turn boundary:

```yaml
gateway:
  telemetry:
    session_segments:
      on_compaction: false  # rotate after context compaction
      max_turns: 0          # 0 = unlimited; N = turns per segment
```

| Key | Default | Behavior |
|---|---:|---|
| `on_compaction` | `false` | Rotate after compaction completes, at the next turn boundary. |
| `max_turns` | `0` | Rotate after every N completed turns; `0` disables the cap. |

Both defaults preserve one session scope for the full session. Rotated spans
retain the same `session_id` and add `hermes.session.segment` plus
`hermes.session.segment_reason` (`compaction` or `max_turns`).

## Process-Wide Plugin Policy and Profile Isolation

Relay plugin configuration is a process-level deployment choice, not a Hermes
profile setting. The first hosted profile triggers lazy initialization, and
every additional profile hosted by that Hermes process shares the resulting
static middleware, dynamic plugins, subscribers, exporters, and guardrail
policy. After initialization succeeds, Hermes logs:

```text
Relay plugins are active process-wide and apply to all profiles hosted by this Hermes process.
```

Profile scopes still preserve causal isolation inside that shared policy.
ATIF groups events by their top-level Agent scope, so simultaneous profile
sessions produce separate trajectories rather than one mixed trajectory.
ATOF and other global subscribers observe events from every hosted profile.
Static and dynamic middleware likewise runs for managed calls from every
profile.

A worker plugin running in a separate worker process does not create a
per-profile security boundary. One process-wide activation dispatches calls
from all hosted profiles to that worker while preserving the invoking
profile's Relay scope stack. Native dynamic plugins are loaded into the Hermes
process and share the same policy boundary.

Run profiles in separate Hermes processes when they require different trust
levels, plugin credentials, exporter destinations, or guardrail policies.
This process-wide plugin contract does not change each profile's independent
shared-metrics consent, local SQLite state, or ATIF trajectory grouping.

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

Remote delivery is opt-in and off by default. A remote exporter must not reuse
the persistent local identifier by default. It requires a separate product and
privacy decision covering consent, identity scope, rotation or keyed
pseudonymization, reset behavior, retention, and deletion.

> Those decisions are recorded in
> [Appendix A](#appendix-a-remote-exporter-decisions-phase-2), and the exporter
> implementing them has shipped. Collection alone still transmits nothing: the
> sender runs only when `telemetry.shared_metrics.send` is also true, and it
> transmits a rotating HMAC of the install identity rather than the identifier
> itself.

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

## Appendix A: Remote Exporter Decisions (Phase 2)

Status: **implemented.** This appendix answers the product and
privacy questions that "Current Slices" defers to a future remote exporter. It
records what was decided and why, so the reasoning survives the implementation.

Sending is off by default and requires both `telemetry.shared_metrics.enabled`
and `telemetry.shared_metrics.send`.

The exporter sends the package files already written under
`$HERMES_HOME/telemetry/shared_metrics/outbox/` to the Hermes telemetry ingest
service. That service validates only the envelope (`schema_version` plus a UUID
`package_id`) and stores the body verbatim in S3.

### A.1 Consent

Transmission is a **separate opt-in** from collection, under a new config key:

```yaml
telemetry:
  shared_metrics:
    enabled: false   # collect locally
    send: false      # NEW: transmit to the Nous telemetry service
```

- `send` defaults to **false**. Collection alone never transmits.
- `send` requires `enabled`. It does **not** imply it: a transmission flag must
  not silently switch on collection. `send: true` with `enabled: false` warns
  and does nothing.
- Like `enabled`, `send` is profile-owned and is not overridden by
  managed-scope configuration.

**Only packages for periods on or after the opt-in day are ever sent.** The
opt-in day (UTC) is recorded when `send` first becomes true, and any package
whose `period_start` predates it is permanently excluded, however late it was
created.

The gate is on the **period**, not on the package's creation time. One period
is split across several packages created on different days: a day's first
package is written that day, and a tail package for the same period typically
follows the next day. Gating on creation time would send a period's tail while
dropping its head, reporting a **silently undercounted** day. Gating on the
period keeps consent forward-only and every transmitted period complete.

Local history can be up to 30 days old, and that data was collected under a
promise that nothing is uploaded. Honouring consent forward-only costs at most
30 days of backlog we never had permission to send.

### A.2 Identity scope — the transmitted identifier is derived, not the local one

`install_id` is the persistent profile-scoped identifier described above. It is
**not transmitted**. Each package sent carries a derived value instead:

```text
transmitted_id = HMAC-SHA256(key = rotation_salt, message = install_id)
```

- `rotation_salt` is random, generated locally, and never leaves the machine.
- The derivation is one-way: the service cannot recover `install_id`.
- Within a rotation window, packages from one profile correlate — so distinct
  installs remain countable, which is the primary analytical question.
- Across windows, they do not.

This satisfies "must not reuse the persistent local identifier by default"
while keeping the data useful. Stripping the identifier entirely was rejected
because "how many installs are reporting" is the first question the data must
answer; sending `install_id` unchanged was rejected because it contradicts the
commitment made above.

**Byte-identical resends still hold.** The derived value is computed **once**,
when the package is first prepared for sending, and stored alongside the
package (the derived id only — not a second copy of the payload, which is
recomputed deterministically from the stored package). A retry therefore
rebuilds identical bytes even if the salt rotated in between. The contract
requires this: resending a `package_id` with different content is undefined
behaviour.

### A.3 Rotation

`rotation_salt` rotates on a fixed schedule (default: every 30 days, aligned to
local history retention). Rotation only affects packages prepared after it;
already-prepared packages keep their derived value so retries stay
byte-identical.

Rotation bounds long-term linkability without destroying short-term cohort
analysis. A profile is one identity for the length of a window, and an
unrelated identity after it.

**What rotation does not bound.** The identifier changes; the rest of the
envelope does not. `resource` (`os_family`, `architecture`, `install_method`,
`hermes_version`) is stable and low-entropy, and `period_start` /
`period_end` are contiguous across a rotation boundary. For a common
configuration this is no help to an observer — measured against the 11 real
packages in a development outbox, every one shares the same
`arm64 / macos / git` tuple. For a **rare** configuration it is a plausible
re-identification aid: an unusual architecture or install method, combined
with an uninterrupted daily period sequence, can bridge two windows. The
claim this design makes is therefore "rotation raises the cost of long-term
correlation", not "rotation makes it impossible". Narrowing that residue
would mean coarsening `resource` or jittering period boundaries, and neither
is worth the analytical loss today — but it should be a conscious decision,
not an unexamined one.

### A.4 Reset behavior

Removing `$HERMES_HOME/telemetry/shared_metrics` still resets local identity,
aggregates, and package files, exactly as documented above. Two honest
qualifications now apply:

- Reset also discards `rotation_salt`, so subsequent packages derive a **new**
  transmitted identity. Local reset does give a new remote identity.
- Reset **cannot unsend**. Packages already transmitted remain in the ingest
  service's storage under their derived identifier. There is no read-back or
  delete API in the v1 contract.

Setting `send: false` stops transmission immediately: consent is re-read
before every package, so a pass already in flight stops after the package it
is currently sending rather than draining its whole batch. It does not delete
previously transmitted packages, and it does not stop local collection.

Turning sending off also **closes the consent window**. Packages collected
while it was off are never transmitted, even if sending is later re-enabled —
re-enabling starts a new window from that day. Without this, a write-once
opt-in date would have retroactively released the entire refused period the
next time the user changed their mind.

### A.5 Retention

- **Local:** unchanged — 30 days for successfully exported history, and pending
  deltas are kept until exported. Send state does **not** extend local
  retention: a package that could never be sent is still pruned at 30 days.
  Unbounded local growth against a permanently unreachable endpoint is a worse
  failure than losing metrics from an install that has been broken for a month.
- **Remote:** raw packages are retained in S3 without expiry in production and
  for 30 days in staging.

### A.6 Deletion

There is no remote deletion path in the v1 contract, and this appendix does not
invent one. What a user can do:

| Action | Effect |
|---|---|
| `send: false` | No further packages leave the machine |
| `enabled: false` | Collection stops; existing local state remains |
| Remove `.../shared_metrics` | Local identity, aggregates, and files reset; future sends use a new derived identity |
| Delete already-sent data | Not self-service — requires an operator acting on the S3 bucket |

If a deletion-on-request obligation is ever taken on, it needs a lookup path
from a user to their derived identifiers. That is deliberately **not** built:
it would require retaining the mapping this design exists to avoid. Any such
change is a new product decision, not an implementation detail.

### A.7 What the outbox directory is

Recorded because it was misread once during Phase 2 planning, in a way that
would have deleted user data.

The directory is **local history, not a send-queue**. `package_outbox` is the
SQLite table; its `exported_at` column means "written to disk", not "sent".
Files are immutable and pruned **by age alone**.

The ingest contract says senders should delete a package from their outbox on
`202`. **The exporter does not do this.** Deleting on acknowledgement would
repurpose the user's 30-day local history as a transmission queue and destroy
state they were promised. Send state lives in new columns on the
`package_outbox` table instead; the files are untouched by transmission.

### A.8 Scope note

The `install_id` field inside the package body is what gets replaced by the
derived value. No other payload field changes, nothing is added, and the
service treats the whole body as opaque. Payload schema evolution therefore
stays a sender-side concern, as before.
