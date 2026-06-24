# Telemetry & Observability

Hermes ships with a built-in, local-first telemetry system. It records what your
agent does — workflows, model calls, tool calls, errors — to your own machine, powers
`/insights`, and (when *you* enable it) exports everything to your own observability
stack. It is private by default and never sends your data to Nous unless you explicitly
opt in to anonymous aggregate metrics.

This page explains the whole feature: the three planes, what's captured, the
`hermes telemetry` commands, and how an enterprise streams or exports all of its data
to its own infrastructure.

> Looking for the **plugin hook contract** (how to write your own observer plugin)?
> That's in [`README.md`](./README.md). This page is about the built-in telemetry
> plane and its CLI.

## The three planes

Telemetry is organized into three planes with a hard wall between them:

| Plane | What it holds | Default | Destination |
| --- | --- | --- | --- |
| **local** | Full-fidelity observability — runs, model/tool calls (real model & provider names), durations, errors | **on** | your machine only |
| **aggregate** | Opt-in metadata to Nous (no uploader ships yet) | **off** (opt-in) | Nous, only if you enable it |
| **trajectories** | Full message content / reasoning / raw tool args | **off** (opt-in) | your own export destinations only |

The local plane is the one you'll use day to day — it records the real values that
happened (actual model ids, providers, tool names). The aggregate plane is the only
thing that could ever leave for Nous; it is opt-in, default-off, and has no uploader
today. The trajectories plane unlocks full-content export to *your own* destinations —
it is never wired to Nous.

## Local plane — always-on observability

The local plane is implemented as a bundled `telemetry` plugin that listens to Hermes
lifecycle hooks (model calls, tool calls, session start/finalize) and writes events to:

- an append-only JSONL log at `~/.hermes/telemetry/events.jsonl` (the source of truth)
- indexed `tel_*` tables in `state.db` (for fast queries and rollups)

Writes are fire-and-forget on a background thread: telemetry can never block, slow, or
fail a model call or tool call. If the local plane is disabled (`telemetry.local: false`)
the plugin does not load at all.

### Seeing your local data

```bash
hermes insights            # usage report — now includes an "Observability" section
hermes telemetry status    # planes, consent, export posture, local data volume
```

The `insights` Observability section shows workflow counts and success rate, duration
p50/p95, tool failure rates by category, provider/model-class mix, and cache hit rate —
all computed locally with exact values.

## `hermes telemetry` commands

```text
hermes telemetry status      Show planes, consent state, export posture, local volume
hermes telemetry preview     Show the aggregate events that would be produced (local)
hermes telemetry export      Export local telemetry to a file/stream or OTLP endpoint
```

Consent and the install id are plain config, not separate verbs — set them with
`hermes config set` (or a managed-scope pin):

```bash
hermes config set telemetry.consent_state aggregate   # opt in to the aggregate plane
hermes config set telemetry.consent_state local       # opt out (local plane stays on)
hermes config set telemetry.install_id ""             # reset the install id (mints a new one)
```

### Aggregate plane (opt-in)

The aggregate plane is **off by default** and has **no uploader today** — nothing is
sent to Nous. Consent lives in `telemetry.consent_state` (`unknown` / `local` /
`aggregate`); setting it to `aggregate` records the opt-in for if/when an uploader ships.
If one is built, it would summarize at that egress boundary.

`hermes telemetry preview` shows your recent runs as they'd be summarized — computed and
shown **locally only**, with the real model and tool names from your own telemetry. It's
a local inspection surface, not an upload.

## Enterprise: getting all of your data

Everything below sends data to **your own** destination — a file, your SIEM, or your own
OpenTelemetry Collector. None of it goes to Nous.

### Bulk export to a file

```bash
# Structural telemetry only (default — no message content)
hermes telemetry export --out telemetry.ndjson

# JSON instead of NDJSON, last 7 days only
hermes telemetry export --out dump.json --format json --since 7
```

By default the export is **structural** — runs, model/tool-call metadata, session shells
with message *counts* but no message bodies.

### Including content (trajectories plane)

To export full message content, enable the trajectories plane. This is a deliberate,
separate consent — it's how an enterprise opts into exporting work-product content to its
own store:

```yaml
# config.yaml
telemetry:
  trajectories:
    enabled: true          # unlocks content export to YOUR destination
  content_redaction: pii   # "none" | "pii"
```

```bash
hermes telemetry export --out full.ndjson --include-content
```

`--include-content` is a no-op unless the trajectories plane is enabled — the consent
plane governs, not the flag.

### Live streaming to your OpenTelemetry Collector / SIEM (OTLP)

Hermes can stream telemetry to your own OTLP endpoint. This requires the optional `otlp`
extra:

```bash
pip install 'hermes-agent[otlp]'
```

```yaml
# config.yaml
telemetry:
  export:
    otlp:
      enabled: true
      endpoint: "https://collector.your-corp.internal:4318/v1/traces"
      headers_env:                 # secrets by reference — env var NAMES, not values
        Authorization: CORP_OTLP_TOKEN
```

```bash
export CORP_OTLP_TOKEN="..."       # the actual token lives in the environment
hermes telemetry export --otlp     # drain current telemetry to your collector
```

Span attributes are structural by default. For authentication, the config holds the
*name* of an environment variable rather than the secret itself; the value is read at
export time and is never written to config or logged.

## Redaction

Two independent controls govern what content looks like on export:

| Control | Values | Effect |
| --- | --- | --- |
| Secret redaction | always on | API keys, tokens, auth headers, connection strings are **always** stripped on every export path. Cannot be disabled. |
| `content_redaction` | `none` \| `pii` | When content is exported, `pii` additionally redacts emails, phone numbers, and id-shaped strings. |

Secret redaction is always on — even at full content fidelity — because a SIEM or
warehouse full of live credentials is a bigger attack target than the data it holds. It
fails closed: if the redactor can't run, the raw string is not emitted.

## Configuration reference

```yaml
telemetry:
  local: true                 # local plane (default on)
  allow_aggregate: true       # hard gate; pin false to forbid the aggregate plane entirely
  consent_state: unknown      # aggregate opt-in: unknown | local | aggregate
  install_id: ""              # stable anon id; "" mints one; clear to rotate
  retention_days: 90          # local event-log retention
  redact_secrets: true        # always-on secret redaction (kept on by design)
  content_redaction: none     # none | pii
  trajectories:
    enabled: false            # unlocks full-content export to your destination
  export:
    otlp:
      enabled: false
      endpoint: null
      headers_env: {}          # {HeaderName: ENV_VAR_NAME}
```

### Enterprise policy via managed scope

Any `telemetry.*` key can be pinned by an administrator through Hermes' managed-scope
layer (`/etc/hermes/config.yaml`), which wins over the user's value on a per-key basis.
There is no telemetry-specific policy block — to lock down a fleet, pin the keys you care
about. Common examples:

- `telemetry.allow_aggregate: false` — the aggregate plane stays off even if
  `consent_state` is set to `aggregate`.
- `telemetry.export.otlp.endpoint` — point every install at the corporate collector.
- `telemetry.trajectories.enabled` — centrally decide whether content export is allowed.

When a key is managed, attempts to change it are rejected by managed scope with a message
naming the source. `hermes telemetry status` shows the current export posture (endpoint
host, whether the auth env var is set, content gate, redaction modes) — it never prints
secret values.

## Privacy summary

- Local telemetry never leaves your machine.
- The aggregate plane (the only thing that could go to Nous) is opt-in, default-off,
  and has no uploader today — nothing is sent.
- All export surfaces (file, OTLP) point at *your* destinations.
- Secrets are always redacted on export; content export is off until you enable the
  trajectories plane; PII redaction is a knob.
