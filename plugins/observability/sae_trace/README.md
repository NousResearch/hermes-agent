# SAE Feature-Trace Observability Plugin

Correlates each Hermes agent turn with the **same-inference** SAE (sparse
autoencoder) feature-trace records emitted by an SAE-hooked local
OpenAI-compatible model server — a per-session interpretability trace of
what your local model's internals were doing while it ran the agent.

This plugin ships bundled with Hermes but is **opt-in** — it only loads
when you explicitly enable it. It is a pure read-only observer on the
`hermes.observer.v1` hooks contract (see
[`docs/observability/README.md`](../../../docs/observability/README.md)):
it never modifies requests, responses, or the sidecar file, uses only the
Python stdlib, and touches only local files (no network).

The plugin has no hardware or ML-framework requirements of its own — it is
pure file correlation (no `torch`, no GPU). Whatever your model runs on —
a CPU-only laptop, a single consumer GPU, Apple Silicon, or a multi-GPU
workstation — the plugin only needs the sidecar JSONL your server writes.

## How it works

1. You run your local model behind an SAE-instrumented OpenAI-compatible
   server. The server wraps `model.generate()` with forward hooks and, in
   the **same inference pass** that returns the completion, appends one
   JSONL record (top-k SAE features per generated token, or a pointer to
   raw activations) to a sidecar file. Reference implementation:
   [SolshineCode/hermes-sae](https://github.com/SolshineCode/hermes-sae) —
   but **any** OpenAI-compatible server emitting the sidecar schema below
   works.
2. Hermes talks to that server as a normal custom model provider.
3. This plugin registers `pre_api_request` / `post_api_request` observer
   hooks. On each provider response it tail-reads the sidecar (remembered
   byte offset; rotation/truncation safe), matches records appended
   during the request against the request's identity and timing, and
   appends one correlated line per agent turn to
   `$HERMES_SAE_TRACE_OUT_DIR/<session_id>.jsonl`.

## Enable

```bash
hermes plugins enable observability/sae_trace
```

### Standalone install

The plugin also works unmodified from the user-plugin directory (no
Hermes source tree needed):

```bash
mkdir -p ~/.hermes/plugins
cp -r plugins/observability/sae_trace ~/.hermes/plugins/sae_trace
hermes plugins enable sae_trace
```

## Configuration

Set in `~/.hermes/.env`:

```bash
# Required — path to the SAE server's sidecar JSONL. Without it the
# hooks are inert (fail-open, like the langfuse plugin).
HERMES_SAE_TRACE_FILE=/path/to/sae_history.jsonl

# Optional
HERMES_SAE_TRACE_OUT_DIR=~/.hermes/sae_trace   # default: $HERMES_HOME/sae_trace
HERMES_SAE_TRACE_SKEW=10                       # time-window slack (seconds)
HERMES_SAE_TRACE_DEBUG=true                    # verbose plugin logging
```

## Pointing Hermes at an SAE-hooked server

Add an OpenAI-compatible provider whose `base_url` is your instrumented
server, e.g. in `config.yaml`:

```yaml
providers:
  sae:
    base_url: http://127.0.0.1:8077/v1
model:
  default: sae-local
  provider: sae
```

Because Hermes talks to the server over the plain OpenAI API, running
with the instrumented model records the SAE feature history of the
agent's own activity automatically — no core changes, no replay, no
second forward pass.

## Sidecar JSONL schema (the interface contract)

One JSON object per line, appended per request **in the same inference
pass that produced the completion**. Recognized fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `request_id` | recommended | Server-side request identity. If the caller passes Hermes' `api_request_id` through (body `request_id` or `X-Request-Id` header), correlation is exact. |
| `session_id` | recommended | Echo of the `X-Hermes-Session-Id` header (or body `session_id`) when the client sends one. |
| `timestamp` or `ts` | yes* | ISO-8601 record time. Naive values are read as local time; a `Z` suffix as UTC. Numeric epoch seconds also accepted. (*Required for time-window matching; records without it match only by `request_id`/`session_id`.) |
| `model` | recommended | Served model id. Checked (case-insensitive) against the request's `model` / `response_model` for time-window matches; absent means no model constraint. |
| `feats_topk` | either this… | Per-token top-k SAE features: `{layer: [[token_pos, feature_id, activation], ...]}`. Summarized into per-layer top features. |
| `npz_path` | …or this | Path to raw activations (`.npz`) when features aren't computed inline. Carried through as a pointer. |
| `layer` / `layers`, `d_model`, `n_records`, `n_gen_tokens`, `prompt_len`, `gen_len`, `same_inference` | optional | Shape/provenance scalars, carried through verbatim. |
| `gen_text_preview` or `gen_text` | optional | Generated-text preview (truncated to 200 chars in output). |
| `allf` | optional | Full-dictionary activations. Never copied — flagged as `has_all_features: true`. |

Unknown fields are ignored; malformed lines are skipped with a debug log.

## Output format

One line per correlated agent turn in
`$HERMES_SAE_TRACE_OUT_DIR/<session_id>.jsonl`:

```json
{
  "ts": "2026-07-21T04:12:31.902+00:00",
  "session_id": "abc123",
  "task_id": "",
  "turn_id": "turn-7",
  "api_request_id": "req-42",
  "model": "sae-local",
  "api_duration": 12.4,
  "match_confidence": "time_window",
  "records": [
    {
      "request_id": "chatcmpl-…",
      "ts": "2026-07-21T04:12:30Z",
      "model": "sae-local",
      "gen_len": 118,
      "top_features": {"16": [[40913, 8.31, 22], [1027, 6.9, 4]]},
      "gen_text_preview": "Sure — here's the plan…"
    }
  ]
}
```

`top_features` is `{layer: [[feature_id, max_activation, n_tokens], ...]}`
(top 10 per layer by max activation). Turns with no matching sidecar
records write nothing (counted in `/sae status`).

## `/sae` slash command

In any CLI or gateway session:

- `/sae status` — sidecar path, records seen/matched this session,
  unmatched turns, last turn's top features
- `/sae last` — the most recent turn's correlated feature summary

## Correlation confidence (limitations)

`match_confidence` records the strongest evidence tier used:

| Tier | Evidence | Notes |
| --- | --- | --- |
| `request_id` | Sidecar `request_id` equals Hermes' `api_request_id` | Exact. Requires the provider path to forward the id (body `request_id` / `X-Request-Id`); not the default. |
| `session_id` | Sidecar `session_id` equals the Hermes session id | Strong. Requires the server to log the `X-Hermes-Session-Id` header (the reference `nla_server` does). |
| `time_window` | Record timestamp inside `[started_at − skew, ended_at + skew]` and model id compatible | Heuristic. Reliable for a single local server (which typically serializes generation), but concurrent Hermes sessions sharing one server within the same window can mis-attribute records. Prefer id-based tiers when possible. |

Other limitations:

- The sidecar's `model` id must equal Hermes' `model` (or the response's
  `model`) for time-window matches; if your server logs a different
  internal id, records without a `session_id`/`request_id` won't match.
- Correlation state is per-process: `/sae` counters reset on restart, and
  sidecar records emitted while Hermes is not running are never matched
  (the tailer starts at end-of-file).
- Only local-model turns routed through the instrumented server produce
  records; cloud-provider turns simply have no trace (by design).

## Security

Local files only. The plugin reads one operator-configured sidecar file
and appends to `$HERMES_SAE_TRACE_OUT_DIR`; it makes **no network
requests** and sends nothing anywhere. Output records contain a bounded
preview of generated text (≤200 chars per record) plus feature ids —
treat the output directory with the same care as session logs. Session
ids are sanitized to a single traversal-free path segment before being
used as filenames.

## Disable

```bash
hermes plugins disable observability/sae_trace
```
