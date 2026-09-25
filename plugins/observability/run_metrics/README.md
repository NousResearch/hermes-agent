# Local run metrics

Opt-in, metadata-only diagnostics for one Hermes turn. Enable with
`hermes plugins enable observability/run_metrics`; disable with
`hermes plugins disable observability/run_metrics`. No network exporter is used.
Each enabled turn writes an atomically replaced, mode-0600 JSON snapshot under
`<active HERMES_HOME>/logs/run-metrics/<run_id>.json`. The file remains
`status: running` after a hard process kill; normal interruptions close it.
`started_at` and `ended_at` are UTC Unix seconds.

The report distinguishes logical model calls from physical provider attempts,
including retries inside a streaming call and the non-streaming 5xx probe.
Each logical call has an `attempts[]` entry; internal wire requests are in its
`transport_attempts[]`. `provider_attempts_complete=false` means the selected
transport or a final budget-summary call has requests not exposed through the
generic hooks; the numeric attempt count is then a lower bound.
Provider-reported tokens belong to the successful
response only; failed requests may have consumed unreported tokens. `input_tokens`
is the provider's total prompt-token count, including cached tokens when the
provider reports them. `uncached_input_tokens`, `cache_read_tokens`, and
`cache_write_tokens` are separate buckets; null means unavailable, not zero.
Native adapters that fill absent usage fields with synthetic zeroes leave
provider-usage attribution unavailable until they expose raw-field provenance.
Request-size estimates are never substituted for provider usage. `time_to_first_chunk_s`
is the first wire chunk, which may not contain generated text;
`time_to_first_delta_s` is the first text, reasoning, or tool-name delta. Both
are client-observed. A non-streamed call has neither. `model_time_s` sums
client-observed physical request durations, excluding outer retry backoff;
it is not GPU execution time. `context_limit_tokens`
is Hermes's resolved configured window, not a server-enforced context ceiling.
`estimated_input_tokens` and `estimated_context_utilization` are explicitly
estimates; provider-reported input/output counts are separate. Decode tokens/s
uses provider-reported output divided by client-observed time after the first
delta, so it is an approximate throughput diagnostic, not GPU benchmark data.
No prompt, response, tool arguments/results,
endpoint, error message, or credentials are stored.

Aggregate saved runs with `python scripts/run_metrics_report.py
<HERMES_HOME>/logs/run-metrics`. Root runs and delegated child runs are counted
separately; the summary does not add child tokens to a parent. Join an external
GPU/VRAM/RAM sampler by `run_id` and `started_at`/`ended_at`; Hermes does not
claim those host-side measurements. The report describes agent execution, not
the correctness of its code or tests. Validate that independently before
running a batch.

`finish_reason` is the loop's recovery reason; `provider_finish_reason` is the
normalized provider response. A partial stream and an inferred truncation are
not counted as provider output caps. Exhausting the iteration budget is a
distinct `budget_exhausted` status even if a final fallback answer was returned.

`sample-report.json` is redacted from a loopback fake-provider smoke. Its
20,000 tokens/s figure is **not** a model-performance measurement. A separate
headless-adapter smoke reached Hermes and produced a report with the fake
provider's exact 100 input / 20 output tokens and `finish_reason=stop`.
Hosts that create ephemeral `HERMES_HOME` directories must explicitly enable
the plugin there and export the report before cleanup. The host must also
join Hermes's session/turn identifiers to its own issue identifier; Hermes's
`task_id` is not necessarily the host's issue ID. This smoke does not prove
real-model speed or batch readiness.

In a local 40-turn, four-hook microbenchmark (no provider call), plugin-off
median/p95 were 0.15/0.17 ms per turn; plugin-on were 1.52/2.24 ms. This
measures only event/report overhead on the test Mac and does not include model,
tool, CLI startup, or host-adapter overhead. Storage errors are fail-open; a
missing artifact is a metrics failure, never evidence that a run completed.
