# Observability Checklist

Use this checklist alongside the `observability-and-instrumentation` skill to verify that a feature ships with the telemetry needed to operate it. Everything below should be confirmed before a production-bound change merges.

## Questions Before Instrumentation
- [ ] The 2–4 on-call questions for this feature are written down
- [ ] Every signal maps to at least one of those questions
- [ ] Telemetry is being added alongside the feature, not "later"

## Structured Logging
- [ ] All log output is structured (JSON) with a stable event name and machine-readable fields
- [ ] A correlation/request ID is attached to every log line, span, and outbound call
- [ ] Log levels (error/warn/info/debug) are used consistently
- [ ] No secrets, tokens, or unredacted PII appear in any log line (spot-check actual output)

## Metrics
- [ ] RED metrics (Rate, Errors, Duration) exist for every new endpoint and external dependency
- [ ] USE metrics (Utilization, Saturation, Errors) exist for constrained resources
- [ ] Latency is a histogram; p95/p99 are queryable (no averages-only)
- [ ] Labels come from small, fixed sets — no user IDs, raw URLs, or error text as labels (cardinality)

## Tracing
- [ ] A single request can be followed end-to-end in the tracing UI with no broken spans
- [ ] Context is propagated across every async/queue/HTTP boundary
- [ ] Manual spans added only around meaningful units of work

## Alerting
- [ ] Every new alert is symptom-based (user-facing), not cause-based
- [ ] Each alert links to a runbook and was test-fired once
- [ ] Thresholds/durations are justified by an SLO or historical data
- [ ] Only two severities used: page (act now) and ticket (this week)

## Verification
- [ ] An induced failure in staging was located via telemetry alone, without reading source
- [ ] Metric series appear with expected labels and sane values under test traffic
