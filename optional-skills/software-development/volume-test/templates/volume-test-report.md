# Volume test report

## Environment and scope
UTC start/end, application revision, runtime/platform, repository, target, seed/run ID, counted entity, fixed workload/concurrency, limits and source contracts.

## Methodology
Preparation versus measurement, first-request/cache controls, warmups/repeats, request versus UI completion metrics, stable sorting/page size, independent oracle.

## Volumes
| Target | Confirmed entity count | Related rows | Preparation status |
| --- | --- | --- | --- |

## Measurements and correctness
| Operation | Volume | Warm samples | Median ms | Min ms | Max ms | Errors | Timeouts | Correctness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Link raw samples. Compare matching operations against baseline without inventing an SLA.

## Findings
For each: observed behavior, expected result, reproducible steps, evidence, scope, and whether naturally observed or deliberately injected. Separate hypotheses from demonstrated causes.

## Coverage and limitations
Completed, failed, interrupted, unrun and blocked operations; browser/agent execution status; no production/general compatibility claims.

## Cleanup
Run-owned identifiers, confirmed residuals, sentinel checks, failures and any required follow-up. Preserve the manifest and raw evidence.
