# Synthetic catalog volume test

Status: completed. Execution: technical HTTP runner, not Hermes agent E2E.

Environment: {'python': '3.14.4', 'platform': 'Linux-7.0.0-38-generic-x86_64-desconhecido-with-glibc2.43', 'sqlite': '3.46.1', 'commit': 'bdd3ac64af2363ab73f7e78e694d23b0060f229b', 'started_utc': '2026-10-30T15:18:10.846026+00:00', 'ended_utc': '2026-10-30T15:18:16.262642+00:00'}
Configuration: {'run': 'volume-817661b305154929b6d2baa430d0da19', 'seed': 42, 'volumes': [100, 1000, 10000], 'samples': 10, 'timeout_seconds': 3, 'duration_seconds': 120, 'max_records': 10000, 'max_errors': 50, 'concurrency': 1, 'page_size': 50, 'batch_size': 250, 'counted_entity': 'products', 'cleanup': 'Delete only products/categories with this exact run; verify unrelated sentinel', 'dataset': {'fields': ['id', 'name', 'price', 'category'], 'relationships': '8 run-owned categories; enforced product category foreign key', 'distribution': 'Repeated Unicode names and prices; skewed categories; seed/index-stable prefix'}, 'operations': [{'name': 'listing', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'search', 'endpoint': '/products', 'parameters': {'q': 'Café', 'page': 1}}, {'name': 'category', 'endpoint': '/products', 'parameters': {'category': '0', 'page': 1}}, {'name': 'price-sort', 'endpoint': '/products', 'parameters': {'sort': 'price', 'page': 1}}, {'name': 'last-page', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'export', 'endpoint': '/export', 'parameters': {}}, {'name': 'filtered-export', 'endpoint': '/export', 'parameters': {'category': '0'}}], 'criteria': 'Exact independently generated ordered content, totals, complete pagination/export, no orphans; zero residuals and intact sentinel', 'cache_policy': 'First request follows preparation/page verification; no OS/SQLite cache flush; 1 warmup + 10 warm samples', 'performance_sla': None, 'target_url': 'http://127.0.0.1:44471', 'application_repository': 'tests/skills/fixtures/volume_test_app'}

Preparation, page-walk verification and measurement are separate. Concurrency: one sequential client; page size: 50; one warmup and ten warmed samples per operation.
The separately labeled first request is after preparation/verification; neither OS nor SQLite caches were flushed. No cold-cache claim. HTTP body download timing excludes oracle comparison and UI rendering.
No performance SLA was supplied. Ratios are observations, not bugs or infrastructure diagnoses. No sleeps manufacture latency; no percentiles are reported from these small samples.

## Confirmed volumes
| Target products | Actual products | Categories | Orphans |
| --- | --- | --- | --- |
| 100 | 100 | 8 | 0 |
| 1000 | 1000 | 8 | 0 |
| 10000 | 10000 | 8 | 0 |

## Warmed HTTP measurements
| Operation | Products | n | Median ms | Min ms | Max ms | Errors | Timeouts | Incorrect | Ratio to baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| listing | 100 | 10 | 0.929 | 0.848 | 1.240 | 0 | 0 | 0 | 1.000 |
| search | 100 | 10 | 0.630 | 0.478 | 0.904 | 0 | 0 | 0 | 1.000 |
| category | 100 | 10 | 0.559 | 0.544 | 1.072 | 0 | 0 | 0 | 1.000 |
| price-sort | 100 | 10 | 0.611 | 0.543 | 1.248 | 0 | 0 | 0 | 1.000 |
| last-page | 100 | 10 | 0.849 | 0.590 | 0.997 | 0 | 0 | 0 | 1.000 |
| export | 100 | 10 | 1.039 | 0.844 | 1.271 | 0 | 0 | 0 | 1.000 |
| filtered-export | 100 | 10 | 1.022 | 0.954 | 1.222 | 0 | 0 | 0 | 1.000 |
| listing | 1000 | 10 | 1.071 | 0.781 | 1.415 | 0 | 0 | 0 | 1.153 |
| search | 1000 | 10 | 0.871 | 0.808 | 1.001 | 0 | 0 | 0 | 1.383 |
| category | 1000 | 10 | 1.271 | 0.782 | 1.399 | 0 | 0 | 0 | 2.272 |
| price-sort | 1000 | 10 | 1.228 | 0.808 | 1.468 | 0 | 0 | 0 | 2.009 |
| last-page | 1000 | 10 | 1.725 | 1.613 | 2.115 | 0 | 0 | 0 | 2.032 |
| export | 1000 | 10 | 1.017 | 0.850 | 1.301 | 0 | 0 | 10 | 0.979 |
| filtered-export | 1000 | 10 | 2.121 | 1.727 | 3.654 | 0 | 0 | 0 | 2.075 |
| listing | 10000 | 10 | 3.395 | 3.087 | 3.639 | 0 | 0 | 0 | 3.654 |
| search | 10000 | 10 | 3.418 | 3.175 | 3.837 | 0 | 0 | 0 | 5.428 |
| category | 10000 | 10 | 3.499 | 3.151 | 4.080 | 0 | 0 | 0 | 6.256 |
| price-sort | 10000 | 10 | 3.212 | 3.017 | 3.595 | 0 | 0 | 0 | 5.253 |
| last-page | 10000 | 10 | 8.144 | 7.798 | 8.419 | 0 | 0 | 0 | 9.595 |
| export | 10000 | 10 | 3.728 | 3.327 | 3.947 | 0 | 0 | 10 | 3.590 |
| filtered-export | 10000 | 10 | 3.650 | 3.452 | 4.036 | 0 | 0 | 10 | 3.570 |

## Correctness and evidence
Incorrect/error samples: 36. See raw samples.json and result.json for expected/actual counts, missing/unexpected rows, stage page walks and errors.
All page walks use an independent Python filter/sort reference, including final and empty pages, ties and full identifier coverage.
Controlled fault enabled: True. Any truncated export in that scenario is deliberately injected, not a spontaneous finding.
First failure: export at 1000 products, expected 1000, observed 50, error=None.
Reproduce with the documented --inject-export-bug flag; compare CSV content to the independently generated complete filtered set.

## Coverage and limitations
HTTP listing, pagination, search, category filtering, stable name/price sorting, totals and full CSV content are covered only where stage entries/samples exist.
Real Hermes agent skill invocation, CPU/memory attribution and other platforms are unrun in this technical execution. UI timings/screenshots and console/network capture are covered only when ui_samples/browser evidence is present; otherwise unrun. Local results do not establish production capacity or arbitrary-app support.
Interruption/blocker: none.

## Cleanup
{'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True, 'residuals': {'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True}, 'verified': True}
