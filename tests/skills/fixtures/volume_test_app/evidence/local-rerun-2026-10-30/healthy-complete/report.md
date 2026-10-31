# Synthetic catalog volume test

Status: completed. Execution: technical HTTP runner, not Hermes agent E2E.

Environment: {'python': '3.14.4', 'platform': 'Linux-7.0.0-38-generic-x86_64-desconhecido-with-glibc2.43', 'sqlite': '3.46.1', 'commit': 'bdd3ac64af2363ab73f7e78e694d23b0060f229b', 'started_utc': '2026-10-30T15:21:41.524165+00:00', 'ended_utc': '2026-10-30T15:21:48.144692+00:00'}
Configuration: {'run': 'volume-e90679faf6f34210a25545ba2fa2aef4', 'seed': 42, 'volumes': [100, 1000, 10000], 'samples': 10, 'timeout_seconds': 3, 'duration_seconds': 120, 'max_records': 10000, 'max_errors': 50, 'concurrency': 1, 'page_size': 50, 'batch_size': 250, 'counted_entity': 'products', 'cleanup': 'Delete only products/categories with this exact run; verify unrelated sentinel', 'dataset': {'fields': ['id', 'name', 'price', 'category'], 'relationships': '8 run-owned categories; enforced product category foreign key', 'distribution': 'Repeated Unicode names and prices; skewed categories; seed/index-stable prefix'}, 'operations': [{'name': 'listing', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'search', 'endpoint': '/products', 'parameters': {'q': 'Café', 'page': 1}}, {'name': 'category', 'endpoint': '/products', 'parameters': {'category': '0', 'page': 1}}, {'name': 'price-sort', 'endpoint': '/products', 'parameters': {'sort': 'price', 'page': 1}}, {'name': 'last-page', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'export', 'endpoint': '/export', 'parameters': {}}, {'name': 'filtered-export', 'endpoint': '/export', 'parameters': {'category': '0'}}], 'criteria': 'Exact independently generated ordered content, totals, complete pagination/export, no orphans; zero residuals and intact sentinel', 'cache_policy': 'First request follows preparation/page verification; no OS/SQLite cache flush; 1 warmup + 10 warm samples', 'performance_sla': None, 'target_url': 'http://127.0.0.1:40121', 'application_repository': 'tests/skills/fixtures/volume_test_app'}

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
| listing | 100 | 10 | 0.552 | 0.538 | 0.671 | 0 | 0 | 0 | 1.000 |
| search | 100 | 10 | 0.565 | 0.499 | 0.984 | 0 | 0 | 0 | 1.000 |
| category | 100 | 10 | 0.900 | 0.561 | 1.089 | 0 | 0 | 0 | 1.000 |
| price-sort | 100 | 10 | 0.982 | 0.593 | 1.124 | 0 | 0 | 0 | 1.000 |
| last-page | 100 | 10 | 0.977 | 0.912 | 1.212 | 0 | 0 | 0 | 1.000 |
| export | 100 | 10 | 0.749 | 0.665 | 1.375 | 0 | 0 | 0 | 1.000 |
| filtered-export | 100 | 10 | 0.898 | 0.641 | 1.352 | 0 | 0 | 0 | 1.000 |
| listing | 1000 | 10 | 0.884 | 0.824 | 1.106 | 0 | 0 | 0 | 1.602 |
| search | 1000 | 10 | 0.917 | 0.836 | 1.369 | 0 | 0 | 0 | 1.623 |
| category | 1000 | 10 | 1.366 | 1.279 | 1.580 | 0 | 0 | 0 | 1.518 |
| price-sort | 1000 | 10 | 1.263 | 0.879 | 1.351 | 0 | 0 | 0 | 1.286 |
| last-page | 1000 | 10 | 1.708 | 1.200 | 1.917 | 0 | 0 | 0 | 1.747 |
| export | 1000 | 10 | 3.940 | 2.760 | 4.893 | 0 | 0 | 0 | 5.263 |
| filtered-export | 1000 | 10 | 2.215 | 1.963 | 4.086 | 0 | 0 | 0 | 2.467 |
| listing | 10000 | 10 | 3.430 | 3.165 | 4.829 | 0 | 0 | 0 | 6.216 |
| search | 10000 | 10 | 3.807 | 3.157 | 4.543 | 0 | 0 | 0 | 6.741 |
| category | 10000 | 10 | 3.648 | 3.286 | 4.223 | 0 | 0 | 0 | 4.055 |
| price-sort | 10000 | 10 | 3.388 | 3.010 | 3.833 | 0 | 0 | 0 | 3.451 |
| last-page | 10000 | 10 | 8.510 | 8.182 | 9.437 | 0 | 0 | 0 | 8.707 |
| export | 10000 | 10 | 24.499 | 23.408 | 28.671 | 0 | 0 | 0 | 32.722 |
| filtered-export | 10000 | 10 | 15.802 | 14.304 | 16.702 | 0 | 0 | 0 | 17.600 |

## Correctness and evidence
Incorrect/error samples: 0. See raw samples.json and result.json for expected/actual counts, missing/unexpected rows, stage page walks and errors.
All page walks use an independent Python filter/sort reference, including final and empty pages, ties and full identifier coverage.
Controlled fault enabled: False. Any truncated export in that scenario is deliberately injected, not a spontaneous finding.

## Coverage and limitations
HTTP listing, pagination, search, category filtering, stable name/price sorting, totals and full CSV content are covered only where stage entries/samples exist.
Real Hermes agent skill invocation, CPU/memory attribution and other platforms are unrun in this technical execution. UI timings/screenshots and console/network capture are covered only when ui_samples/browser evidence is present; otherwise unrun. Local results do not establish production capacity or arbitrary-app support.
Interruption/blocker: none.

## Cleanup
{'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True, 'residuals': {'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True}, 'verified': True}
