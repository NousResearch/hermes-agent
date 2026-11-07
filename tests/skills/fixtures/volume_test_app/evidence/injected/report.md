# Synthetic catalog volume test

Status: completed. Execution: technical HTTP runner, not Hermes agent E2E.

Environment: {'python': '3.12.14', 'platform': 'Linux-6.18.55-x86_64-with-glibc2.41', 'sqlite': '3.53.1', 'commit': '102dee9ef7a4a8bfbf6b6e0e7a2ed25dfc8d8b56', 'started_utc': '2026-10-30T14:51:23.596101+00:00', 'ended_utc': '2026-10-30T14:51:44.095220+00:00'}
Configuration: {'run': 'volume-65b1493d1f6f47728a2f45162a4f64d5', 'seed': 42, 'volumes': [100, 1000, 10000], 'samples': 10, 'timeout_seconds': 3, 'duration_seconds': 120, 'max_records': 10000, 'max_errors': 50, 'concurrency': 1, 'page_size': 50, 'batch_size': 250, 'counted_entity': 'products', 'cleanup': 'Delete only products/categories with this exact run; verify unrelated sentinel', 'dataset': {'fields': ['id', 'name', 'price', 'category'], 'relationships': '8 run-owned categories; enforced product category foreign key', 'distribution': 'Repeated Unicode names and prices; skewed categories; seed/index-stable prefix'}, 'operations': [{'name': 'listing', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'search', 'endpoint': '/products', 'parameters': {'q': 'Café', 'page': 1}}, {'name': 'category', 'endpoint': '/products', 'parameters': {'category': '0', 'page': 1}}, {'name': 'price-sort', 'endpoint': '/products', 'parameters': {'sort': 'price', 'page': 1}}, {'name': 'last-page', 'endpoint': '/products', 'parameters': {'page': 1}}, {'name': 'export', 'endpoint': '/export', 'parameters': {}}, {'name': 'filtered-export', 'endpoint': '/export', 'parameters': {'category': '0'}}], 'criteria': 'Exact independently generated ordered content, totals, complete pagination/export, no orphans; zero residuals and intact sentinel', 'cache_policy': 'First request follows preparation/page verification; no OS/SQLite cache flush; 1 warmup + 10 warm samples', 'performance_sla': None, 'target_url': 'http://127.0.0.1:42575', 'application_repository': 'tests/skills/fixtures/volume_test_app'}

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
| listing | 100 | 10 | 0.857 | 0.679 | 1.322 | 0 | 0 | 0 | 1.000 |
| search | 100 | 10 | 0.604 | 0.568 | 0.848 | 0 | 0 | 0 | 1.000 |
| category | 100 | 10 | 0.809 | 0.642 | 2.067 | 0 | 0 | 0 | 1.000 |
| price-sort | 100 | 10 | 1.041 | 0.869 | 2.975 | 0 | 0 | 0 | 1.000 |
| last-page | 100 | 10 | 0.768 | 0.709 | 0.923 | 0 | 0 | 0 | 1.000 |
| export | 100 | 10 | 1.019 | 0.899 | 1.187 | 0 | 0 | 0 | 1.000 |
| filtered-export | 100 | 10 | 0.930 | 0.703 | 1.179 | 0 | 0 | 0 | 1.000 |
| listing | 1000 | 10 | 1.082 | 1.013 | 1.208 | 0 | 0 | 0 | 1.263 |
| search | 1000 | 10 | 1.102 | 1.060 | 1.378 | 0 | 0 | 0 | 1.825 |
| category | 1000 | 10 | 1.100 | 1.010 | 1.372 | 0 | 0 | 0 | 1.360 |
| price-sort | 1000 | 10 | 1.043 | 1.021 | 1.350 | 0 | 0 | 0 | 1.002 |
| last-page | 1000 | 10 | 1.222 | 1.209 | 1.380 | 0 | 0 | 0 | 1.591 |
| export | 1000 | 10 | 0.982 | 0.936 | 1.011 | 0 | 0 | 10 | 0.964 |
| filtered-export | 1000 | 10 | 2.101 | 2.071 | 2.157 | 0 | 0 | 0 | 2.258 |
| listing | 10000 | 10 | 2.731 | 2.658 | 2.919 | 0 | 0 | 0 | 3.186 |
| search | 10000 | 10 | 2.766 | 2.696 | 2.914 | 0 | 0 | 0 | 4.579 |
| category | 10000 | 10 | 2.896 | 2.839 | 3.075 | 0 | 0 | 0 | 3.581 |
| price-sort | 10000 | 10 | 2.727 | 2.698 | 2.824 | 0 | 0 | 0 | 2.620 |
| last-page | 10000 | 10 | 7.392 | 7.323 | 7.562 | 0 | 0 | 0 | 9.625 |
| export | 10000 | 10 | 2.892 | 2.831 | 2.944 | 0 | 0 | 10 | 2.839 |
| filtered-export | 10000 | 10 | 3.034 | 2.991 | 3.254 | 0 | 0 | 10 | 3.261 |

## Correctness and evidence
Incorrect/error samples: 36. See raw samples.json and result.json for expected/actual counts, missing/unexpected rows, stage page walks and errors.
All page walks use an independent Python filter/sort reference, including final and empty pages, ties and full identifier coverage.
Controlled fault enabled: True. Any truncated export in that scenario is deliberately injected, not a spontaneous finding.
First failure: export at 1000 products, expected 1000, observed 50, error=None.
Reproduce with the documented --inject-export-bug flag; compare CSV content to the independently generated complete filtered set.

## Technical UI measurements
Browser: {'version': '154.0.8037.57', 'viewport': {'width': 1100, 'height': 800}, 'errors': [], 'screenshots': ['ui-100.png', 'ui-1000.png', 'ui-10000.png']}. DOM rows, displayed totals and request completion are checked against the independent reference. UI timing includes interaction and waiting for correct rendered content, not just navigation.
| Operation | Products | Warm n | Median UI ms | Errors | Timeouts |
| --- | --- | --- | --- | --- | --- |
| listing | 100 | 10 | 63.54509100492578 | 0 | 0 |
| search | 100 | 10 | 66.07091749901883 | 0 | 0 |
| category | 100 | 10 | 63.951773510780185 | 0 | 0 |
| price-sort | 100 | 10 | 64.45911549963057 | 0 | 0 |
| last-page | 100 | 10 | 62.26824900659267 | 0 | 0 |
![Catalog at 100 products](ui-100.png)
| listing | 1000 | 10 | 62.33645899919793 | 0 | 0 |
| search | 1000 | 10 | 63.948132999939844 | 0 | 0 |
| category | 1000 | 10 | 61.1478744976921 | 0 | 0 |
| price-sort | 1000 | 10 | 63.36279501556419 | 0 | 0 |
| last-page | 1000 | 10 | 64.60593751398847 | 0 | 0 |
![Catalog at 1000 products](ui-1000.png)
| listing | 10000 | 10 | 59.469030500622466 | 0 | 0 |
| search | 10000 | 10 | 61.925451998831704 | 0 | 0 |
| category | 10000 | 10 | 60.90806899010204 | 0 | 0 |
| price-sort | 10000 | 10 | 62.58332700235769 | 0 | 0 |
| last-page | 10000 | 10 | 61.71042699133977 | 0 | 0 |
![Catalog at 10000 products](ui-10000.png)

## Coverage and limitations
HTTP listing, pagination, search, category filtering, stable name/price sorting, totals and full CSV content are covered only where stage entries/samples exist.
Real Hermes agent skill invocation, CPU/memory attribution and other platforms are unrun in this technical execution. UI timings/screenshots and console/network capture are covered only when ui_samples/browser evidence is present; otherwise unrun. Local results do not establish production capacity or arbitrary-app support.
Interruption/blocker: none.

## Cleanup
{'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True, 'residuals': {'products': 0, 'categories': 0, 'orphaned': 0, 'sentinel_intact': True}, 'verified': True}
