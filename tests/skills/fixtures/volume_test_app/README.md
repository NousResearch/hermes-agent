# Synthetic Product Catalog fixture

This is a disposable local demonstration, not an adapter required by the skill. It has no external services, credentials, commercial samples or professional data. The normal application has correct list/filter/sort/export behavior. `--inject-export-bug` deliberately truncates exports with at least 1,000 matching products to 50 rows; this is a detector validation, not a naturally discovered defect or manufactured slowdown.

## Reproduce

From the repository root, using Python 3.11+ (the Hermes/test environment uses Python 3.14):

```sh
python tests/skills/fixtures/volume_test_app/run_demo.py --output /path/to/fresh/healthy
python tests/skills/fixtures/volume_test_app/run_demo.py --output /path/to/fresh/injected --inject-export-bug
```

The runner starts a loopback-only HTTP server on an OS-assigned free port, uses a temporary SQLite database, writes its plan before seeding, and checks 100, 1,000 and 10,000 **total run-owned products** with seed 42. Eight categories and one unrelated sentinel product/category are counted separately. Each stage uses bounded batches of 250, fixed page size 50, a single sequential client, complete page walks and a Python filter/sort oracle independent of the SQL implementation. Query content includes Unicode, repeated names/prices, stable identifier tie-breaks and skewed category distributions.

Each operation has a first-request sample, one warmup and ten warmed samples. The first request follows preparation and page-walk verification: OS/SQLite caches are not controlled. Request timing includes body download, not parsing/oracle validation. A 3-second per-request timeout, 120-second default overall budget (checked between operations), 10,000-product ceiling and 50-error/correctness-failure ceiling bound the demonstration. Cleanup makes a bounded attempt even on interruption and preserves the sentinel. Only the exact run namespace is deleted. The temporary database itself is discarded after the fixture stops, so cleanup verification is recorded **before** its removal.

Optional UI verification uses an already installed Playwright and a local Chromium executable in a disposable environment, not a new Hermes dependency:

```sh
python tests/skills/fixtures/volume_test_app/run_demo.py --output /path/to/fresh/healthy-ui --browser-executable /path/to/chromium
python tests/skills/fixtures/volume_test_app/run_demo.py --output /path/to/fresh/injected-ui --browser-executable /path/to/chromium --inject-export-bug
```

UI measurements use viewport 1100×800, the same search/category/sort/page operations, actual form/button interaction and a verified completion condition: a newer request, correct displayed total/page and exact rendered table content. There are no fixed sleeps. Screenshots, console/page errors and HTTP failure responses are captured. UI and request timing populations remain separate. CSV correctness is checked through HTTP, not by pretending the visible list proves a download complete. The container demonstration launches Chromium with `--no-sandbox` only against this synthetic loopback fixture; do not use that configuration for untrusted websites.

The UI and HTTP client run sequentially, never concurrently. This harness is an application-specific technical test. It does **not** prove Hermes loaded or executed the skill. A real agent run requires a configured model and an isolated Hermes home; install using `hermes skills install official/software-development/volume-test --yes`, then invoke the documented skill with repository, loopback URL, seed, volumes and output directory. Never copy a personal Hermes profile into the demonstration.

## Data contracts

- `POST /prepare`: `{run, rows}`; at most 250 rows, transactional batch, primary key `(run, id)`. Identical retries are idempotent; conflicting content for the same identity is rejected. On a timeout/partial or inconsistent response, the runner reconciles persisted identifiers and saves the manifest instead of retrying blindly. Calling preparation again resumes missing indices without duplication.
- `GET /products`: `run`, `q`, `category`, `sort=name|price`, one-based `page`, `page_size` (1–200). The total is for the complete filtered set. Both orders break ties by numeric logical identifier. Search is case-sensitive substring matching.
- `GET /export`: same filters/order, complete filtered CSV by default.
- `GET /counts`: products, category rows, orphan count, unrelated sentinel check for this run.
- `POST /cleanup`: exact `{run}` namespace starting `volume-`; removes only its products/categories. Generic resets and sentinel deletion are refused.

The runner generates a new run identity independently from the seed. Output must be a fresh directory. `plan.json`, `manifest.json`, raw `samples.json`, `result.json`, `report.md` and optional PNGs survive interruption/cleanup. A completed preparation stage is recorded only after actual counts/relations match; raw content or request failures remain visible. No SLA, p95 claim, causal database diagnosis, production-capacity guarantee or arbitrary-app compatibility is inferred.

## Contribution provenance

Base inspected: `2d2808ee84549f9836b0056750265058e5488a8a`.
Normative sources: root `AGENTS.md`, `CONTRIBUTING.md`, `skills/AGENTS.md`, `tests/AGENTS.md`, `.github/PULL_REQUEST_TEMPLATE.md`, and `website/docs/reference/package-management.md`. The bundled dogfood skill and optional adversarial UX skill were read; their purposes are preserved. No additional AGENTS.md exists under the new fixture or optional skill path.

Duplicate discovery used GitHub issue/PR searches (all states, issue and PR types separately) for exact `volume-test`, `volume testing`, `data-heavy`, `large dataset`, `performance testing`, `dogfood`, and `seeding`, plus source searches. Exact-name searches had no matches. Relevant exploratory-QA precedent was examined: issue #315, PR #38866 and merged category relocation #70428. This is not proof that no equivalent work can ever exist; placement remains subject to maintainers. Proposal: [#146896](https://github.com/NousResearch/hermes-agent/issues/146896).
