# Pact Consumer CDC — Framework Setup

## Principle

When scaffolding a Pact.js consumer contract testing framework, align every artifact — directory layout, vitest config, package.json scripts, shell scripts, CI workflow, and test files — with the canonical `@seontechnologies/pactjs-utils` conventions. Consistency across repositories eliminates onboarding friction and ensures CI pipelines are copy-paste portable.

## Rationale

The TEA framework workflow generates scaffolding for consumer-driven contract (CDC) testing. Without opinionated, battle-tested conventions, each project invents its own structure — different script names, different env var patterns, different CI step ordering — making cross-repo maintenance expensive. This fragment codifies the production-proven patterns from the pactjs-utils reference implementation so that every new project starts correctly.

## Pattern Examples

### Example 1: Directory Structure & File Naming

**Context**: Consumer contract test project layout using pactjs-utils conventions.

**Implementation**:

```
tests/contract/
├── consumer/
│   ├── get-filter-fields.pacttest.ts    # Consumer test (one per endpoint group)
│   ├── filter-transactions.pacttest.ts
│   └── get-transaction-stats.pacttest.ts
└── support/
    ├── pact-config.ts                   # PactV4 factory (consumer/provider names, output dir)
    ├── provider-states.ts               # Provider state factory functions
    └── consumer-helpers.ts              # Local shim (until pactjs-utils is published)

scripts/
├── env-setup.sh                         # Shared env loader (sourced by all broker scripts)
├── publish-pact.sh                      # Publish pact files to broker
├── can-i-deploy.sh                      # Deployment safety check
└── record-deployment.sh                 # Record deployment after merge

.github/
├── actions/
│   └── detect-breaking-change/
│       └── action.yml                   # PR checkbox-driven breaking change detection
└── workflows/
    └── contract-test-consumer.yml       # Consumer CDC CI workflow
```

**Key Points**:

- Consumer tests use `.pacttest.ts` extension (not `.pact.spec.ts` or `.contract.ts`)
- Support files live in `tests/contract/support/`, not mixed with consumer tests
- Shell scripts live in `scripts/` at project root, not nested inside test directories
- CI workflow named `contract-test-consumer.yml` (not `pact-consumer.yml` or other variants)

---

### Example 2: Vitest Configuration for Pact

**Context**: Minimal vitest config dedicated to contract tests — do NOT copy settings from the project's main `vitest.config.ts`.

**Implementation**:

```typescript
// vitest.config.pact.ts
// See pact-consumer-framework-setup.md Example 2 "Key Points" for rationale on
// fileParallelism + pool:forks + singleFork. Do not remove those three settings.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/contract/**/*.pacttest.ts'],
    testTimeout: 30000,
    fileParallelism: false,
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
  },
});
```

**Key Points**:

- **`fileParallelism: false` is required** — primary defense against non-deterministic pact generation. Without it, parallel workers race on the shared pact JSON file and corrupt interactions. Symptom: local runs pass, CI randomly fails with `Cannot change pact content for already published pact`. See Example 10 for the determinism gate that enforces byte-stability across re-runs.
- **`pool: 'forks'` + `singleFork: true` is required for multi-file consumer suites** — same config the provider side uses (`pactjs-utils-provider-verifier.md` Example 7). Best current understanding: the `@pact-foundation/pact` napi-rs binding is not robust across Vitest worker threads sharing a process; with the default threads pool (Vitest v1) and multiple `.pacttest.ts` files on the same consumer+provider pair, we observed reproducible "request was expected but not received" flakes on Linux CI only. `singleFork: true` serializes every pact file into one forked subprocess and eliminated the flake on two repos (`pactjs-utils`, `seon-mcp-server`). Vitest v2+ defaults to `forks`, but set the pool explicitly so the contract does not drift with Vitest version bumps.
- **Single-file consumer suites** (one `.pacttest.ts` per consumer+provider pair) have not been observed to flake under default threads pool, because FFI state is not shared across files when there is only one file. Adding `pool: 'forks'` is still recommended — it future-proofs you the moment a second file is added — but a suite passing today with only `fileParallelism: false` is not broken.
- **Interacting settings**: leave `isolate` at its default (`true`). Do NOT set `sequence.concurrent: true`, `maxConcurrency > 1`, or `maxWorkers > 1` in this config — they defeat the serialization this rule relies on. `hookTimeout` may be raised if mock-server startup is slow, but keep `testTimeout` ≥ `hookTimeout`.
- Do NOT add `setupFiles`, `coverage`, or other settings from the unit test config
- Keep it minimal — Pact tests run in Node environment with extended timeout
- 30 second timeout accommodates Pact mock server startup and interaction verification
- Use a dedicated config file (`vitest.config.pact.ts`), not the main vitest config

---

### Example 3: Package.json Script Naming

**Context**: Colon-separated naming matching pactjs-utils exactly. Scripts source `env-setup.sh` inline.

**Implementation**:

```json
{
  "scripts": {
    "test:pact:consumer": "./scripts/check-pact-determinism.sh 'npm run test:pact:consumer:run' 3 ./pacts",
    "test:pact:consumer:run": "vitest run --config vitest.config.pact.ts",
    "publish:pact": ". ./scripts/env-setup.sh && ./scripts/publish-pact.sh",
    "can:i:deploy:consumer": ". ./scripts/env-setup.sh && PACTICIPANT=<service-name> ./scripts/can-i-deploy.sh",
    "record:consumer:deployment": ". ./scripts/env-setup.sh && PACTICIPANT=<service-name> ./scripts/record-deployment.sh"
  }
}
```

Replace `<service-name>` with the consumer's pacticipant name (e.g., `my-frontend-app`).

**Key Points**:

- **`test:pact:consumer` IS the determinism gate** — it runs the inner command 3× and fails if pact output is not byte-stable. This is the command CI and developers run before pushing. See Example 10 for the `check-pact-determinism.sh` script itself.
- **`test:pact:consumer:run` is the fast inner command** for TDD loops (a single pass of the suite, no gate). Developers can iterate with this; CI always goes through the outer gated script.
- Use colon-separated naming: `test:pact:consumer`, NOT `test:contract` or `test:contract:consumer`
- Broker scripts source `env-setup.sh` inline in package.json (`. ./scripts/env-setup.sh && ...`)
- `PACTICIPANT` is set per-script invocation, not globally
- Do NOT use `npx pact-broker` — use `pact-broker` directly (installed as a dependency)

---

### Example 4: Shell Scripts

**Context**: Reusable bash scripts aligned with pactjs-utils conventions.

#### `scripts/env-setup.sh` — Shared Environment Loader

```bash
#!/bin/bash
# -e: exit on error  -u: error on undefined vars (catches typos/missing env vars in CI)
set -eu

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse --short HEAD)}"
export GITHUB_BRANCH="${GITHUB_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
```

#### `scripts/publish-pact.sh` — Publish Pacts to Broker (with defense-in-depth normalization)

```bash
#!/bin/bash
# Publish generated pact files to PactFlow/Pact Broker.
#
# Before publish, normalize each pact JSON: sort interactions by (description, provider state name,
# method, path) and sort object keys via `jq -S`. This gives byte-stable output to the broker even
# if the PactV4 generator produces ordering drift between runs. Paired with scripts/check-pact-determinism.sh
# as defense-in-depth — the gate catches drift pre-publish; normalization ensures "Cannot change pact
# content" from PactFlow never fires on ordering-only changes that slip past the gate.
#
# Requires: PACT_BROKER_BASE_URL, PACT_BROKER_TOKEN, GITHUB_SHA, GITHUB_BRANCH, jq
# -e: exit on error  -u: error on undefined vars  -o pipefail: fail if any pipe segment fails
set -euo pipefail

. ./scripts/env-setup.sh

PACT_DIR="./pacts"

# Defense-in-depth: normalize interaction order for byte-stable publishes.
for f in "$PACT_DIR"/*.json; do
  tmp="$(mktemp)"
  jq -S '.interactions |= sort_by(.description, (.providerStates[0].name // ""), .request.method, .request.path)' \
     "$f" > "$tmp"
  mv "$tmp" "$f"
done

pact-broker publish "$PACT_DIR" \
    --consumer-app-version="$GITHUB_SHA" \
    --branch="$GITHUB_BRANCH" \
    --broker-base-url="$PACT_BROKER_BASE_URL" \
    --broker-token="$PACT_BROKER_TOKEN"
```

#### `scripts/can-i-deploy.sh` — Deployment Safety Check

```bash
#!/bin/bash
# Check if a pacticipant version can be safely deployed
#
# Requires: PACTICIPANT (set by caller), PACT_BROKER_BASE_URL, PACT_BROKER_TOKEN, GITHUB_SHA
# -e: exit on error  -u: error on undefined vars  -o pipefail: fail if any pipe segment fails
set -euo pipefail

. ./scripts/env-setup.sh

PACTICIPANT="${PACTICIPANT:?PACTICIPANT env var is required}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

pact-broker can-i-deploy \
    --pacticipant "$PACTICIPANT" \
    --version="$GITHUB_SHA" \
    --to-environment "$ENVIRONMENT" \
    --retry-while-unknown=10 \
    --retry-interval=30
```

#### `scripts/record-deployment.sh` — Record Deployment

```bash
#!/bin/bash
# Record a deployment to an environment in Pact Broker
# Only records on main/master branch (skips feature branches)
#
# Requires: PACTICIPANT, PACT_BROKER_BASE_URL, PACT_BROKER_TOKEN, GITHUB_SHA, GITHUB_BRANCH
# -e: exit on error  -u: error on undefined vars  -o pipefail: fail if any pipe segment fails
set -euo pipefail

. ./scripts/env-setup.sh

PACTICIPANT="${PACTICIPANT:?PACTICIPANT env var is required}"

if [ "$GITHUB_BRANCH" = "main" ] || [ "$GITHUB_BRANCH" = "master" ]; then
  pact-broker record-deployment \
      --pacticipant "$PACTICIPANT" \
      --version "$GITHUB_SHA" \
      --environment "${npm_config_env:-dev}"
else
  echo "Skipping record-deployment: not on main branch (current: $GITHUB_BRANCH)"
fi
```

**Key Points**:

- `env-setup.sh` uses `set -eu` (no pipefail — it only sources `.env`, no pipes); broker scripts use `set -euo pipefail`
- Use `pact-broker` directly, NOT `npx pact-broker`
- Use `PACTICIPANT` env var (required via `${PACTICIPANT:?...}`), not hardcoded service names
- `can-i-deploy` includes `--retry-while-unknown=10 --retry-interval=30` (waits for provider verification)
- `record-deployment` has branch guard (only records on main/master)
- **`publish-pact.sh` normalizes interactions with `jq -S` + `sort_by(...)` before publishing** — defense-in-depth alongside the determinism gate (Example 10). The gate catches drift; normalization ensures byte-stable payload to the broker regardless of generator quirks. Keep both; they protect against different failure modes.
- Do NOT invent custom env vars like `PACT_CONSUMER_VERSION` or `PACT_BREAKING_CHANGE` in scripts — those are handled by `env-setup.sh` and the CI detect-breaking-change action respectively

---

### Example 5: CI Workflow (`contract-test-consumer.yml`)

**Context**: GitHub Actions workflow for consumer CDC, matching pactjs-utils structure exactly.

**Implementation**:

```yaml
name: Contract Test - Consumer
on:
  pull_request:
    types: [opened, synchronize, reopened, edited]
  push:
    branches: [main]

env:
  PACT_BROKER_BASE_URL: ${{ secrets.PACT_BROKER_BASE_URL }}
  PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}
  GITHUB_SHA: ${{ github.sha }}
  GITHUB_BRANCH: ${{ github.head_ref || github.ref_name }}

concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true

jobs:
  consumer-contract-test:
    if: github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-node@v6
        with:
          node-version-file: '.nvmrc'
          cache: 'npm'

      - name: Detect Pact breaking change
        uses: ./.github/actions/detect-breaking-change

      - name: Install dependencies
        run: npm ci

      # (1) Generate pact files — runs the determinism gate (3 runs + byte-stable check via jq)
      - name: Consumer pact tests (determinism gate)
        run: npm run test:pact:consumer

      # (2) Publish pacts to broker (publish-pact.sh also normalizes interaction order as defense-in-depth)
      - name: Publish pacts to PactFlow
        run: npm run publish:pact

      # After publish, PactFlow fires a webhook that triggers
      # the provider's contract-test-provider.yml workflow.
      # can-i-deploy retries while waiting for provider verification.

      # (4) Check deployment safety (main only — on PRs, local verification is the gate)
      - name: Can I deploy consumer? (main only)
        if: github.ref == 'refs/heads/main' && env.PACT_BREAKING_CHANGE != 'true'
        run: npm run can:i:deploy:consumer

      # (5) Record deployment (main only)
      - name: Record consumer deployment (main only)
        if: github.ref == 'refs/heads/main'
        run: npm run record:consumer:deployment --env=dev
```

**Key Points**:

- **1:1 local/CI parity is a hard rule**: every CI step is `npm run <same-name-a-dev-uses>`. Never let CI invoke `vitest` or `pact-broker` directly — that divergence is how "works on my machine" slips in. The determinism gate, publish, can-i-deploy, and record-deployment are all the same commands a developer runs locally.
- **The determinism gate is its own visible step, not a side-effect of publish.** A failing gate must be debuggable from the CI log without re-running. Do not fold it into a `prepublish:pact` hook — folding hides the failure inside a publish log and makes attribution harder.
- **Workflow-level `env` block** for broker secrets and git vars — not per-step
- **`detect-breaking-change` step** runs before install to set `PACT_BREAKING_CHANGE` env var
- **Step numbering skips (3)** — step 3 is the webhook-triggered provider verification (happens externally)
- **can-i-deploy condition**: `github.ref == 'refs/heads/main' && env.PACT_BREAKING_CHANGE != 'true'`
- **Comment on (4)**: "on PRs, local verification is the gate"
- **No upload-artifact step** — the broker is the source of truth for pact files
- **`dependabot[bot]` skip** on the job (contract tests don't run for dependency updates)
- **PR types include `edited`** — needed for breaking change checkbox detection in PR body
- **`GITHUB_BRANCH`** uses `${{ github.head_ref || github.ref_name }}` — `head_ref` for PRs, `ref_name` for pushes

---

### Example 6: Detect Breaking Change Composite Action

**Context**: GitHub composite action that reads a `[x] Pact breaking change` checkbox from the PR body.

**Implementation**:

Create `.github/actions/detect-breaking-change/action.yml`:

```yaml
name: 'Detect Pact Breaking Change'
description: 'Reads the PR template checkbox to determine if this change is a Pact breaking change. Sets PACT_BREAKING_CHANGE env var.'

outputs:
  is_breaking_change:
    description: 'Whether the change is a breaking change (true/false)'
    value: ${{ steps.result.outputs.is_breaking_change }}

runs:
  using: 'composite'
  steps:
    # PR event path: read checkbox directly from current PR body.
    - name: Set PACT_BREAKING_CHANGE from PR description (PR only)
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const prBody = context.payload.pull_request.body || '';
          const breakingChangePattern = /\[\s*[xX]\s*\]\s*Pact breaking change/i;
          const isBreakingChange = breakingChangePattern.test(prBody);
          core.exportVariable('PACT_BREAKING_CHANGE', isBreakingChange ? 'true' : 'false');
          console.log(`PACT_BREAKING_CHANGE=${isBreakingChange ? 'true' : 'false'} (from PR description checkbox).`);

    # Push-to-main path: resolve the merged PR and read the same checkbox.
    - name: Set PACT_BREAKING_CHANGE from merged PR (push to main)
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      uses: actions/github-script@v7
      with:
        script: |
          const { data: prs } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
            owner: context.repo.owner,
            repo: context.repo.repo,
            commit_sha: context.sha,
          });
          const merged = prs.find(pr => pr.merged_at);
          const mergedBody = merged?.body || '';
          const breakingChangePattern = /\[\s*[xX]\s*\]\s*Pact breaking change/i;
          const isBreakingChange = breakingChangePattern.test(mergedBody);
          core.exportVariable('PACT_BREAKING_CHANGE', isBreakingChange ? 'true' : 'false');
          console.log(`PACT_BREAKING_CHANGE=${isBreakingChange ? 'true' : 'false'} (from merged PR lookup).`);

    - name: Export result
      id: result
      shell: bash
      run: echo "is_breaking_change=${PACT_BREAKING_CHANGE:-false}" >> "$GITHUB_OUTPUT"
```

**Key Points**:

- Two separate conditional steps (better CI log readability than single if/else)
- PR path: reads checkbox directly from PR body
- Push-to-main path: resolves merged PR via GitHub API, reads same checkbox
- Exports `PACT_BREAKING_CHANGE` env var for downstream steps
- `outputs.is_breaking_change` available for consuming workflows
- Uses a case-insensitive checkbox regex (`/\[\s*[xX]\s*\]\s*Pact breaking change/i`) to detect checked states robustly

---

### Example 7: Consumer Test Using PactV4 Builder

**Context**: Consumer pact test using PactV4 `addInteraction()` builder pattern. The test MUST call **real consumer code** (your actual API client/service functions) against the mock server — not raw `fetch()`. Using `fetch()` directly defeats the purpose of CDC testing because it doesn't verify your actual consumer code works with the contract.

**Implementation**:

The consumer code must expose a way to inject the base URL (e.g., `setApiUrl()`, constructor parameter, or environment variable). This is a prerequisite for contract testing.

```typescript
// src/api/movie-client.ts — The REAL consumer code (already exists in your project)
import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: process.env.API_URL || 'http://localhost:3001',
});

// Expose a way to override the base URL for Pact testing
export const setApiUrl = (url: string) => {
  axiosInstance.defaults.baseURL = url;
};

export const getMovies = async () => {
  const res = await axiosInstance.get('/movies');
  return res.data;
};

export const getMovieById = async (id: number) => {
  const res = await axiosInstance.get(`/movies/${id}`);
  return res.data;
};
```

```typescript
// tests/contract/consumer/get-movies.pacttest.ts
import { MatchersV3 } from '@pact-foundation/pact';
import type { V3MockServer } from '@pact-foundation/pact';
import { createProviderState, setJsonBody, setJsonContent } from '../support/consumer-helpers';
import { movieExists } from '../support/provider-states';
import { createPact } from '../support/pact-config';
// Import REAL consumer code — this is what we're actually testing
import { getMovies, getMovieById, setApiUrl } from '../../../src/api/movie-client';

const { like, integer, string } = MatchersV3;

const pact = createPact();

describe('Movies API Consumer Contract', () => {
  const movieWithId = { id: 1, name: 'The Matrix', year: 1999, rating: 8.7, director: 'Wachowskis' };

  it('should get a movie by ID', async () => {
    const [stateName, stateParams] = createProviderState(movieExists(movieWithId));

    await pact
      .addInteraction()
      .given(stateName, stateParams)
      .uponReceiving('a request to get movie by ID')
      .withRequest(
        'GET',
        '/movies/1',
        setJsonContent({
          headers: { Accept: 'application/json' },
        }),
      )
      .willRespondWith(
        200,
        setJsonBody(
          like({
            id: integer(1),
            name: string('The Matrix'),
            year: integer(1999),
            rating: like(8.7),
            director: string('Wachowskis'),
          }),
        ),
      )
      .executeTest(async (mockServer: V3MockServer) => {
        // Inject mock server URL into the REAL consumer code
        setApiUrl(mockServer.url);

        // Call the REAL consumer function — this is what CDC testing validates
        const movie = await getMovieById(1);

        expect(movie.id).toBe(1);
        expect(movie.name).toBe('The Matrix');
      });
  });

  it('should handle movie not found', async () => {
    await pact
      .addInteraction()
      .given('No movies exist')
      .uponReceiving('a request for a non-existent movie')
      .withRequest('GET', '/movies/999')
      .willRespondWith(404, setJsonBody({ error: 'Movie not found' }))
      .executeTest(async (mockServer: V3MockServer) => {
        setApiUrl(mockServer.url);

        await expect(getMovieById(999)).rejects.toThrow();
      });
  });
});
```

**Key Points**:

- **CRITICAL**: Always test your REAL consumer code — import and call actual API client functions, never raw `fetch()`
- Using `fetch()` directly only tests that Pact's mock server works, which is meaningless
- Consumer code MUST expose a URL injection mechanism: `setApiUrl()`, env var override, or constructor parameter
- If the consumer code doesn't support URL injection, add it — this is a design prerequisite for CDC testing
- Use PactV4 `addInteraction()` builder (not PactV3 fluent API with `withRequest({...})` object)
- **Interaction naming convention**: Use the pattern `"a request to <action> <resource> [<condition>]"` for `uponReceiving()`. Examples: `"a request to get a movie by ID"`, `"a request to delete a non-existing movie"`, `"a request to create a movie that already exists"`. These names appear in Pact Broker UI and verification logs — keep them descriptive and unique within the consumer-provider pair.
- Use `setJsonContent` for request/response builder callbacks with query/header/body concerns; use `setJsonBody` for body-only response callbacks
- Provider state factory functions (`movieExists`) return `ProviderStateInput` objects
- `createProviderState` converts to `[stateName, stateParams]` tuple for `.given()`

**Common URL injection patterns** (pick whichever fits your consumer architecture):

| Pattern              | Example                                      | Best For              |
| -------------------- | -------------------------------------------- | --------------------- |
| `setApiUrl(url)`     | Mutates axios instance `baseURL`             | Singleton HTTP client |
| Constructor param    | `new ApiClient({ baseUrl: mockServer.url })` | Class-based clients   |
| Environment variable | `process.env.API_URL = mockServer.url`       | Config-driven apps    |
| Factory function     | `createApi({ baseUrl: mockServer.url })`     | Functional patterns   |

---

### Example 8: Support Files

#### Pact Config Factory

```typescript
// tests/contract/support/pact-config.ts
import path from 'node:path';
import { PactV4 } from '@pact-foundation/pact';

export const createPact = (overrides?: { consumer?: string; provider?: string }) =>
  new PactV4({
    dir: path.resolve(process.cwd(), 'pacts'),
    consumer: overrides?.consumer ?? 'MyConsumerApp',
    provider: overrides?.provider ?? 'MyProviderAPI',
    logLevel: 'warn',
  });
```

#### Provider State Factories

```typescript
// tests/contract/support/provider-states.ts
import type { ProviderStateInput } from './consumer-helpers';

export const movieExists = (movie: { id: number; name: string; year: number; rating: number; director: string }): ProviderStateInput => ({
  name: 'An existing movie exists',
  params: movie,
});

export const hasMovieWithId = (id: number): ProviderStateInput => ({
  name: 'Has a movie with a specific ID',
  params: { id },
});
```

#### Local Consumer Helpers Shim

```typescript
// tests/contract/support/consumer-helpers.ts
// TODO(temporary scaffolding): Replace local TemplateHeaders/TemplateQuery types
// with '@seontechnologies/pactjs-utils' exports when available.

type TemplateHeaders = Record<string, string | number | boolean>;
type TemplateQueryValue = string | number | boolean | Array<string | number | boolean>;
type TemplateQuery = Record<string, TemplateQueryValue>;

export type ProviderStateInput = {
  name: string;
  params: Record<string, unknown>;
};

type JsonMap = { [key: string]: boolean | number | string | null | JsonMap | Array<unknown> };
type JsonContentBuilder = {
  headers: (headers: TemplateHeaders) => unknown;
  jsonBody: (body: unknown) => unknown;
  query?: (query: TemplateQuery) => unknown;
};

export type JsonContentInput = {
  body?: unknown;
  headers?: TemplateHeaders;
  query?: TemplateQuery;
};

export const toJsonMap = (obj: Record<string, unknown>): JsonMap =>
  Object.fromEntries(
    Object.entries(obj).map(([key, value]) => {
      if (value === null || value === undefined) return [key, 'null'];
      if (typeof value === 'object' && !(value instanceof Date) && !Array.isArray(value)) return [key, JSON.stringify(value)];
      if (typeof value === 'number' || typeof value === 'boolean') return [key, value];
      if (value instanceof Date) return [key, value.toISOString()];
      return [key, String(value)];
    }),
  );

export const createProviderState = ({ name, params }: ProviderStateInput): [string, JsonMap] => [name, toJsonMap(params)];

export const setJsonContent =
  ({ body, headers, query }: JsonContentInput) =>
  (builder: JsonContentBuilder): void => {
    if (query && builder.query) {
      builder.query(query);
    }

    if (headers) {
      builder.headers(headers);
    }

    if (body !== undefined) {
      builder.jsonBody(body);
    }
  };

export const setJsonBody = (body: unknown) => setJsonContent({ body });
```

**Key Points**:

- If `@seontechnologies/pactjs-utils` is not yet installed, create a local shim that mirrors the API
- Add a TODO comment noting to swap for the published package when available
- The shim exports `createProviderState`, `toJsonMap`, `setJsonContent`, `setJsonBody`, and helper input types
- Keep shim types local (or sourced from public exports only); do not import from internal Pact paths like `@pact-foundation/pact/src/*`

---

### Example 9: .gitignore Entries

**Context**: Pact-specific entries to add to `.gitignore`.

```
# Pact contract testing artifacts
/pacts/
pact-logs/
```

---

### Example 10: Determinism Gate Script (Primary Defense)

**Context**: Even with `fileParallelism: false` (Example 2) and one-interaction-per-`it()` (see `pactjs-utils-consumer-helpers.md`), the PactV4 Rust FFI layer can occasionally produce byte-different pact JSON between runs — interaction ordering drift, nested matcher serialization quirks, or `Date` / random-value matchers that weren't locked down. This causes PactFlow to reject re-publishes of the same consumer SHA with `Cannot change pact content for already published pact`. The determinism gate runs the consumer suite N times locally and in CI, hashes the normalized pact files, and fails fast if drift is detected — before any publish is attempted.

**Implementation**:

#### `scripts/check-pact-determinism.sh`

```bash
#!/bin/bash
# Run a pact consumer command N times and fail if the generated pact files are not byte-stable.
# Primary defense against PactV4 non-deterministic output.
#
# Usage:  ./scripts/check-pact-determinism.sh "<cmd>" [runs] [pact-dir]
# Example: ./scripts/check-pact-determinism.sh 'npm run test:pact:consumer:run' 3 ./pacts
#
# Requires: jq installed on the runner (ubuntu-latest has it; macOS users need `brew install jq`).
set -euo pipefail

CMD="${1:?usage: ./scripts/check-pact-determinism.sh \"<cmd>\" [runs] [pact-dir]}"
RUNS="${PACT_DETERMINISM_RUNS:-${2:-3}}"
PACT_DIR="${3:-./pacts}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

hash_pact_file() {
  # Sort interactions by (description, first provider state name, method, path), sort keys with -S.
  # The sorted output is what we hash — so ordering-only drift does NOT count as non-determinism here.
  # (The gate catches deeper drift; ordering drift is handled by publish-pact.sh normalization.)
  jq -S '.interactions |= sort_by(.description, (.providerStates[0].name // ""), .request.method, .request.path)' "$1" \
    | shasum -a 256 | awk '{print $1}'
}

for run in $(seq 1 "$RUNS"); do
  echo "→ determinism run $run/$RUNS"
  rm -f "$PACT_DIR"/*.json 2>/dev/null || true
  eval "$CMD" >"$TMP_DIR/run-$run.log" 2>&1 || {
    echo "❌ run $run failed — dumping log:"
    cat "$TMP_DIR/run-$run.log"
    exit 1
  }
  : > "$TMP_DIR/run-$run.hashes"
  for f in "$PACT_DIR"/*.json; do
    [ -f "$f" ] || continue
    printf '%s  %s\n' "$(hash_pact_file "$f")" "$(basename "$f")" >> "$TMP_DIR/run-$run.hashes"
  done
  sort -o "$TMP_DIR/run-$run.hashes" "$TMP_DIR/run-$run.hashes"
done

# Compare every subsequent run against run 1.
FAIL=0
for run in $(seq 2 "$RUNS"); do
  if ! diff -q "$TMP_DIR/run-1.hashes" "$TMP_DIR/run-$run.hashes" >/dev/null; then
    FAIL=1
    echo ""
    echo "❌ Pact output differs between run 1 and run $run:"
    diff "$TMP_DIR/run-1.hashes" "$TMP_DIR/run-$run.hashes" || true
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "Pact output is non-deterministic across $RUNS runs. Likely causes:"
  echo "  • multiple .addInteraction() chained in a single it() block (PactV4 FFI drops one non-deterministically)"
  echo "  • fileParallelism: true in vitest.config.pact.ts (workers race on shared pact JSON)"
  echo "  • missing pool: 'forks' + singleFork: true (threads pool shares FFI state across files on Linux CI)"
  echo "  • Date / random matchers that don't lock a stable example value"
  echo "  • provider state params mutating between runs (e.g. Date.now())"
  exit 1
fi

echo "✅ Pact output is byte-stable across $RUNS runs."
```

**Key Points**:

- **Wire this script into `test:pact:consumer`** (see Example 3). The outer script IS the gate; the inner `test:pact:consumer:run` is the single-pass command for TDD loops.
- **Default 3 runs** is the sweet spot — 2 runs miss intermittent drops, >3 slows CI without catching more. Override with an env var or the positional arg if you're actively debugging a flake.
- **Treat gate failures as a P0 bug, not a "retry until green" condition.** Find the source of non-determinism (chained `addInteraction`, unsorted interactions, Date-dependent matchers). Do not raise `RUNS` to 10 to mask the symptom.
- **Requires `jq`** — installed by default on `ubuntu-latest`. For macOS local dev, document `brew install jq` in the project README.
- **In CI, make this its own visible step** (see Example 5 step (1) naming). Do not fold into a `prepublish:pact` hook — that hides the failure inside a publish log.
- **Defense-in-depth with `publish-pact.sh` normalization** (Example 4): the gate catches pre-publish drift; the publish-time `jq` sort ensures any ordering-only drift that slipped past the gate still produces a byte-stable payload to PactFlow.

---

## Validation Checklist

Before presenting the consumer CDC framework to the user, verify:

- [ ] `vitest.config.pact.ts` is minimal **and sets `fileParallelism: false` AND `pool: 'forks'` with `poolOptions.forks.singleFork: true`** (`fileParallelism: false` prevents shared pact JSON corruption from parallel workers; forks + `singleFork: true` eliminates the Linux-CI "request was expected but not received" flake observed once a second `.pacttest.ts` is added — see Example 2 Key Points for evidence, mechanism qualifier, and single-file exception)
- [ ] `vitest.config.pact.ts` does NOT set `sequence.concurrent: true`, `maxConcurrency > 1`, `maxWorkers > 1`, or `isolate: false` — all four defeat the serialization the rule relies on
- [ ] `package.json` splits `test:pact:consumer` (gated determinism runner) and `test:pact:consumer:run` (inner single-pass command)
- [ ] `scripts/check-pact-determinism.sh` is present, hashes via `jq -S` + `sort_by`, defaults to 3 runs, and is the body of the `test:pact:consumer` script
- [ ] `scripts/publish-pact.sh` normalizes interactions with `jq -S '.interactions |= sort_by(.description, (.providerStates[0].name // ""), .request.method, .request.path)'` before the `pact-broker publish` call (defense-in-depth alongside the gate)
- [ ] Script names match pactjs-utils (`test:pact:consumer`, `test:pact:consumer:run`, `publish:pact`, `can:i:deploy:consumer`, `record:consumer:deployment`)
- [ ] Scripts source `env-setup.sh` inline in package.json
- [ ] Shell scripts use `pact-broker` not `npx pact-broker`
- [ ] Shell scripts use `PACTICIPANT` env var pattern
- [ ] `can-i-deploy.sh` has `--retry-while-unknown=10 --retry-interval=30`
- [ ] `record-deployment.sh` has branch guard
- [ ] `env-setup.sh` uses `set -eu`; broker scripts use `set -euo pipefail` — each with explanatory comment
- [ ] CI workflow named `contract-test-consumer.yml`
- [ ] CI has workflow-level env block (not per-step)
- [ ] CI has `detect-breaking-change` step before install
- [ ] CI step (1) is the determinism gate (calls `npm run test:pact:consumer`) — its own visible step, not folded into publish
- [ ] CI steps are 1:1 with developer commands — every CI step calls `npm run <same-name>` a dev would run locally (no direct `vitest` or `pact-broker` invocation)
- [ ] CI step numbering skips (3) — webhook-triggered provider verification
- [ ] CI can-i-deploy has `PACT_BREAKING_CHANGE != 'true'` condition
- [ ] CI has NO upload-artifact step
- [ ] `.github/actions/detect-breaking-change/action.yml` exists
- [ ] Consumer tests use `.pacttest.ts` extension
- [ ] Consumer tests use PactV4 `addInteraction()` builder
- [ ] `uponReceiving()` names follow `"a request to <action> <resource> [<condition>]"` pattern and are unique within the consumer-provider pair
- [ ] Interaction callbacks use `setJsonContent` for query/header/body and `setJsonBody` for body-only responses
- [ ] Request bodies use exact values (no `like()` wrapper) — Postel's Law: be strict in what you send
- [ ] `like()`, `eachLike()`, `string()`, `integer()` matchers are only used in `willRespondWith` (responses), not in `withRequest` (requests) — matchers check type/shape, not exact values
- [ ] Consumer tests call REAL consumer code (actual API client functions), NOT raw `fetch()`
- [ ] Consumer code exposes URL injection mechanism (`setApiUrl()`, env var, or constructor param)
- [ ] Local consumer-helpers shim present if pactjs-utils not installed
- [ ] `.gitignore` includes `/pacts/` and `pact-logs/`

## Related Fragments

- `pactjs-utils-overview.md` — Library decision tree and installation
- `pactjs-utils-consumer-helpers.md` — `createProviderState`, `toJsonMap`, `setJsonContent`, `setJsonBody`, **one-interaction-per-`it()` rule**
- `pactjs-utils-provider-verifier.md` — Provider-side verification patterns; consumer and provider BOTH require `pool: 'forks'` + `singleFork: true` — same FFI-safety rule applies on both sides
- `pactjs-utils-request-filter.md` — Auth injection for provider verification
- `pact-broker-webhooks.md` — PactFlow → GitHub webhook auth pattern (dedicated user, classic PAT, PactFlow secret) and staleness monitoring
- `contract-testing.md` — Foundational CDC patterns and resilience coverage
