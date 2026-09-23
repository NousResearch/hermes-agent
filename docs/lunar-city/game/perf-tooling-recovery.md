# Lunar City performance tooling recovery — 2026-09-07

Recovered missing tooling from historical Git commit `0d9a879ce487cd6886c742ab06828547882c708c`, inspected before copying. Existing current files were preserved, including the byte-identical `scripts/perf/lib/lunar-city-provenance.mjs`. This is source/tooling validation, not packaged performance acceptance or a supervised-live receipt.

## Restored and reconciled

- `apps/desktop/scripts/perf/lunar-city{,-runner,-orchestrator}.mjs` and their three self-test files: raw sampling, receipt validation, and canonical orchestration.
- `apps/desktop/e2e/lunar-city-fixtures.ts` and fixture, packaged, and development specs: immutable population contract, exact-source collision checks, real fixture helpers, and launch assertions.
- `apps/desktop/electron/lunar-city-perf-{main,preload}.ts` and two test files: packaged nonce/build/lifetime-bound IPC, Chromium memory-infra GPU allocation, renderer request routing, and preload capability handshake.
- New topical `lunar-city-perf-install.ts` and `lunar-city-perf-preload-install.ts` integrate those controllers with current Electron APIs. Main entry changes are limited to import, construction, window attachment, and quit disposal. Preload only imports/calls its installer. Installer test exercises owned-sender bootstrap, activation, navigation revocation, and listener disposal through actual installed IPC callbacks.
- Package scripts: `perf:lunar-city` (runner), `perf:lunar-city:accept` (orchestrator), and `perf:lunar-city:validate` (validator).

The current mock backend has no `mockDispatcherReadiness` option; the recovered development spec now uses its current API. Removed the obsolete Python facade monkeypatch. Fixture seeding imports `hermes_cli.kanban_db_connect.connect` directly. Future fixture gateway launches require the authorized `~/.local/bin/hermes` launcher and a successful `--version-local` check; seeding requires this checkout's `.venv/bin/python`. There is no legacy Hermes checkout executable fallback. Added nonlaunching `--help` to the raw runner and validator.

The Electron adapter uses actual `app.isPackaged`, never the separate development compatibility override. It preserves clean exact-SHA build stamps, unique launch nonce, owned renderer identity and lifetime, invalidation on navigation/crash/destruction, real window transitions, and native GPU memory provenance. Unknown/unavailable measurements stay unavailable. Cleanup removes this adapter's IPC/window listeners. GPU capture remains serialized.

## Verification performed

From repository root:

```sh
node --test apps/desktop/scripts/perf/lunar-city.test.mjs apps/desktop/scripts/perf/lunar-city-runner.test.mjs apps/desktop/scripts/perf/lunar-city-orchestrator.test.mjs
node apps/desktop/scripts/perf/lunar-city.mjs --help
node apps/desktop/scripts/perf/lunar-city-runner.mjs --help
node apps/desktop/scripts/perf/lunar-city-orchestrator.mjs --help
node apps/desktop/scripts/perf/lunar-city-orchestrator.mjs --sha invalid --scenario visible-idle --output /private/tmp/lunar-perf-refused-dry-run
```

58 Node self-tests passed, zero skipped. All help paths return without launching. The invalid-SHA dry run exits 1 with `REFUSED: expected git SHA must be exact`, before creating a run directory or launching.

From `apps/desktop`:

```sh
../../node_modules/.bin/vitest run --project electron electron/lunar-city-perf-main.test.ts electron/lunar-city-perf-preload.test.ts electron/lunar-city-perf-install.test.ts
../../node_modules/.bin/tsc -p tsconfig.electron.json --noEmit --incremental false --composite false
../../node_modules/.bin/tsc -p tsconfig.e2e.json --noEmit
../../node_modules/.bin/playwright test e2e/lunar-city-fixtures.spec.ts --grep 'deterministic immutable contract|retains exact-source collisions|byte parity with production|GPU packaged path rejects|GPU packaged launch never' --workers=1 --reporter=list
```

22 Electron tests and seven pure fixture/launch-contract tests passed. Electron and E2E TypeScript checks passed. The restored Playwright specs discover 18 tests; discovery is not execution. Scoped ESLint passed after formatting corrections. Both current Electron entrypoints bundle successfully in memory with production esbuild options (no `dist` replacement and no application launch).

## Remaining acceptance boundary

The default orchestrator deliberately refuses owned authenticated fixture spawning, process inspection, and population probing: its `spawnGateway`, `inspectProcess`, and `probePopulation` boundaries remain unavailable. Completing canonical acceptance requires implementing those lifecycle adapters against three isolated authenticated gateway homes, retaining exact process start identity, verified population/collision contracts, authenticated subagent lifecycle observations, and teardown of only owned processes. Injected test dependencies are not production implementation or live proof. Supervised-live orchestration also remains explicitly unsupported until a supported real-gateway preseed API exists.

After that lifecycle work, acceptance still requires a clean exact-commit packaged artifact with matching stamp, nonce-bound launched app/process identity, actual GPU/window measurements, canonical population and scenario repetitions, and validator-approved receipts. None of those claims follows from the source tests above. No shared Hermes state was modified and no gateway/app was launched or terminated during this recovery.
