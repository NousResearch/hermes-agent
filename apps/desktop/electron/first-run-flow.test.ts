/**
 * Integration tests for the desktop first-run install-vs-connect boot flow.
 *
 * Run with: vitest run --project electron electron/first-run-flow.test.ts
 * (Auto-discovered by the electron vitest project; runs in npm test:desktop:platforms.)
 *
 * first-run-gate.test.ts covers the pure latch in isolation. THIS file covers the
 * PRODUCTION integration that main.ts wires around it: the real first-run-flow.ts
 * helpers (guardFirstRunBootstrap / isFirstRunAborted / applyFirstRunChoice /
 * readFirstRunInstallChoice), the real createFirstRunGate latch, the real
 * modeIsRemoteLike classifier, and real on-disk marker/config files (temp dirs).
 *
 * Each test drives a bootOnce()/boot() harness that mirrors startHermes's
 * contract faithfully (resolve backend from the saved config → if nothing is
 * installed, guardFirstRunBootstrap → run the bootstrap installer; the catch
 * re-runs on a firstRunAborted error, modeling
 * `backendConnectionState.clearPromiseForAttempt(connectionAttempt);
 * return startHermes()`). The bottom section pins main.ts's wiring with the
 * repo's source-assertion pattern (see windows-hermes-resolution.test.ts).
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { modeIsRemoteLike } from './connection-config'
import {
  applyFirstRunChoice,
  guardFirstRunBootstrap,
  isFirstRunAborted,
  readFirstRunInstallChoice
} from './first-run-flow'
import { createFirstRunGate } from './first-run-gate'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Let queued microtasks (the parked boot's await chain) settle between steps.
const flush = () => new Promise(resolve => setImmediate(resolve))

// A throwaway temp workspace holding the on-disk marker + connection config, so
// the flow reads/writes REAL files exactly like main.ts's fs-backed injections.
function makeWorkspace() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-first-run-'))
  const markerPath = path.join(dir, 'first-run.json')
  const configPath = path.join(dir, 'connection.json')

  // Mirrors readDesktopConnectionConfig's tolerance: missing/malformed → local.
  const readConnectionConfig = () => {
    try {
      const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'))

      if (parsed && typeof parsed === 'object') {
        return parsed
      }
    } catch {
      // fall through to the local default
    }

    return { mode: 'local', remote: {}, profiles: {} }
  }

  return {
    dir,
    configPath,
    markerPath,
    readConnectionConfig,
    writeConnectionConfig: (config: any) => fs.writeFileSync(configPath, JSON.stringify(config, null, 2)),
    writeInstallChoice: () => fs.writeFileSync(markerPath, JSON.stringify({ choice: 'install' }, null, 2)),
    // Injected exactly as firstRunInstallChosen() does in main.ts.
    installChosen: () => readFirstRunInstallChoice(() => fs.readFileSync(markerPath, 'utf8')),
    cleanup: () => fs.rmSync(dir, { recursive: true, force: true })
  }
}

// Faithful model of startHermes's single boot attempt: resolve a backend from the
// saved connection config; a remote-like mode resolves a REMOTE backend without
// ever touching the local install tree, otherwise it's bootstrap-needed and we
// park on the gate (guardFirstRunBootstrap) BEFORE running the installer.
async function bootOnce({
  gate,
  readConnectionConfig,
  installChosen,
  runBootstrap
}: {
  gate: ReturnType<typeof createFirstRunGate>
  readConnectionConfig: () => any
  installChosen: () => boolean
  runBootstrap: () => void
}) {
  const config = readConnectionConfig()

  if (modeIsRemoteLike(config.mode)) {
    return { mode: config.mode, source: 'remote', url: config.remote?.url }
  }

  // Only the primary boot enables the gate here (enabled: true), matching
  // startHermes's ensureRuntime(..., { firstRunGate: true }).
  await guardFirstRunBootstrap({ gate, enabled: true, installChosen })
  runBootstrap()

  return { mode: 'local', source: 'local' }
}

// Faithful model of startHermes's `.catch()`: a firstRunAborted error re-runs the
// boot (main.ts clears the cached attempt and returns startHermes()); any other
// error propagates (the latching failure path).
async function boot(deps: Parameters<typeof bootOnce>[0]) {
  try {
    return await bootOnce(deps)
  } catch (error) {
    if (isFirstRunAborted(error)) {
      return boot(deps)
    }

    throw error
  }
}

// ---------------------------------------------------------------------------
// (a) Install transition, end to end.
// ---------------------------------------------------------------------------

test('install: parked boot resolves on choose, runs bootstrap once, and the marker short-circuits the next launch', async () => {
  const ws = makeWorkspace()

  try {
    const events: Array<{ required: boolean }> = []
    const gate = createFirstRunGate(state => events.push(state))
    let bootstrapRuns = 0

    const runBootstrap = () => {
      bootstrapRuns += 1
    }

    // Primary boot on a fresh machine parks on the gate.
    const booting = boot({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    })

    await flush()

    // The overlay is required, onChanged fired required:true, and a late renderer
    // calling hermes:first-run:get (→ gate.state()) is seeded with that state.
    assert.deepEqual(gate.state(), { required: true })
    assert.deepEqual(events, [{ required: true }])
    assert.equal(bootstrapRuns, 0, 'bootstrap must not start until the user chooses install')

    // hermes:first-run:choose 'install'.
    const stateAfter = applyFirstRunChoice({
      choice: 'install',
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      writeConnectionConfig: ws.writeConnectionConfig,
      writeInstallChoice: ws.writeInstallChoice,
      modeIsRemoteLike
    })

    assert.deepEqual(stateAfter, { required: false })
    assert.deepEqual(await booting, { mode: 'local', source: 'local' })
    assert.equal(bootstrapRuns, 1, 'the boot must proceed to run bootstrap exactly once')
    assert.deepEqual(events, [{ required: true }, { required: false }])

    // The marker was persisted, so firstRunInstallChosen() is now true.
    assert.ok(ws.installChosen(), 'install choice must be persisted to the marker file')

    // A SECOND launch (persisted marker) must NOT park — interrupted-install resume.
    const relaunchEvents = events.length

    const relaunch = await boot({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    })

    assert.deepEqual(relaunch, { mode: 'local', source: 'local' })
    assert.equal(bootstrapRuns, 2, 'the relaunch resumes bootstrap directly')
    assert.deepEqual(gate.state(), { required: false })
    assert.equal(events.length, relaunchEvents, 'the relaunch must not re-enter the wait (no new onChanged)')
  } finally {
    ws.cleanup()
  }
})

// ---------------------------------------------------------------------------
// (b) Install choice rewrites a half-configured remote back to local.
// ---------------------------------------------------------------------------

test('install: a saved remote-like config is rewritten to local with the remote block/profiles preserved', () => {
  const ws = makeWorkspace()

  try {
    // The remote form persisted mode:'remote' (e.g. an OAuth sign-in that saves
    // before Connect) while the user was still deciding.
    const savedRemote = {
      mode: 'remote',
      remote: { url: 'https://gateway.example', authMode: 'token', token: { encoding: 'plain', value: 'secret-abc' } },
      profiles: { work: { mode: 'remote', url: 'https://work.example' } }
    }

    ws.writeConnectionConfig(savedRemote)

    const gate = createFirstRunGate()

    applyFirstRunChoice({
      choice: 'install',
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      writeConnectionConfig: ws.writeConnectionConfig,
      writeInstallChoice: ws.writeInstallChoice,
      modeIsRemoteLike
    })

    const rewritten = ws.readConnectionConfig()
    assert.equal(rewritten.mode, 'local', 'a remote-like mode must be rewritten to local on install')
    assert.deepEqual(rewritten.remote, savedRemote.remote, 'the remote block must be preserved for later reuse')
    assert.deepEqual(rewritten.profiles, savedRemote.profiles, 'per-profile overrides must be preserved')
  } finally {
    ws.cleanup()
  }
})

// ---------------------------------------------------------------------------
// (c) Remote-apply transition, end to end.
// ---------------------------------------------------------------------------

test('remote-apply: an aborted parked boot re-runs against the freshly-saved remote without parking or bootstrapping', async () => {
  const ws = makeWorkspace()

  try {
    const events: Array<{ required: boolean }> = []
    const gate = createFirstRunGate(state => events.push(state))
    let bootstrapRuns = 0

    const runBootstrap = () => {
      bootstrapRuns += 1
    }

    const booting = boot({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    })

    await flush()
    assert.deepEqual(gate.state(), { required: true })

    // Model connection-config:apply's ordering: write the new remote config to
    // disk FIRST, then teardown (gate.abort()).
    ws.writeConnectionConfig({
      mode: 'remote',
      remote: { url: 'https://gateway.example', authMode: 'token' },
      profiles: {}
    })
    gate.abort()

    // The parked boot rejected with a firstRunAborted error; its retry read the
    // freshly-saved remote config and resolved a remote backend.
    const result = await booting
    assert.deepEqual(result, { mode: 'remote', source: 'remote', url: 'https://gateway.example' })
    assert.equal(bootstrapRuns, 0, 'the local install tree must never be touched on the remote-apply path')
    // The retry read remote and returned; it never re-parked the gate.
    assert.deepEqual(events, [{ required: true }, { required: false }])
    assert.deepEqual(gate.state(), { required: false })
  } finally {
    ws.cleanup()
  }
})

// ---------------------------------------------------------------------------
// (d) Bounded retry / re-park correctness.
// ---------------------------------------------------------------------------

test('remote-apply: if the aborted boot finds no remote config, the retry re-parks the gate instead of bootstrapping', async () => {
  const ws = makeWorkspace()

  try {
    const events: Array<{ required: boolean }> = []
    const gate = createFirstRunGate(state => events.push(state))
    let bootstrapRuns = 0

    const runBootstrap = () => {
      bootstrapRuns += 1
    }

    const booting = boot({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    })

    await flush()
    assert.deepEqual(gate.state(), { required: true })

    // Teardown-abort, but the saved config is NOT remote (still local default).
    gate.abort()
    await flush()

    // The retry re-parked the gate (the overlay re-appears) rather than looping
    // or bootstrapping — the user must still choose.
    assert.deepEqual(gate.state(), { required: true })
    assert.equal(bootstrapRuns, 0, 'a non-remote re-run must not bootstrap on its own')
    assert.deepEqual(events, [{ required: true }, { required: false }, { required: true }])

    // Choosing install now resolves the re-parked boot exactly once.
    gate.chooseInstall()
    assert.deepEqual(await booting, { mode: 'local', source: 'local' })
    assert.equal(bootstrapRuns, 1)
    assert.deepEqual(events, [{ required: true }, { required: false }, { required: true }, { required: false }])
  } finally {
    ws.cleanup()
  }
})

// ---------------------------------------------------------------------------
// (e) Orphaned-run protection: no double-installer.
// ---------------------------------------------------------------------------

test('orphaned-run: teardown-abort settles the orphan so a later install resolves only one boot (one bootstrap)', async () => {
  const ws = makeWorkspace()

  try {
    const gate = createFirstRunGate()
    let bootstrapRuns = 0

    const runBootstrap = () => {
      bootstrapRuns += 1
    }

    // Boot A parks, then its caller goes away (e.g. a teardown for a profile
    // switch). Model the abandoned run with a single non-retrying attempt whose
    // rejection is swallowed.
    let orphanAborted = false

    const orphan = bootOnce({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    }).catch(error => {
      orphanAborted = isFirstRunAborted(error)
    })

    await flush()
    assert.deepEqual(gate.state(), { required: true })

    // teardownPrimaryBackendAndWait's firstRunGate.abort() settles the orphan and
    // RESETS the latch, so the orphan can't be re-joined by the next boot.
    gate.abort()
    await orphan
    assert.ok(orphanAborted, 'the orphaned parked run must settle with a firstRunAborted error')
    assert.equal(bootstrapRuns, 0, 'the orphan must never reach the installer')

    // A subsequent boot parks on a FRESH latch.
    const booting = boot({
      gate,
      readConnectionConfig: ws.readConnectionConfig,
      installChosen: ws.installChosen,
      runBootstrap
    })

    await flush()
    assert.deepEqual(gate.state(), { required: true })

    // A single install click resolves only this one run — exactly one installer.
    gate.chooseInstall()
    assert.deepEqual(await booting, { mode: 'local', source: 'local' })
    assert.equal(bootstrapRuns, 1, 'exactly one bootstrap invocation ever happens (no double-installer)')
  } finally {
    ws.cleanup()
  }
})

// ---------------------------------------------------------------------------
// Source-assertion guards: main.ts has no exports, so pin its wiring by regex
// (repo pattern — see windows-hermes-resolution.test.ts).
// ---------------------------------------------------------------------------

function readMain() {
  return fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')
}

test('main.ts: ensureRuntime calls the first-run guard BEFORE the Windows bootstrap-recovery handoff', () => {
  const source = readMain()
  const guardIndex = source.indexOf('guardFirstRunBootstrap({')
  const handoffIndex = source.indexOf("handOffWindowsBootstrapRecovery('bootstrap-needed')")

  assert.notEqual(guardIndex, -1, 'ensureRuntime must call guardFirstRunBootstrap')
  assert.notEqual(handoffIndex, -1, "ensureRuntime must call handOffWindowsBootstrapRecovery('bootstrap-needed')")
  assert.ok(
    guardIndex < handoffIndex,
    'the first-run choice must be resolved BEFORE the recovery handoff (a "connect to a server" user must never trigger recovery)'
  )
})

test('main.ts: the primary boot enables the gate and the pool call site does not', () => {
  const source = readMain()

  assert.match(
    source,
    /ensureRuntime\(resolveHermesBackend\(backendArgs\), \{ firstRunGate: true \}\)/,
    'startHermes must enable the first-run gate'
  )
  assert.match(
    source,
    /ensureRuntime\(resolveHermesBackend\(backendArgs\)\)/,
    'the pool call site must call ensureRuntime WITHOUT the firstRunGate option'
  )
})

test('main.ts: startHermes catch re-runs non-latching on a first-run abort', () => {
  const source = readMain()

  assert.match(
    source,
    /if \(isFirstRunAborted\(error\)\) \{\s*backendConnectionState\.clearPromiseForAttempt\(connectionAttempt\)\s*\n\s*return startHermes\(\)/,
    'the catch must recognize the abort via isFirstRunAborted, clear the cached attempt, and re-run startHermes (non-latching)'
  )
})

test('main.ts: teardownPrimaryBackendAndWait aborts the gate before capturing the dying process', () => {
  const source = readMain()
  const fnStart = source.indexOf('async function teardownPrimaryBackendAndWait(')
  assert.notEqual(fnStart, -1, 'teardownPrimaryBackendAndWait must exist')

  const fnEnd = source.indexOf('\nasync function ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)
  const abortIndex = body.indexOf('firstRunGate.abort()')
  const dyingIndex = body.indexOf('const dying =')

  assert.ok(abortIndex !== -1 && dyingIndex !== -1, 'teardown must abort the gate and capture the dying process')
  assert.ok(
    abortIndex < dyingIndex,
    'the gate must be aborted before capturing the dying process (settle the orphan so no later Install resumes two bootstraps)'
  )
})

test('main.ts: connection-config:apply writes the config before tearing the primary backend down', () => {
  const source = readMain()
  const handlerStart = source.indexOf("ipcMain.handle('hermes:connection-config:apply'")
  assert.notEqual(handlerStart, -1, 'the connection-config:apply handler must exist')

  const handlerBody = source.slice(handlerStart, handlerStart + 1200)
  const writeIndex = handlerBody.indexOf('writeDesktopConnectionConfig(config)')
  const teardownIndex = handlerBody.indexOf('teardownPrimaryBackendAndWait({ soft: true })')

  assert.ok(
    writeIndex !== -1 && teardownIndex !== -1,
    'the handler must write the config and soft-teardown the primary'
  )
  assert.ok(
    writeIndex < teardownIndex,
    'the new remote config must be written BEFORE the abort so the aborted boot re-dials the freshly-saved remote'
  )
})
