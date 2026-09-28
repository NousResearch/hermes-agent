import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { test } from 'vitest'

// Resolve the installed Electron binary the way the portal-session live test
// does, so this ownership test can run wherever Electron is installed rather
// than only where a hard-coded electron.exe exists. An explicit override still
// wins; when nothing resolves, the test skips explicitly instead of failing
// the suite on an environment that has no Electron at all.
const defaultElectron = (() => {
  try {
    return String(createRequire(import.meta.url)('electron'))
  } catch {
    return ''
  }
})()

const executable = path.resolve(process.env.HERMES_MANAGED_ROLLOUT_ELECTRON_EXECUTABLE ||
  defaultElectron || path.join(os.tmpdir(), 'managed-rollout-electron-unavailable'))

// Chromium needs a display. Windows and macOS always have one; a headless
// Linux runner gets a virtual one through xvfb-run — the hosted JS lane
// installs xvfb for exactly this class of Electron test, and this mirrors the
// arrangement portal-session-live.test.ts uses so this ownership test runs on
// that lane instead of silently skipping. `null` means the test cannot run
// here and skips explicitly.
const displayPrefix = (() => {
  if (process.platform !== 'linux' || process.env.DISPLAY || process.env.WAYLAND_DISPLAY) {
    return []
  }

  const xvfbRun = (process.env.PATH ?? '')
    .split(path.delimiter)
    .map(dir => path.join(dir, 'xvfb-run'))
    .find(fs.existsSync)

  return xvfbRun ? [xvfbRun, '-a'] : null
})()

const fixtureParent = path.resolve(process.env.HERMES_MANAGED_ROLLOUT_FIXTURE_ROOT || os.tmpdir())

interface FixtureProcess {
  child: ChildProcess
  output: () => string
}

function launch(appDirectory: string, mode: string, generation: number, userData: string, result: string, stale: string): FixtureProcess {
  const environment: NodeJS.ProcessEnv = {
    ...process.env,
    HERMES_OWNER_FIXTURE_MODE: mode,
    HERMES_OWNER_FIXTURE_USER_DATA: userData,
    HERMES_OWNER_FIXTURE_RESULT: result,
    HERMES_OWNER_FIXTURE_STALE_CAPABILITY: stale,
    HERMES_OWNER_FIXTURE_GENERATION: String(generation)
  }

  delete environment.ELECTRON_RUN_AS_NODE

  // On Linux the sandbox aborts before the fixture runs (the npm-installed
  // chrome-sandbox helper is not setuid), and a headless host needs the
  // virtual display prefix; both mirror the portal-session and Playwright
  // fixtures that run on the hosted lane.
  const [command, ...prefixArguments] = [...(displayPrefix ?? []), executable]
  const child = spawn(command, [
    ...prefixArguments,
    appDirectory,
    ...(process.platform === 'linux' ? ['--no-sandbox'] : []),
    '--disable-gpu'
  ], {
    env: environment,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true
  })

  let output = ''

  for (const stream of [child.stdout, child.stderr]) {
    stream?.on('data', data => { output = (output + String(data)).slice(-16_384) })
  }

  return { child, output: () => output }
}

async function resultOf(process: FixtureProcess, filename: string, windowMs = 15_000): Promise<Record<string, any>> {
  let lastParseError = ''
  const attempts = Math.ceil(windowMs / 50)

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (fs.existsSync(filename)) {
      try {
        return JSON.parse(fs.readFileSync(filename, 'utf8'))
      } catch (error) {
        // The fixture publishes its result with a truncate-then-write; a reader racing that
        // write can observe a partial document, so a parse failure here is retried, not fatal.
        lastParseError = error instanceof Error ? error.message : String(error)
      }
    } else if (process.child.exitCode !== null || process.child.signalCode !== null) {
      throw new Error(`Fixture process exited ${process.child.exitCode ?? process.child.signalCode} before result: ${process.output()}`)
    }

    await delay(50)
  }

  throw new Error(`Timed out waiting for fixture result${lastParseError ? ` (last parse error: ${lastParseError})` : ''}: ${process.output()}`)
}

/**
 * Windows note: after TerminateProcess, libuv can delay the child's 'exit'
 * event for many seconds — or never deliver it — while the OS already reports
 * the pid gone. This box reproduced both: exit events arriving ~1s after
 * kill(), and OS-dead-with-no-event beyond 18s. The authoritative liveness
 * signal is therefore the OS (signal 0), with one grace cycle so a queued
 * event still lands when it is on time.
 *
 * The parameter is named `fixture`, not `process`: a parameter called
 * `process` shadows Node's global here and makes `process.kill` a TypeError
 * the catch would swallow — the process then reads as dead while it is alive.
 */
function processAlive(fixture: FixtureProcess): boolean {
  if (fixture.child.exitCode !== null || fixture.child.signalCode !== null) {return false}

  if (fixture.child.pid === undefined) {return false}

  try {
    process.kill(fixture.child.pid, 0)

    return true
  } catch {
    return false
  }
}

async function stopOwnedProcess(fixture: FixtureProcess): Promise<void> {
  if (!processAlive(fixture)) {return}

  const pid = fixture.child.pid

  fixture.child.kill()

  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (!processAlive(fixture)) {
      // Give a same-tick 'exit' event one cycle to land, then treat OS death as final.
      await delay(50)

      if (!processAlive(fixture)) {return}
    }

    // Escalate once the polite signal has had ~4s; on Windows this maps to TerminateProcess.
    if (attempt === 40) {fixture.child.kill('SIGKILL')}

    await delay(100)
  }

  assert.ok(!processAlive(fixture),
    `Fixture process ${pid} did not exit: ${fixture.output()}`)
}

function journalBytes(directory: string): Record<string, string> {
  return Object.fromEntries(
    fs.readdirSync(directory).sort().map(name => [name, fs.readFileSync(path.join(directory, name)).toString('base64')])
  )
}

const runTest = fs.existsSync(executable) && displayPrefix !== null ? test : test.skip

/**
 * Electron's OS single-instance lock is released asynchronously after the
 * previous holder's process dies: a successor launched immediately after
 * `kill()` can still see the lock held (~1.9s observed on this box). Retrying
 * launches until the successor actually acquires is the correct fix; asserting
 * on the first attempt races the OS lock teardown.
 *
 * Under full-suite load the bottleneck is Electron's boot time, not the lock:
 * a successor can take longer than one result window to publish anything.
 * A slow attempt is therefore retried, not fatal — only an explicit
 * `acquired: false` is evidence about the owner lock. The loop is bounded by
 * wall clock so a genuine hang still fails with a precise message.
 */
async function launchUntilAcquired(
  launchSuccessor: (resultFile: string, attempt: number) => FixtureProcess,
  resultsDirectory: string,
  deadlineMs = 90_000,
  perAttemptMs = 20_000
): Promise<{ process: FixtureProcess; result: Record<string, any> }> {
  const startedAt = Date.now()
  let lastResult: Record<string, any> = {}
  let lastError: unknown

  for (let attempt = 0; ; attempt += 1) {
    const remaining = deadlineMs - (Date.now() - startedAt)

    if (remaining <= 0) {break}

    const resultFile = path.join(resultsDirectory, `successor-${attempt}.json`)
    const process = launchSuccessor(resultFile, attempt)

    try {
      lastResult = await resultOf(process, resultFile, Math.min(perAttemptMs, remaining))
    } catch (error) {
      lastError = error

      try {
        await stopOwnedProcess(process)
      } catch {
        // A helper tree can outlive the attempt; the next attempt still proves
        // the lock behavior the test is about.
      }

      await delay(250)

      continue
    }

    await stopOwnedProcess(process)

    if (lastResult.acquired === true) {
      return { process, result: lastResult }
    }

    await delay(250)
  }

  const detail = lastError instanceof Error ? ` (last attempt error: ${lastError.message})` : ''

  throw new Error(`successor never acquired the process-owner lock within ${deadlineMs}ms: ${JSON.stringify(lastResult)}${detail}`)
}

runTest('one Electron userData owner admits updates; loser leaves journal unchanged; successor rejects stale authority', async () => {
  fs.mkdirSync(fixtureParent, { recursive: true })
  const runDirectory = fs.mkdtempSync(path.join(fixtureParent, 'single-writer-'))
  const userData = path.join(runDirectory, 'user-data')
  const results = path.join(runDirectory, 'results')
  const stale = path.join(runDirectory, 'stale-capability.json')
  const journalDirectory = path.join(userData, 'managed-rollouts', 'journal')
  const processes: FixtureProcess[] = []

  try {
    fs.mkdirSync(results)
    const bundle = path.join(runDirectory, 'main.cjs')
    await build({
      entryPoints: [fileURLToPath(new URL('./fixtures/managed-rollout-single-writer-child.ts', import.meta.url))],
      outfile: bundle,
      bundle: true,
      platform: 'node',
      format: 'cjs',
      target: 'node22',
      external: ['electron'],
      logLevel: 'silent'
    })

    const appDirectory = (name: string) => {
      const directory = path.join(runDirectory, name)
      fs.mkdirSync(directory)
      fs.copyFileSync(bundle, path.join(directory, 'main.cjs'))
      fs.writeFileSync(path.join(directory, 'package.json'), JSON.stringify({
        name,
        productName: name,
        version: '1.0.0',
        main: 'main.cjs'
      }))

      return directory
    }

    const firstApp = appDirectory('managed-rollout-owner-first')
    const secondApp = appDirectory('managed-rollout-owner-second')

    const winner = launch(firstApp, 'hold', 101, userData, path.join(results, 'winner.json'), stale)
    processes.push(winner)
    const winnerResult = await resultOf(winner, path.join(results, 'winner.json'))
    assert.equal(winnerResult.acquired, true, winner.output())
    assert.equal(winnerResult.journalRevision, 1)
    assert.equal(winnerResult.mutations, 1)
    const beforeLoser = journalBytes(journalDirectory)

    const loser = launch(secondApp, 'takeover', 202, userData, path.join(results, 'loser.json'), stale)
    processes.push(loser)
    const loserResult = await resultOf(loser, path.join(results, 'loser.json'))
    assert.equal(loserResult.acquired, false, loser.output())
    assert.equal(loserResult.admissionRefused, true)
    assert.equal(loserResult.journalConstructed, false)
    assert.notEqual(winnerResult.appName, loserResult.appName)
    assert.deepEqual(journalBytes(journalDirectory), beforeLoser)
    await stopOwnedProcess(loser)

    await stopOwnedProcess(winner)

    const successorRun = await launchUntilAcquired(
      (resultFile) => launch(secondApp, 'takeover', 202, userData, resultFile, stale),
      results
    )

    const successor = successorRun.process
    const successorResult = successorRun.result
    assert.equal(successorResult.acquired, true, successor.output())
    assert.equal(successorResult.priorGeneration, 101)
    assert.equal(successorResult.processGeneration, 202)
    assert.equal(successorResult.staleCapabilityRefused, true)
    assert.equal(successorResult.staleProofRefused, true)
    assert.equal(successorResult.journalRevision, 2)
    assert.equal(successorResult.mutations, 0)
    await stopOwnedProcess(successor)

    console.info(JSON.stringify({ winner: winnerResult, loser: loserResult, successor: successorResult }))
  } finally {
    for (const process of processes) {await stopOwnedProcess(process)}
    // The only recursive deletion target is the exact directory created above.
    assert.equal(path.dirname(path.resolve(runDirectory)), fixtureParent)
    // Electron's helper processes can outlive the killed main process for a
    // while and keep the userData directory open; a Windows EPERM here is a
    // lock race, not a behavior failure, and a throw from finally would mask
    // the real verdict. Wait for the lock to clear; if it never does, leave
    // the scratch directory under the OS temp root instead of failing an
    // already-completed behavioral receipt.
    let removed = false

    for (let attempt = 0; attempt < 120 && !removed; attempt += 1) {
      try {
        fs.rmSync(runDirectory, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })

        removed = true
      } catch {
        await delay(250)
      }
    }

    if (!removed) {console.warn(`fixture cleanup left scratch directory in place: ${runDirectory}`)}
  }
}, 240_000)
