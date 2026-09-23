import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { test } from 'vitest'

const executable = path.resolve(process.env.HERMES_MANAGED_ROLLOUT_ELECTRON_EXECUTABLE ||
  path.join(path.dirname(fileURLToPath(import.meta.url)), '../node_modules/electron/dist/electron.exe'))

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

  const child = spawn(executable, [appDirectory, '--disable-gpu'], {
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

async function resultOf(process: FixtureProcess, filename: string): Promise<Record<string, any>> {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (fs.existsSync(filename)) {return JSON.parse(fs.readFileSync(filename, 'utf8'))}

    if (process.child.exitCode !== null || process.child.signalCode !== null) {
      throw new Error(`Fixture process exited ${process.child.exitCode ?? process.child.signalCode} before result: ${process.output()}`)
    }

    await delay(50)
  }

  throw new Error(`Timed out waiting for fixture result: ${process.output()}`)
}

async function stopOwnedProcess(process: FixtureProcess): Promise<void> {
  if (process.child.exitCode !== null || process.child.signalCode !== null) {return}
  process.child.kill()

  for (let attempt = 0; attempt < 100 && process.child.exitCode === null && process.child.signalCode === null; attempt += 1) {await delay(50)}
  assert.ok(process.child.exitCode !== null || process.child.signalCode !== null,
    `Fixture process did not exit: ${process.output()}`)
}

function journalBytes(directory: string): Record<string, string> {
  return Object.fromEntries(
    fs.readdirSync(directory).sort().map(name => [name, fs.readFileSync(path.join(directory, name)).toString('base64')])
  )
}

const runTest = fs.existsSync(executable) ? test : test.skip

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
    const successor = launch(secondApp, 'takeover', 202, userData, path.join(results, 'successor.json'), stale)
    processes.push(successor)
    const successorResult = await resultOf(successor, path.join(results, 'successor.json'))
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
    fs.rmSync(runDirectory, { recursive: true, force: true })
  }
}, 60_000)
