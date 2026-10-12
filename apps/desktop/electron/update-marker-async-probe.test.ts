/**
 * Review 5411222842: the update gate judges a marker on Electron's main
 * thread every poll while one exists, so its creation-time and process-state
 * probes must never spawn synchronously (`execFileSync` ps / getconf /
 * powershell blocked the UI for up to 5 s / 15 s). They go through async
 * `execFile`; the synchronous variants stay for the module-init repair lock.
 */

import assert from 'node:assert/strict'
import type * as ChildProcess from 'node:child_process'

import { afterAll, afterEach, beforeAll, test, vi } from 'vitest'

import { hostJudgeEnv, processCreateTime, psCreateTime } from './update-marker'
import { cleanupMarkerFixtures, liveOwner } from './update-marker.test-helpers'

const syncSpawns = vi.hoisted(() => [] as string[])

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcess>()

  return {
    ...actual,
    execFileSync: ((file: string, ...rest: unknown[]) => {
      syncSpawns.push(file)

      return (actual.execFileSync as (...args: unknown[]) => unknown)(file, ...rest)
    }) as typeof actual.execFileSync
  }
})

let owner: Awaited<ReturnType<typeof liveOwner>>

beforeAll(async () => {
  owner = await liveOwner()
})

afterAll(() => {
  owner.kill()
  cleanupMarkerFixtures()
})

afterEach(() => {
  syncSpawns.length = 0
})

function asPlatform<T>(platform: NodeJS.Platform, run: () => Promise<T>): Promise<T> {
  const original = Object.getOwnPropertyDescriptor(process, 'platform')!
  Object.defineProperty(process, 'platform', { ...original, value: platform })

  return run().finally(() => Object.defineProperty(process, 'platform', original))
}

test.skipIf(process.platform !== 'linux')('the Linux creation-time probe never spawns synchronously', async () => {
  const ct = await processCreateTime(owner.pid)

  assert.ok(ct !== null && Math.abs(ct - Date.now() / 1000) < 600, `ct ${ct}`)
  assert.deepEqual(syncSpawns, [])
})

// macOS reads `ps`; procps on Linux prints the same columns, so the darwin
// branch runs here against a real `ps`.
test.skipIf(process.platform === 'win32')('the macOS probes ask ps without blocking the main thread', async () => {
  const reference = psCreateTime(owner.pid)
  syncSpawns.length = 0

  const [ct, alive] = await asPlatform('darwin', async () => [
    await processCreateTime(owner.pid),
    await hostJudgeEnv().isAlive(owner.pid)
  ])

  assert.ok(ct !== null && reference !== null && Math.abs(ct - reference) <= 1, `async ${ct}, sync ${reference}`)
  assert.equal(alive, true)
  assert.deepEqual(syncSpawns, [], 'no execFileSync on the judgement path')
})
