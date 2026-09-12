import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { App } from 'electron'
import { afterEach, beforeEach, test, vi } from 'vitest'

import {
  consumePendingRelaunch,
  PENDING_RELAUNCH_FILENAME,
  registerUpdateRelaunch,
  type RelaunchRegistration,
  writePendingRelaunch
} from './relaunch'
import type { RelaunchWaiterHandle } from './relaunch-waiter'

let root: string
let hermesHome: string
let stable: Pick<App, 'getPath'>
let canary: Pick<App, 'getPath'>
let commit: Pick<App, 'getPath'>

function installation(name: string): Pick<App, 'getPath'> {
  const userData: string = path.join(root, 'app-data', name)
  fs.mkdirSync(userData, { recursive: true })

  return {
    getPath: (key: Parameters<App['getPath']>[0]): string => {
      assert.equal(key, 'userData')

      return userData
    }
  }
}

function marker(app: Pick<App, 'getPath'>): string {
  return path.join(app.getPath('userData'), PENDING_RELAUNCH_FILENAME)
}

beforeEach((): void => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'relaunch-isolation-'))
  hermesHome = path.join(root, 'hermes-home')
  fs.mkdirSync(hermesHome)
  vi.stubEnv('HERMES_HOME', hermesHome)
  stable = installation('stable')
  canary = installation('canary')
  commit = installation('commit')
})

afterEach((): void => {
  vi.unstubAllEnvs()
  fs.rmSync(root, { recursive: true, force: true })
})

test('writes and startup consumption stay in each installation despite a shared Hermes home', async (): Promise<void> => {
  const globalMarker: string = path.join(hermesHome, PENDING_RELAUNCH_FILENAME)
  const legacyContents: string = JSON.stringify({ schemaVersion: 1, fromVersion: 'legacy', startedAt: 1 })
  fs.writeFileSync(globalMarker, legacyContents)

  assert.equal(writePendingRelaunch(stable, 'stable-old'), true)
  const stableContents: string = fs.readFileSync(marker(stable), 'utf8')

  const registration: RelaunchRegistration = await registerUpdateRelaunch(canary, 'canary-old', {
    relaunch: (): undefined => undefined
  })

  assert.equal(registration.automatic, false)
  const canaryContents: string = fs.readFileSync(marker(canary), 'utf8')

  assert.deepEqual(consumePendingRelaunch(commit, 'commit-build'), { wasUpdateRelaunch: false })
  assert.equal(fs.readFileSync(marker(stable), 'utf8'), stableContents)
  assert.equal(fs.readFileSync(marker(canary), 'utf8'), canaryContents)
  assert.deepEqual(consumePendingRelaunch(canary, 'canary-new'), {
    wasUpdateRelaunch: true,
    fromVersion: 'canary-old'
  })
  assert.equal(fs.existsSync(marker(canary)), false)
  assert.equal(fs.readFileSync(marker(stable), 'utf8'), stableContents)
  assert.deepEqual(consumePendingRelaunch(canary, 'canary-new'), { wasUpdateRelaunch: false })
  assert.equal(writePendingRelaunch(canary, 'canary-new'), true)
  const nextCanaryContents: string = fs.readFileSync(marker(canary), 'utf8')
  assert.deepEqual(consumePendingRelaunch(stable, 'stable-old'), {
    wasUpdateRelaunch: false,
    fromVersion: 'stable-old'
  })
  assert.equal(fs.existsSync(marker(stable)), false)
  assert.equal(fs.readFileSync(marker(canary), 'utf8'), nextCanaryContents)
  assert.equal(fs.readFileSync(globalMarker, 'utf8'), legacyContents)
})

test('cancellation and failed registration remove only their own installation marker', async (): Promise<void> => {
  const survivor: RelaunchRegistration = await registerUpdateRelaunch(stable, 'stable-old', {
    relaunch: (): undefined => undefined
  })

  const stableContents: string = fs.readFileSync(marker(stable), 'utf8')
  let cancelled: number = 0

  for (const automatic of [true, false]) {
    const registration: RelaunchRegistration = await registerUpdateRelaunch(canary, 'canary-old', {
      relaunch: (): RelaunchWaiterHandle | undefined =>
        automatic
          ? {
              cancel: async (): Promise<void> => {
                cancelled++
              }
            }
          : undefined
    })

    assert.equal(fs.existsSync(marker(canary)), true)
    await Promise.all([registration.cancel(), registration.cancel()])
    assert.equal(fs.existsSync(marker(canary)), false)
    assert.equal(fs.readFileSync(marker(stable), 'utf8'), stableContents)
  }

  assert.equal(cancelled, 1)
  const failure: Error = new Error('waiter failed')
  await assert.rejects(
    registerUpdateRelaunch(canary, 'canary-old', {
      relaunch: async (): Promise<never> => {
        throw failure
      }
    }),
    (error: unknown): boolean => error === failure
  )
  assert.equal(fs.existsSync(marker(canary)), false)
  assert.equal(fs.readFileSync(marker(stable), 'utf8'), stableContents)

  const failedCancellation: RelaunchRegistration = await registerUpdateRelaunch(canary, 'canary-old', {
    relaunch: (): RelaunchWaiterHandle => ({
      cancel: async (): Promise<never> => {
        throw failure
      }
    })
  })

  await assert.rejects(failedCancellation.cancel(), AggregateError)
  assert.equal(fs.existsSync(marker(canary)), false)
  assert.equal(fs.readFileSync(marker(stable), 'utf8'), stableContents)
  assert.equal(fs.existsSync(path.join(hermesHome, PENDING_RELAUNCH_FILENAME)), false)

  const sibling: RelaunchRegistration = await registerUpdateRelaunch(canary, 'canary-old', {
    relaunch: (): undefined => undefined
  })

  const siblingContents: string = fs.readFileSync(marker(canary), 'utf8')
  await survivor.cancel()
  assert.equal(fs.existsSync(marker(stable)), false)
  assert.equal(fs.readFileSync(marker(canary), 'utf8'), siblingContents)
  await sibling.cancel()
})
