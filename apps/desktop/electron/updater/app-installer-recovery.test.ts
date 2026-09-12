import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { AppInstallerStrategy, type AppInstallerStrategyDeps } from './app-installer'
import { PENDING_RELAUNCH_FILENAME, registerUpdateRelaunch, type RelaunchRegistration } from './relaunch'

for (const failureAt of ['prepare', 'register', 'teardown', 'open', 'none']) {
  test(`App Installer restores only after teardown begins (${failureAt})`, async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'appinstaller-recovery-'))
    const marker = path.join(home, PENDING_RELAUNCH_FILENAME)
    const failure = new Error(`failed ${failureAt}`)
    const calls: string[] = []
    const progress: string[] = []
    let running = true

    const act = (at: string): void => {
      calls.push(at)

      if (at === failureAt) {throw failure}
    }

    const deps: AppInstallerStrategyDeps = {
      python: 'unused-checker', script: 'unused.py',
      run: async () => { throw new Error('configured feed does not need the checker') },
      channel: 'stable', light: false, appVersion: '1.0', feedBaseUrl: 'https://example.invalid',
      installer: {
        prepare: async () => {
          act('prepare')

          return 'fixture.appinstaller'
        },
        open: async () => {
          assert.equal(running, false)
          act('open')

          return ''
        }
      },
      registerPendingRelaunch: (version: string): Promise<RelaunchRegistration> => registerUpdateRelaunch({ getPath: (): string => home }, version, {
        relaunch: async () => {
          act('register')

          return { cancel: async () => { calls.push('cancel') } }
        }
      }),
      teardownBundledBackend: async () => { running = false; act('teardown') },
      restoreBundledBackend: async () => { running = true; calls.push('restore') },
      emitUpdateProgress: value => { progress.push(value.stage) },
      quit: () => { calls.push('quit') }
    }

    try {
      if (failureAt === 'none') {
        const result = await new AppInstallerStrategy(deps).apply({})
        assert.equal(result.ok, true)
        assert.equal(result.handedOff, true, 'keep backend restart blocked until the quitting app exits')
        assert.deepEqual(calls, ['prepare', 'register', 'teardown', 'open', 'quit'])
        assert.equal(fs.existsSync(marker), true)
      } else {
        await assert.rejects(new AppInstallerStrategy(deps).apply({}), error => error === failure)
        assert.equal(running, true)
        assert.equal(calls.includes('quit'), false)
        assert.equal(calls.includes('restore'), ['teardown', 'open'].includes(failureAt))
        assert.equal(calls.includes('cancel'), ['teardown', 'open'].includes(failureAt))
        assert.equal(fs.existsSync(marker), false)
        assert.equal(progress.at(-1), 'error')
      }
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  })
}

test('handoff errors retain cleanup failures while still restoring the backend', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'appinstaller-errors-'))
  const original = new Error('descriptor open failed')
  const cancellation = new Error('waiter stop failed')
  const recovery = new Error('backend restart failed')
  const order: string[] = []

  const deps: AppInstallerStrategyDeps = {
    python: 'unused', script: 'unused.py', run: async () => ({ code: 0, stdout: '' }),
    channel: 'stable', light: false, appVersion: '1.0', feedBaseUrl: 'https://example.invalid',
    installer: { prepare: async () => 'file', open: async () => { throw original } },
    registerPendingRelaunch: (version: string): Promise<RelaunchRegistration> => registerUpdateRelaunch({ getPath: (): string => home }, version, {
      relaunch: async () => ({ cancel: async () => { order.push('cancel'); throw cancellation } })
    }),
    teardownBundledBackend: async () => { order.push('stop') },
    restoreBundledBackend: async () => { order.push('restore'); throw recovery },
    emitUpdateProgress: event => { if (event.stage === 'error') {order.push('error')} },
    quit: () => { throw new Error('failed handoff must not quit') }
  }

  try {
    await assert.rejects(new AppInstallerStrategy(deps).apply({}), error => {
      assert.ok(error instanceof AggregateError)
      assert.equal(error.cause, original)
      assert.equal(error.errors[0], original)
      assert.ok(error.errors[1] instanceof AggregateError)
      assert.equal(error.errors[1].errors[0], cancellation)
      assert.equal(error.errors[2], recovery)

      return true
    })
    assert.deepEqual(order, ['stop', 'cancel', 'restore', 'error'])
    assert.equal(fs.existsSync(path.join(home, PENDING_RELAUNCH_FILENAME)), false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})
