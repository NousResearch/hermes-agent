import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { PENDING_RELAUNCH_FILENAME, registerUpdateRelaunch, type RelaunchRegistration } from './relaunch'
import type { RelaunchWaiterHandle } from './relaunch-waiter'

for (const automatic of [true, false]) {
  test(`registration owns its marker and cancellation (automatic=${automatic})`, async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'relaunch-registration-'))
    const marker = path.join(home, PENDING_RELAUNCH_FILENAME)
    let resolveReady!: (handle: RelaunchWaiterHandle | undefined) => void
    const ready = new Promise<RelaunchWaiterHandle | undefined>(resolve => { resolveReady = resolve })
    let cancelled = 0
    let registered = false

    try {
      const pending: Promise<RelaunchRegistration> = registerUpdateRelaunch({ getPath: (): string => home }, '1.0', { relaunch: (): Promise<RelaunchWaiterHandle | undefined> => ready })
        .then(result => { registered = true;

 return result })

      await new Promise(setImmediate)
      assert.equal(registered, false)
      assert.equal(JSON.parse(fs.readFileSync(marker, 'utf8')).fromVersion, '1.0')
      resolveReady(automatic ? { cancel: async () => { cancelled++ } } : undefined)
      const registration = await pending
      assert.equal(registration.automatic, automatic)
      assert.equal(fs.existsSync(marker), true)
      await Promise.all([registration.cancel(), registration.cancel()])
      await registration.cancel()
      assert.equal(cancelled, automatic ? 1 : 0)
      assert.equal(fs.existsSync(marker), false)
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  })
}

test('start and cancellation failures preserve the cause and still clean owned markers', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'relaunch-failures-'))
  const marker = path.join(home, PENDING_RELAUNCH_FILENAME)
  const failure = new Error('owned child refused to stop')

  try {
    await assert.rejects(registerUpdateRelaunch({ getPath: (): string => home }, '1.0', {
      relaunch: async () => { throw failure }
    }), error => error === failure)
    assert.equal(fs.existsSync(marker), false)

    let attempts = 0

    const registration: RelaunchRegistration = await registerUpdateRelaunch({ getPath: (): string => home }, '1.0', {
      relaunch: async () => ({ cancel: async () => { attempts++; throw failure } })
    })

    await assert.rejects(registration.cancel(), error => error instanceof AggregateError && error.errors.includes(failure))
    await assert.rejects(registration.cancel(), error => error instanceof AggregateError && error.errors.includes(failure))
    assert.equal(attempts, 1)
    assert.equal(fs.existsSync(marker), false)

    const blocked: RelaunchRegistration = await registerUpdateRelaunch({ getPath: (): string => home }, '1.0', {
      relaunch: async () => ({ cancel: async () => { throw failure } })
    })

    fs.unlinkSync(marker)
    fs.mkdirSync(marker)
    fs.writeFileSync(path.join(marker, 'foreign-data'), 'keep')
    await assert.rejects(blocked.cancel(), error => {
      assert.ok(error instanceof AggregateError)
      assert.equal(error.errors[0], failure)
      assert.equal(error.errors.length, 2)

      return true
    })
    assert.equal(fs.readFileSync(path.join(marker, 'foreign-data'), 'utf8'), 'keep')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})
