import { expect, test } from 'vitest'

import { UpdateOperation } from './operation'

import type { UpdaterApplyResultWire, UpdaterStrategy } from './index'

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (error: Error) => void
}

function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise: Promise<T> = new Promise<T>((done, fail): void => {
    resolve = done
    reject = fail
  })

  return { promise, resolve, reject }
}

function strategy(): UpdaterStrategy {
  return {
    mechanism: 'external',
    check: async () => ({ supported: false }),
    apply: async () => ({ ok: true })
  }
}

test('check and apply share one pending strategy, and a failed initialization can retry', async (): Promise<void> => {
  const pending = deferred<UpdaterStrategy | null>()
  const installed: UpdaterStrategy = strategy()
  let attempts: number = 0

  const operation = new UpdateOperation((): Promise<UpdaterStrategy | null> => {
    attempts += 1

    return attempts === 1 ? pending.promise : Promise.resolve(installed)
  })

  const first: Promise<UpdaterStrategy | null> = operation.resolve()
  const second: Promise<UpdaterStrategy | null> = operation.resolve()
  expect(attempts).toBe(1)
  const failures = Promise.allSettled([first, second])
  pending.reject(new Error('native inspection failed'))
  expect((await failures).map((result): string => result.status)).toEqual(['rejected', 'rejected'])
  expect(await operation.resolve()).toBe(installed)
  expect(await operation.resolve()).toBe(installed)
  expect(attempts).toBe(2)
})

test('apply ownership precedes async resolution, survives restoration, and retains a successful handoff', async (): Promise<void> => {
  const pending = deferred<UpdaterStrategy | null>()
  const operation = new UpdateOperation((): Promise<UpdaterStrategy | null> => pending.promise)
  const restoration = deferred()
  const restoring = deferred()
  let applications: number = 0

  const first: Promise<UpdaterApplyResultWire> = operation.apply(async (): Promise<UpdaterApplyResultWire> => {
    applications += 1
    await operation.resolve()
    restoring.resolve()
    await restoration.promise
    throw new Error('install failed')
  })

  const competing: Promise<UpdaterApplyResultWire> = operation.apply(async () => ({ ok: true }))
  const rejected = expect(competing).rejects.toThrow('already in progress')
  pending.resolve(strategy())
  await rejected
  await restoring.promise
  await expect(operation.apply(async () => ({ ok: true }))).rejects.toThrow('already in progress')
  restoration.resolve()
  await expect(first).rejects.toThrow('install failed')
  expect(applications).toBe(1)
  await expect(operation.apply(async () => ({ ok: false }))).resolves.toEqual({ ok: false })
  await expect(operation.apply(async () => ({ ok: true, handedOff: true }))).resolves.toEqual({
    ok: true,
    handedOff: true
  })
  await expect(operation.apply(async () => ({ ok: true }))).rejects.toThrow('already in progress')
})
