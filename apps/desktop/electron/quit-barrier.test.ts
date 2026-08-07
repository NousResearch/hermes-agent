import assert from 'node:assert/strict'

import { test } from 'vitest'

import { QuitBarrier } from './quit-barrier'

function deferred<T = void>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

test('arm() waits for every registered promise before resolving', async () => {
  const barrier = new QuitBarrier()
  const ssh = deferred()
  const backend = deferred()
  barrier.add(ssh.promise)
  barrier.add(backend.promise)

  const armed = barrier.arm()
  assert.ok(armed, 'barrier should arm when there is pending work')

  let settled = false
  void armed.then(() => {
    settled = true
  })
  // Neither promise resolved yet — the barrier must still be pending.
  await Promise.resolve()
  assert.strictEqual(settled, false, 'barrier resolved before all work settled')

  // SSH tears down first — the barrier must NOT re-quit yet (backend pending).
  ssh.resolve()
  await Promise.resolve()
  assert.strictEqual(settled, false, 'barrier resolved after only SSH settled')

  // Both settled — now the single re-quit may fire.
  backend.resolve()
  await armed
  assert.strictEqual(settled, true)
})

test('re-entrant arm() returns null and cannot bypass an outstanding wait', () => {
  const barrier = new QuitBarrier()
  const slow = deferred()
  barrier.add(slow.promise)

  const first = barrier.arm()
  assert.ok(first)

  // A re-entrant before-quit tries to arm again — must be refused (no double
  // schedule), and work added after arming is dropped so the barrier stays the
  // single source of truth.
  assert.strictEqual(barrier.arm(), null)
  barrier.add(deferred().promise) // must be ignored
  assert.strictEqual(barrier.isEmpty, true, 'post-arm adds must be ignored')
  assert.strictEqual(barrier.arm(), null)
})

test('arm() resolves immediately (no artificial delay) when nothing is pending', () => {
  const barrier = new QuitBarrier()
  assert.strictEqual(barrier.arm(), null, 'empty barrier should be null, not wait')
})

test('allSettled semantics: a rejected wait still releases the barrier', async () => {
  const barrier = new QuitBarrier()
  barrier.add(Promise.reject(new Error('backend SIGKILL failed')))
  const armed = barrier.arm()
  assert.ok(armed)
  // Even a rejected promise releases the barrier (SIGKILL fallback hit its
  // bound, so there is nothing left to wait on) — it must not re-quit hang.
  await armed
})
