import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { test, vi } from 'vitest'

import { stopBackendChild, type WaitableChild, waitForBackendExit } from './backend-child'
import { createLocalBackendLifecycle } from './local-backend-lifecycle'
import {
  LocalBackendSpawnCoordinator,
  registerLocalBackendExitFinalizer,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'

test('bounded stop retains a live child and its slot until late exit, without evicting a successor', async () => {
  vi.useFakeTimers()

  try {
    // A child that acknowledges signals without exiting exercises the otherwise
    // unprovable OS failure path; the waiter, stopper, lifecycle and allocator are real.
    const child = Object.assign(new EventEmitter(), {
      exitCode: null as number | null,
      signalCode: null as string | null,
      kill: vi.fn()
    })

    const deps = { forceKillProcessTree: () => {} }

    const lifecycle = createLocalBackendLifecycle<WaitableChild>({
      stopChild: child => stopBackendChild(child, deps),
      waitForExit: child => waitForBackendExit(child, deps, 20),
      cancelSetup: () => {}
    })

    const coordinator = new LocalBackendSpawnCoordinator(1)
    const release = await coordinator.acquire('profile')
    const entry = { process: lifecycle.spawn(() => child) }
    const pool = new Map<string, { process: WaitableChild | null }>([['profile', entry]])
    child.once('exit', () => lifecycle.release(child))
    registerLocalBackendExitFinalizer(pool, 'profile', entry, release)

    const stopper = createPoolStopper({
      pool,
      stopChild: child => stopBackendChild(child as WaitableChild, deps),
      waitForExit: child => waitForBackendExit(child as WaitableChild, deps, 20)
    })

    const next = coordinator.request('profile')
    const stopping = releaseLocalBackendSlotAfterExit(release, () => stopper.stop('profile'))

    const outcome = stopping.then(
      () => null,
      error => error
    )

    await vi.advanceTimersByTimeAsync(1020)
    assert.match(String(await outcome), /did not exit/)
    assert.equal(lifecycle.hasPending(), true, 'bounded failure must retain physical ownership')
    assert.equal(coordinator.activeCount, 1)
    assert.equal(coordinator.queuedCount, 1, 'timeout is not a free slot')

    // A replacement routing generation may already be waiting for the old lease.
    const successor = { process: null }
    pool.set('profile', successor)
    child.signalCode = 'SIGKILL'
    child.emit('exit', null, child.signalCode)
    const releaseNext = await next.acquired
    assert.equal(lifecycle.hasPending(), false)
    assert.equal(pool.get('profile'), successor)
    assert.equal(coordinator.activeCount, 1)
    assert.equal(coordinator.queuedCount, 0)

    child.emit('exit', null, child.signalCode)
    await releaseLocalBackendSlotAfterExit(release, () => waitForBackendExit(child, deps, 20))
    assert.equal(coordinator.activeCount, 1, 'old exit and cleanup cannot release the successor lease')
    assert.equal(child.listenerCount('exit'), 0)
    assert.equal(vi.getTimerCount(), 0)
    releaseNext()
    assert.equal(coordinator.activeCount, 0)
  } finally {
    vi.useRealTimers()
  }
})

test('real spawn failures finalize before a pending identity claim settles', async () => {
  const home = mkdtempSync(join(tmpdir(), 'pool-child-exit-'))

  try {
    // Exercise both an early crash and spawn's no-PID error (which has no exit event).
    for (const command of [process.execPath, join(home, 'missing-backend')]) {
      const coordinator = new LocalBackendSpawnCoordinator(1)
      const release = await coordinator.acquire('profile')

      const child = spawn(command, ['-e', 'process.exit(17)'], {
        env: { ...process.env, HERMES_HOME: home },
        stdio: 'ignore'
      })

      const entry = { process: child }
      const pool = new Map([['profile', entry]])

      const ended = new Promise<void>(resolve => {
        child.once('exit', () => resolve())
        child.once('error', () => resolve())
      })

      registerLocalBackendExitFinalizer(pool, 'profile', entry, release)
      let finishClaim!: () => void

      const claim = new Promise<void>(resolve => {
        finishClaim = resolve
      })

      let claimSettled = false

      const claimed = claim.then(() => {
        claimSettled = true
      })

      await ended
      assert.equal(claimSettled, false)
      assert.equal(coordinator.activeCount, 0, 'early failure must not wait for the identity claim')
      assert.equal(pool.has('profile'), false)

      const releaseNext = await coordinator.acquire('profile')
      finishClaim()
      await claimed
      await releaseLocalBackendSlotAfterExit(release, () =>
        waitForBackendExit(child, { forceKillProcessTree: () => {} }, 20)
      )
      assert.equal(coordinator.activeCount, 1, 'late failed-start cleanup owns only the old lease')
      releaseNext()
    }
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})
