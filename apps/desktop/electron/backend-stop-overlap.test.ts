import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'

import { test } from 'vitest'

import { stopBackendChild, waitForBackendExit } from './backend-child'
import { createLocalBackendLifecycle } from './local-backend-lifecycle'
import {
  assertPoolEntryStillOwned,
  type LocalBackendSlotEntry,
  LocalBackendSpawnCoordinator,
  releaseLocalBackendSlot,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'

test.skipIf(process.platform === 'win32')(
  'eviction, update and quit share physical exit before releasing a slot',
  async (): Promise<void> => {
    let signals = 0
    let released = false

    const physical = {
      forceKillProcessTree: (): never => {
        throw new Error('POSIX test')
      }
    }

    const lifecycle = createLocalBackendLifecycle<ChildProcess>({
      cancelSetup: (): void => {},
      stopChild: (child: ChildProcess): void => {
        signals++
        stopBackendChild(child, physical)
      },
      waitForExit: (child: ChildProcess): Promise<void> => waitForBackendExit(child, physical)
    })

    const child = lifecycle.spawn((): ChildProcess =>
      spawn(
        process.execPath,
        [
          '-e',
          `
    process.on('SIGTERM', () => process.send('stopping'));
    process.on('message', () => process.exit(0));
    setInterval(() => {}, 1000);
    process.send('ready');
  `
        ],
        { detached: true, stdio: ['ignore', 'ignore', 'ignore', 'ipc'] }
      )
    )

    child.once('exit', (): boolean => lifecycle.release(child))

    const deps = {
      pool: new Map([['profile', { process: child }]]),
      stopChild: (c: unknown): void => {
        void lifecycle.stop(c as ChildProcess)
      },
      waitForExit: (c: unknown): Promise<void> => lifecycle.stop(c as ChildProcess)
    }

    const pool = createPoolStopper(deps)

    try {
      await once(child, 'message')
      const signalled = once(child, 'message')

      const eviction = releaseLocalBackendSlotAfterExit(
        (): void => {
          released = true
        },
        (): Promise<void> => pool.stop('profile')
      )

      const update = lifecycle.stop(child)
      const quit = lifecycle.shutdown()
      await signalled
      assert.equal(signals, 1)
      assert.equal(released, false, 'a pool slot must remain occupied while its process is alive')
      assert.ok(pool.inFlight('profile'))
      assert.throws((): ChildProcess => lifecycle.spawn((): ChildProcess => child))
      child.send('exit')
      await Promise.all([eviction, update, quit])
      assert.equal(child.exitCode, 0)
      assert.equal(released, true)
      assert.equal(pool.inFlight('profile'), undefined)
      assert.equal(lifecycle.hasPending(), false)
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL')
        await once(child, 'exit')
      }
    }
  }
)

test.skipIf(process.platform === 'win32')(
  'a claim completed after eviction retains its live child capacity',
  async (): Promise<void> => {
    const slots = new LocalBackendSpawnCoordinator(1)
    const signal = new AbortController().signal
    const unspawned: LocalBackendSlotEntry = { releaseLocalBackendSlot: await slots.acquire('not-spawned') }
    assert.throws((): void => assertPoolEntryStillOwned('not-spawned', unspawned, new Map(), signal))
    assert.equal(slots.activeCount, 0, 'pre-spawn cancellation releases its reservation')

    const physical = {
      forceKillProcessTree: (): never => {
        throw new Error('POSIX test')
      }
    }

    const lifecycle = createLocalBackendLifecycle<ChildProcess>({
      cancelSetup: (): void => {},
      stopChild: (child: ChildProcess): void => stopBackendChild(child, physical),
      waitForExit: (child: ChildProcess): Promise<void> => waitForBackendExit(child, physical)
    })

    const release = await slots.acquire('claiming')

    const child = lifecycle.spawn((): ChildProcess =>
      spawn(
        process.execPath,
        [
          '-e',
          `
    process.on('SIGTERM', () => process.send('stopping'));
    process.on('message', () => process.exit(0));
    setInterval(() => {}, 1000);
    process.send('ready');
  `
        ],
        { detached: true, stdio: ['ignore', 'ignore', 'ignore', 'ipc'] }
      )
    )

    child.once('exit', (): boolean => lifecycle.release(child))
    const entry = { process: child, releaseLocalBackendSlot: release }
    const entries = new Map([['claiming', entry]])

    const pool = createPoolStopper({
      pool: entries,
      stopChild: (c: unknown): void => {
        void lifecycle.stop(c as ChildProcess)
      },
      waitForExit: (c: unknown): Promise<void> => lifecycle.stop(c as ChildProcess)
    })

    let finishClaim!: () => void

    const claim = new Promise<void>((resolve: () => void): void => {
      finishClaim = resolve
    })

    const starting = claim.then((): void => assertPoolEntryStillOwned('claiming', entry, entries, signal))
    const rejected = assert.rejects(starting, /cancelled/)

    try {
      await once(child, 'message')
      const signalled = once(child, 'message')

      const eviction = releaseLocalBackendSlotAfterExit(
        (): void => releaseLocalBackendSlot(entry),
        (): Promise<void> => pool.stop('claiming')
      )

      await signalled
      finishClaim()
      await rejected
      assert.equal(child.exitCode, null)
      assert.equal(child.signalCode, null)
      assert.ok(pool.inFlight('claiming'))
      assert.equal(slots.activeCount, 1, 'the post-claim guard must not release a live child slot')
      const replacement = slots.request('different-profile')
      assert.equal(replacement.queued, true)
      child.send('exit')
      await eviction
      const releaseReplacement = await replacement.acquired
      assert.equal(child.exitCode, 0)
      releaseReplacement()
      assert.equal(slots.activeCount, 0)
    } finally {
      finishClaim()
      await rejected

      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL')
        await once(child, 'exit')
      }
    }
  }
)
