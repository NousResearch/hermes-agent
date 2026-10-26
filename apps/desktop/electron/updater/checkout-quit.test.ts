import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, it, vi } from 'vitest'

import { createBackendShutdownCoordinator } from '../backend-ownership'
import { quitPromptFor } from '../quit-guard'
import { backendQuitNeedsWait, createQuitTeardownCoordinator } from '../quit-teardown'
import * as updaterProcess from '../updater-process'

import { createCheckoutStrategy } from './checkout'
import { createCheckoutQuitLifecycle } from './checkout-quit'

function deferred() {
  let resolve!: () => void
  let reject!: (error: Error) => void

  const promise = new Promise<void>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, resolve, reject }
}

it.skipIf(process.platform === 'win32').each(['success', 'failure'] as const)(
  'ordinary quit waits for a pending snapshot (%s) without handing off the cancelled update',
  async outcome => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-quit-'))
    const scripts = path.join(home, 'scripts', 'desktop-update')
    fs.mkdirSync(scripts, { recursive: true })
    fs.writeFileSync(path.join(scripts, 'posix.sh'), '#!/bin/sh\nexit 0\n')
    const snapshot = deferred()
    const started = deferred()
    const events: string[] = []
    const app = new EventEmitter()

    const quit = vi.fn(() => {
      const event = { preventDefault: vi.fn() }
      app.emit('before-quit', event)

      return event
    })

    const stopBackend = vi.fn(async () => {
      events.push('backend-stop')
    })

    const shutdown = createBackendShutdownCoordinator(stopBackend)
    const teardown = createQuitTeardownCoordinator(quit)

    const spawn = vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(() => {
      throw new Error('unexpected updater spawn')
    })

    const markHandoff = vi.fn()
    const stopForUpdate = vi.fn(async () => {})

    const lifecycle = createCheckoutQuitLifecycle({
      preflightStateDb: () => {
        started.resolve()

        return snapshot.promise
      },
      // Idle app-owned backend: no active-work confirmation is needed.
      heldQuitForActiveWork: () => false,
      quit,
      teardown: event => {
        if (
          teardown.begin([
            {
              run: () => shutdown.run(),
              waitForCompletion: backendQuitNeedsWait({
                connectionPending: false,
                processAttached: true,
                shutdownPending: shutdown.isPending()
              })
            }
          ])
        ) {
          event.preventDefault()
        }
      }
    })

    // The same preflight/before-quit pair is installed in main.ts. Exercise the
    // real checkout continuation and shutdown coordinators, not a mock updater.
    app.on('before-quit', lifecycle.beforeQuit)

    const strategy = createCheckoutStrategy({
      hermesHome: home,
      isWindows: process.platform === 'win32',
      isMac: process.platform === 'darwin',
      defaultUpdateBranch: 'main',
      updateHandoffDwellMs: 0,
      readSourceUpdate: async () => ({ supported: true, branch: 'main' }),
      resolveUpdateRoot: () => home,
      resolveUpdaterBinary: () => null,
      remoteGatewayActive: () => false,
      emitUpdateProgress: vi.fn(),
      rememberLog: vi.fn(),
      startHermes: vi.fn(async () => {}),
      stopBackendsForUpdate: stopForUpdate,
      repairMacUpdaterHelper: vi.fn(),
      preflightStateDb: lifecycle.preflightStateDb,
      runningAppBundle: () => null,
      markQuittingForHandoff: markHandoff,
      quit
    })

    const update = strategy.apply().then(
      result => result,
      error => error as Error
    )

    try {
      await started.promise
      expect(quit().preventDefault).toHaveBeenCalledOnce()
      expect(quit().preventDefault).toHaveBeenCalledOnce()
      await new Promise(resolve => setImmediate(resolve))
      expect(stopBackend, 'ordinary quit must not tear down an unresolved snapshot').not.toHaveBeenCalled()
      expect(spawn).not.toHaveBeenCalled()

      events.push(`snapshot-${outcome}`)

      if (outcome === 'success') {
        snapshot.resolve()
      } else {
        snapshot.reject(new Error('snapshot failed'))
      }

      const result = await update
      expect(result).toBeInstanceOf(Error)
      expect((result as Error).message).toMatch(outcome === 'success' ? /cancelled.*quit/i : /snapshot failed/)
      await new Promise(resolve => setImmediate(resolve))
      expect(events).toEqual([`snapshot-${outcome}`, 'backend-stop'])
      expect(stopBackend).toHaveBeenCalledOnce()
      expect(quit).toHaveBeenCalledTimes(4) // two requests, one join, one teardown completion
      expect(stopForUpdate).not.toHaveBeenCalled()
      expect(spawn).not.toHaveBeenCalled()
      expect(markHandoff).not.toHaveBeenCalled()
      expect(fs.readdirSync(home)).toEqual(['scripts'])
      await expect(lifecycle.preflightStateDb(home, vi.fn())).rejects.toThrow(/cancelled.*quit/i)
    } finally {
      snapshot.resolve()
      await update
      spawn.mockRestore()
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)

it('Keep Running does not cancel preflight or latch the next ordinary quit', async () => {
  const snapshot = deferred()
  let hold = true
  const teardown = vi.fn()
  const quit = vi.fn()
  const event = { preventDefault: vi.fn() }

  const lifecycle = createCheckoutQuitLifecycle({
    preflightStateDb: () => snapshot.promise,
    heldQuitForActiveWork: current => {
      if (hold) {
        current.preventDefault()
      }

      return hold
    },
    teardown,
    quit
  })

  const preflight = lifecycle.preflightStateDb('home', vi.fn())
  lifecycle.beforeQuit(event)
  expect(event.preventDefault).toHaveBeenCalledOnce()
  expect(teardown).not.toHaveBeenCalled()
  snapshot.resolve()
  await expect(preflight).resolves.toBeUndefined()
  expect(quit).not.toHaveBeenCalled()
  hold = false
  lifecycle.beforeQuit(event)
  expect(teardown).toHaveBeenCalledExactlyOnceWith(event)
})

it.each(['before', 'after'] as const)(
  'Keep Running %s snapshot settlement releases future updates without reviving cancelled work',
  async timing => {
    const snapshot = deferred()
    const laterSnapshot = deferred()
    const app = new EventEmitter()
    let activeWork = false
    const teardown = vi.fn()
    const promptShown = vi.fn()

    const backup = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(() => snapshot.promise)
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(() => laterSnapshot.promise)

    const quit = vi.fn(() => app.emit('before-quit', { preventDefault: vi.fn() }))

    const lifecycle = createCheckoutQuitLifecycle({
      preflightStateDb: backup,
      heldQuitForActiveWork: event => {
        const prompt = quitPromptFor({ count: activeWork ? 1 : 0, titles: [] }, false)

        if (!prompt) {
          return false
        }

        event.preventDefault()
        promptShown(prompt.buttons)

        // The native guard holds this quit; Keep Running does not re-enter
        // app.quit(). Only Quit Anyway would make another admitted request.
        return true
      },
      teardown,
      quit
    })

    app.on('before-quit', lifecycle.beforeQuit)

    const interrupted = lifecycle.preflightStateDb('home', vi.fn())
    const interruptedResult = expect(interrupted).rejects.toThrow(/cancelled.*quit/i)
    await Promise.resolve()
    quit() // Initially idle: admitted, waiting on the snapshot.
    activeWork = true

    if (timing === 'before') {
      quit() // Active work now holds a repeated quit before snapshot settlement.
    }

    snapshot.resolve()
    await interruptedResult
    await new Promise(resolve => setImmediate(resolve))
    expect(promptShown).toHaveBeenCalledExactlyOnceWith(['Keep Running', 'Quit Anyway'])
    expect(quit).toHaveBeenCalledTimes(2) // No stale third quit after Keep Running.
    expect(teardown).not.toHaveBeenCalled()
    await expect(lifecycle.preflightStateDb('home', vi.fn())).resolves.toBeUndefined()
    expect(backup).toHaveBeenCalledTimes(2)

    // A new admitted quit must still wait and re-enter once; an old wait latch
    // must not strand it, nor may cancelling the earlier quit revive its update.
    activeWork = false
    const later = lifecycle.preflightStateDb('home', vi.fn())
    const laterResult = expect(later).rejects.toThrow(/cancelled.*quit/i)
    await Promise.resolve()
    quit()
    expect(teardown).not.toHaveBeenCalled()
    laterSnapshot.resolve()
    await laterResult
    await new Promise(resolve => setImmediate(resolve))
    expect(backup).toHaveBeenCalledTimes(3)
    expect(teardown).toHaveBeenCalledOnce()
    expect(quit).toHaveBeenCalledTimes(4)
  }
)
