import { EventEmitter } from 'node:events'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import * as updaterProcess from '../updater-process'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import type { SourceUpdate } from './checkout-source'

const IS_WINDOWS: boolean = process.platform === 'win32'
const NO_GATEWAY_FLAG: string = IS_WINDOWS ? '-NoGateway' : '--no-gateway'

afterEach((): void => {
  vi.restoreAllMocks()
})

/** A checkout root with the repo hand-off script staged, plus a strategy over it. */
function handoffFixture(remote: boolean): { root: string; deps: CheckoutStrategyDeps } {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-handoff-'))
  const home: string = path.join(root, 'profile')
  const scriptDirectory: string = path.join(root, 'scripts', 'desktop-update')
  fs.mkdirSync(home)
  fs.mkdirSync(scriptDirectory, { recursive: true })
  fs.writeFileSync(path.join(scriptDirectory, IS_WINDOWS ? 'windows.ps1' : 'posix.sh'), '')
  fs.writeFileSync(path.join(scriptDirectory, 'runtime.ps1'), '')
  fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })
  fs.writeFileSync(path.join(root, '.hermes', 'bin', 'hermes.exe'), '')

  const status: SourceUpdate = { supported: true, branch: 'main', targetSha: 'a'.repeat(40), updateAvailable: true }

  const deps: CheckoutStrategyDeps = {
    readSourceUpdate: async (): Promise<SourceUpdate> => status,
    hermesHome: home,
    isWindows: IS_WINDOWS,
    isMac: process.platform === 'darwin',
    defaultUpdateBranch: 'main',
    updateHandoffDwellMs: 0,
    resolveUpdateRoot: (): string => root,
    resolveUpdaterBinary: (): null => null,
    remoteGatewayActive: (): boolean => remote,
    emitUpdateProgress: vi.fn(),
    rememberLog: vi.fn(),
    startHermes: vi.fn(async (): Promise<void> => {}),
    stopBackendsForUpdate: async (): Promise<void> => {},
    repairMacUpdaterHelper: (): void => {},
    preflightStateDb: (): void => {},
    runningAppBundle: (): null => null,
    markQuittingForHandoff: vi.fn(),
    quit: vi.fn()
  }

  return { root, deps }
}

// One gateway per host (#117529): a Desktop served by a remote gateway must
// tell the hand-off script not to (re)start a local one, and a locally-owned
// Desktop must keep the default so its gateway comes back after the update.
it.each([true, false])(
  'hand-off passes the no-gateway flag iff a remote gateway serves the app: %s',
  async (remote: boolean): Promise<void> => {
    const { root, deps } = handoffFixture(remote)
    const spawned: string[][] = []
    const spawnOptions: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2][] = []
    vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
      (_command: string, args: string[], options: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]): updaterProcess.UpdaterChild => {
        spawned.push(args)
        spawnOptions.push(options)

        return { unref: (): void => {} }
      }
    )

    try {
      expect(await createCheckoutStrategy(deps).apply()).toMatchObject({ ok: true, handedOff: true })
      expect(spawned).toHaveLength(1)
      const args: string[] = spawned[0]!
      // The Windows cmd wrapper must inherit its hidden console; the POSIX
      // script needs to outlive Electron as a detached child (#116161).
      expect(spawnOptions[0]?.detached).toBe(!IS_WINDOWS)
      expect(args).toContain(IS_WINDOWS ? '-Branch' : '--branch')

      if (remote) {
        expect(args).toContain(NO_GATEWAY_FLAG)
      } else {
        expect(args).not.toContain(NO_GATEWAY_FLAG)
      }
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)

// A hand-off that never became viable (#66753) must not quit into nothing:
// the app stays, the backend restarts, and the user reads plain copy with
// the raw spawn outcome confined to a Details line.
it('a failed hand-off spawn keeps the app alive and reports the failure in plain copy', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false)
  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation((): updaterProcess.UpdaterChild => {
    const child: EventEmitter & updaterProcess.UpdaterChild = Object.assign(new EventEmitter(), {
      unref: (): void => {}
    })

    queueMicrotask((): void => {
      child.emit('error', Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }))
    })

    return child
  })

  try {
    const result: Awaited<ReturnType<ReturnType<typeof createCheckoutStrategy>['apply']>> =
      await createCheckoutStrategy(deps).apply()

    expect(result).toMatchObject({ ok: false, error: 'updater-spawn-failed' })
    expect(result.message).toMatch(/Hermes keeps running/)
    expect(result.message).toMatch(/Details: .*ENOENT/)
    expect(result.message?.indexOf('Details:')).toBeGreaterThan(0)
    expect(deps.quit).not.toHaveBeenCalled()
    expect(deps.markQuittingForHandoff).not.toHaveBeenCalled()
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
