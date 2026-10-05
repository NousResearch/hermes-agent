import type { spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { markerPath, parseUpdateMarker } from '../update-marker'
import { claimantChild, cleanupMarkerFixtures, deadPid, killAndReap } from '../update-marker.test-helpers'
import * as updaterProcess from '../updater-process'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import type { SourceUpdate } from './checkout-source'
import { fakeHelperScript } from './marker-helper.test-helpers'

const IS_WINDOWS: boolean = process.platform === 'win32'
const NO_GATEWAY_FLAG: string = IS_WINDOWS ? '-NoGateway' : '--no-gateway'

afterEach((): void => {
  vi.restoreAllMocks()
  cleanupMarkerFixtures()
})

/** A checkout root with the repo hand-off script staged, plus a strategy over it. */
function handoffFixture(remote: boolean, scriptText: string = ''): { root: string; deps: CheckoutStrategyDeps } {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-handoff-'))
  const home: string = path.join(root, 'profile')
  const scriptDirectory: string = path.join(root, 'scripts', 'desktop-update')
  fs.mkdirSync(home)
  fs.mkdirSync(scriptDirectory, { recursive: true })
  fs.writeFileSync(path.join(scriptDirectory, IS_WINDOWS ? 'windows.ps1' : 'posix.sh'), scriptText)
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
    handoffClaimTimeoutMs: 2000,
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

/**
 * A legacy (pre-protocol-2) script's first act: it claims the marker in its
 * own name, echoing HERMES_UPDATE_STARTED_AT on line 2 (origin/main shape).
 * The vitest parent stands in for the live script process.
 */
function scriptTakesMarker(home: string, options: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]): void {
  fs.writeFileSync(markerPath(home), `${process.ppid}\n${options.env?.HERMES_UPDATE_STARTED_AT}\n`)
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
      (
        _command: string,
        args: string[],
        options: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]
      ): updaterProcess.UpdaterChild => {
        spawned.push(args)
        spawnOptions.push(options)
        scriptTakesMarker(deps.hermesHome, options)

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

// #103222: the Windows wrapper must own the hidden console the script shares.
// Spawned detached it has none, so `start /b` gives PowerShell a visible
// console whose QuickEdit selection stalls the hand-off before relaunch.
it('the Windows hand-off wrapper is spawned non-detached so the script shares its hidden console', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false)
  fs.writeFileSync(path.join(root, 'scripts', 'desktop-update', 'windows.ps1'), '')
  const resolveHandoff: typeof updaterProcess.resolveUpdateScriptHandoff = updaterProcess.resolveUpdateScriptHandoff
  vi.spyOn(updaterProcess, 'resolveUpdateScriptHandoff').mockImplementation(
    (updateRoot: string): updaterProcess.UpdateScriptHandoff | null => resolveHandoff(updateRoot, { isWindows: true })
  )
  const spawned: { command: string; args: string[]; detached: unknown }[] = []
  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
    (
      command: string,
      args: string[],
      options: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]
    ): updaterProcess.UpdaterChild => {
      spawned.push({ command, args, detached: options.detached })
      scriptTakesMarker(deps.hermesHome, options)

      return { unref: (): void => {} }
    }
  )

  try {
    expect(await createCheckoutStrategy({ ...deps, isWindows: true }).apply()).toMatchObject({ ok: true })
    expect(spawned).toHaveLength(1)
    expect(spawned[0]).toMatchObject({ command: 'cmd.exe', detached: false })
    expect(spawned[0]!.args.slice(0, 6)).toEqual(['/d', '/s', '/c', 'start', '', '/b'])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

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

// V7: the wrapper's exit 0 is not a hand-off. Without the script taking the
// marker the Desktop stays up, restarts its backend and reports the failure.
// A legacy script (no protocol line) gets no bridge at all (SPEC 4b).
it('a hand-off whose legacy script never takes the marker keeps the app alive and wrote no bridge', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false)
  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
    (): updaterProcess.UpdaterChild => ({ unref: (): void => {} })
  )

  try {
    const result = await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 300 }).apply()

    expect(result).toMatchObject({ ok: false, error: 'updater-spawn-failed' })
    expect(result.message).toMatch(/previous version/)
    expect(deps.quit).not.toHaveBeenCalled()
    expect(deps.markQuittingForHandoff).not.toHaveBeenCalled()
    // Windows stopped its backends before spawning; POSIX never did.
    expect(vi.mocked(deps.startHermes).mock.calls.length > 0).toBe(IS_WINDOWS)
    expect(fs.existsSync(markerPath(deps.hermesHome))).toBe(false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch {
    return false
  }
}

// MINOR-3: after the 20 s claim wait gives up, the UI says "the updater did
// not start". A script that starts late must not then run an update anyway:
// the Desktop withdraws its bridge AND kills the launcher tree it spawned.
it.skipIf(IS_WINDOWS)('a timed-out hand-off kills the real script tree it spawned', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false)
  const grandchildPidFile: string = path.join(root, 'late-script.pid')
  let launcherPid: number | undefined
  const realSpawn: typeof updaterProcess.spawnUpdaterProcess = updaterProcess.spawnUpdaterProcess

  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
    (_command: string, _args: string[], options: Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]) => {
      // A real launcher (detached, own process group) whose "script" is slow
      // to start: it would claim the marker 30 s later.
      const child = realSpawn(
        'bash',
        ['-c', `sleep 30 & echo $! > ${JSON.stringify(grandchildPidFile)}; wait`],
        options
      ) as ReturnType<typeof spawn>

      launcherPid = child.pid

      return child
    }
  )

  try {
    const result = await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 600 }).apply()

    expect(result).toMatchObject({ ok: false, error: 'updater-spawn-failed' })
    expect(fs.existsSync(markerPath(deps.hermesHome))).toBe(false)
    const latePid: number = Number(fs.readFileSync(grandchildPidFile, 'utf8').trim())
    const deadline: number = Date.now() + 3000

    while ((pidAlive(launcherPid!) || pidAlive(latePid)) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }

    expect(pidAlive(latePid)).toBe(false)
    expect(pidAlive(launcherPid!)).toBe(false)
  } finally {
    for (const pid of [launcherPid]) {
      try {
        process.kill(-pid!, 'SIGKILL')
      } catch {
        // gone
      }
    }

    fs.rmSync(root, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------------------
// Hand-off protocol 2 (SPEC 4a, A7 rules 3 and 6) — POSIX: the marker helper
// runs as the checkout's REAL (fake) script process.
// ---------------------------------------------------------------------------

type SpawnOptions = Parameters<typeof updaterProcess.spawnUpdaterProcess>[2]

function argAfter(args: string[], flag: string): string | undefined {
  const at = args.indexOf(flag)

  return at >= 0 ? args[at + 1] : undefined
}

function helperCalls(home: string): string[] {
  try {
    return fs.readFileSync(path.join(home, 'helper-calls.log'), 'utf8').trim().split('\n').filter(Boolean)
  } catch {
    return []
  }
}

interface Protocol2Run {
  args: string[]
  bridgeAtSpawn: string | null
}

/**
 * Spawn double for the protocol-2 hand-off: records the argv and the bridge
 * the Desktop published BEFORE spawning, then (per `script`) lets a REAL
 * claimant process adopt the run, adopt-and-die, or never show up.
 */
function mockProtocol2Spawn(home: string, script: 'adopts' | 'adopts-then-dies' | 'never'): Protocol2Run[] {
  const runs: Protocol2Run[] = []

  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
    (_command: string, args: string[], options: SpawnOptions): updaterProcess.UpdaterChild => {
      let bridgeAtSpawn: string | null = null

      try {
        bridgeAtSpawn = fs.readFileSync(markerPath(home), 'utf8')
      } catch {
        bridgeAtSpawn = null
      }

      runs.push({ args, bridgeAtSpawn })
      const run = argAfter(args, '--handoff-run')
      const startedAt = Number(options.env?.HERMES_UPDATE_STARTED_AT)

      if (script !== 'never') {
        void claimantChild(markerPath(home), { startedAt, run }).then(async ({ child }) => {
          if (script === 'adopts-then-dies') {
            await killAndReap(child)
          }
        })
      }

      return { unref: (): void => {} }
    }
  )

  return runs
}



describe.skipIf(IS_WINDOWS)('protocol 2 hand-off', () => {
  it('bridges with a run id, passes --handoff-run, and acknowledges the live adopting script', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    const runs = mockProtocol2Spawn(deps.hermesHome, 'adopts')

    try {
      expect(await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 10_000 }).apply()).toMatchObject({
        ok: true,
        handedOff: true
      })
      expect(runs).toHaveLength(1)
      const runId = argAfter(runs[0]!.args, '--handoff-run')!
      expect(runId).toMatch(new RegExp(`^desk-${process.pid}-[0-9a-z]+-[0-9a-f]{4}$`))
      const bridge = parseUpdateMarker(runs[0]!.bridgeAtSpawn ?? '')
      expect(bridge).toMatchObject({ pid: process.pid, run: runId })
      expect(bridge?.ct).not.toBeNull()
      expect(argAfter(runs[0]!.args, '--desktop-pid')).toBe(String(process.pid))
      expect(helperCalls(deps.hermesHome)).toEqual([])
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('R5: a script that adopted the run and died before the ack is a failed hand-off', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'withdrawn')
    const runs = mockProtocol2Spawn(deps.hermesHome, 'adopts-then-dies')

    try {
      const result = await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 1_500 }).apply()

      expect(result).toMatchObject({ ok: false, error: 'updater-spawn-failed' })
      expect(deps.quit).not.toHaveBeenCalled()
      const runId = argAfter(runs[0]!.args, '--handoff-run')!
      expect(helperCalls(deps.hermesHome)).toEqual([
        `--marker-op withdraw --install-root ${root} --desktop-pid ${process.pid} --handoff-run ${runId}`
      ])
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  // Review 5411223284 regression 1: a bridge the helper could not withdraw
  // stays adoptable by a late (setsid, unkillable) daemon. Reporting "did not
  // start" and restarting the backend would run `hermes update` beside it.
  it.each(['busy', 'error'])(
    'a withdraw that stays `%s` keeps the bridge and quits instead of restarting the backend',
    async (answer: string): Promise<void> => {
      const { root, deps } = handoffFixture(false, fakeHelperScript())
      fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), answer === 'error' ? 'garbage' : answer)
      const runs = mockProtocol2Spawn(deps.hermesHome, 'never')
      const kill = vi.spyOn(updaterProcess, 'killHandoffTree')

      try {
        const result = await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 400 }).apply()

        expect(result).toMatchObject({ ok: true, handedOff: true })
        expect(deps.startHermes).not.toHaveBeenCalled()
        expect(kill).not.toHaveBeenCalled()
        expect(deps.markQuittingForHandoff).toHaveBeenCalled()
        // Asked again while unsettled, and Electron left the bytes alone.
        expect(helperCalls(deps.hermesHome)).toHaveLength(3)
        expect(fs.readFileSync(markerPath(deps.hermesHome), 'utf8')).toBe(runs[0]!.bridgeAtSpawn)
      } finally {
        fs.rmSync(root, { recursive: true, force: true })
      }
    }
  )

  it('a definitive `withdrawn` reports the failure and kills the spawned tree', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'withdrawn')
    mockProtocol2Spawn(deps.hermesHome, 'never')
    const kill = vi.spyOn(updaterProcess, 'killHandoffTree').mockImplementation(() => {})

    try {
      const result = await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 400 }).apply()

      expect(result).toMatchObject({ ok: false, error: 'updater-spawn-failed' })
      expect(kill).toHaveBeenCalledTimes(1)
      expect(helperCalls(deps.hermesHome)).toHaveLength(1)
      expect(deps.quit).not.toHaveBeenCalled()
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('`taken <pid>` from the withdraw helper is a late success', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'taken 31337')
    mockProtocol2Spawn(deps.hermesHome, 'never')

    try {
      expect(await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 300 }).apply()).toMatchObject({
        ok: true,
        handedOff: true
      })
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('a dead marker is reclaimed by the script helper, then the bridge is created once', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    fs.writeFileSync(markerPath(deps.hermesHome), `${await deadPid()}\n${Math.floor(Date.now() / 1000)}\nct:1.000\n`)
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'reclaimed')
    const runs = mockProtocol2Spawn(deps.hermesHome, 'adopts')

    try {
      expect(await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 10_000 }).apply()).toMatchObject({
        ok: true
      })
      expect(helperCalls(deps.hermesHome)).toEqual([
        `--marker-op reclaim --install-root ${root} --desktop-pid ${process.pid}`
      ])
      expect(parseUpdateMarker(runs[0]!.bridgeAtSpawn ?? '')?.pid).toBe(process.pid)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('R6: a dead marker whose checkout lock is still `held` refuses the hand-off and is left in place', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    const body = `${await deadPid()}\n${Math.floor(Date.now() / 1000)}\nct:1.000\n`
    fs.writeFileSync(markerPath(deps.hermesHome), body)
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'held')
    const runs = mockProtocol2Spawn(deps.hermesHome, 'adopts')

    try {
      expect(await createCheckoutStrategy(deps).apply()).toMatchObject({ ok: false, error: 'update-already-running' })
      expect(runs).toHaveLength(0)
      expect(fs.readFileSync(markerPath(deps.hermesHome), 'utf8')).toBe(body)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('a stale bridge of THIS process is withdrawn with its own run before the new bridge', async (): Promise<void> => {
    const { root, deps } = handoffFixture(false, fakeHelperScript())
    const ct = (await import('../update-marker')).processCreateTime
    const own = await ct(process.pid)
    fs.writeFileSync(
      markerPath(deps.hermesHome),
      `${process.pid}\n${Math.floor(Date.now() / 1000)}\nct:${own!.toFixed(3)}\nrun:desk-old-run\n`
    )
    fs.writeFileSync(path.join(deps.hermesHome, 'helper-verdict'), 'withdrawn')
    const runs = mockProtocol2Spawn(deps.hermesHome, 'adopts')

    try {
      expect(await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 10_000 }).apply()).toMatchObject({
        ok: true
      })
      expect(helperCalls(deps.hermesHome)).toEqual([
        `--marker-op withdraw --install-root ${root} --desktop-pid ${process.pid} --handoff-run desk-old-run`
      ])
      expect(parseUpdateMarker(runs[0]!.bridgeAtSpawn ?? '')?.run).toBe(argAfter(runs[0]!.args, '--handoff-run'))
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})

// SPEC 4b: an old script (no protocol line) writes its own claim
// unconditionally and knows no run id: the Desktop writes no bridge and
// passes no run argument.
it.skipIf(IS_WINDOWS)('a legacy script gets no bridge and no --handoff-run', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false, '#!/usr/bin/env bash\necho legacy\n')
  const runs = mockProtocol2Spawn(deps.hermesHome, 'never')

  try {
    await createCheckoutStrategy({ ...deps, handoffClaimTimeoutMs: 200 }).apply()
    expect(runs).toHaveLength(1)
    expect(runs[0]!.bridgeAtSpawn).toBeNull()
    expect(runs[0]!.args).not.toContain('--handoff-run')
    expect(helperCalls(deps.hermesHome)).toEqual([])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

// Windows protocol 2: the run id rides the wrapped PowerShell argv.
it('the Windows protocol-2 hand-off passes -HandoffRun with the bridged run id', async (): Promise<void> => {
  const { root, deps } = handoffFixture(false)
  fs.writeFileSync(path.join(root, 'scripts', 'desktop-update', 'windows.ps1'), '# hermes-handoff-protocol: 2\r\n')
  const resolveHandoff: typeof updaterProcess.resolveUpdateScriptHandoff = updaterProcess.resolveUpdateScriptHandoff
  vi.spyOn(updaterProcess, 'resolveUpdateScriptHandoff').mockImplementation(
    (updateRoot: string): updaterProcess.UpdateScriptHandoff | null => resolveHandoff(updateRoot, { isWindows: true })
  )
  const helper = vi.fn(async () => ({ kind: 'withdrawn' as const }))
  const runs: string[][] = []
  let bridge: string | null = null
  vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation((_command: string, args: string[]) => {
    runs.push(args)
    bridge = fs.readFileSync(markerPath(deps.hermesHome), 'utf8')

    return { unref: (): void => {} }
  })

  try {
    await createCheckoutStrategy({ ...deps, isWindows: true, markerHelper: helper, handoffClaimTimeoutMs: 200 }).apply()
    const runId = argAfter(runs[0]!, '-HandoffRun')
    expect(runId).toMatch(/^desk-\d+-[0-9a-z]+-[0-9a-f]{4}$/)
    expect(parseUpdateMarker(bridge ?? '')?.run).toBe(runId)
    expect(argAfter(runs[0]!, '-DesktopPid')).toBe(String(process.pid))
    // Nothing adopted it: the bridge is withdrawn through the helper, with the run.
    expect(helper).toHaveBeenCalledWith('withdraw', expect.objectContaining({ runId, desktopPid: process.pid, isWindows: true }))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
