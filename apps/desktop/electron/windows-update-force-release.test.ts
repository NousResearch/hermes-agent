/**
 * Unit suite for the force-release orchestration: pure logic on fake clocks,
 * fake process tables and generated-script contracts. Anything that spawns a
 * real process lives in windows-update-force-release.windows-live.test.ts, so
 * a green run here never implies the Windows boundary was exercised.
 */

import assert from 'node:assert/strict'

import { describe, it, vi } from 'vitest'

import {
  buildExactTerminateScript,
  parseTerminateScriptOutput,
  TERMINATE_JOB_WATCHER_BRIDGE,
  TERMINATE_JOB_WATCHER_COMMAND,
  TERMINATE_JOB_WRAPPER_COMMAND,
  terminateWindowsHolderWithinDeadline
} from './windows-process-terminate'
import { buildRestartManagerScript, parseRestartManagerOutput } from './windows-restart-manager'
import {
  attachHolderTreeRelationships,
  type ForceReleaseHolder,
  type ForceReleaseTerminateResult,
  formatHolderLine,
  mergeInstallHolders,
  orderHoldersLeafFirst,
  raceWithBudget,
  runWindowsUpdateForceRelease,
  type WindowsUpdateForceReleaseDeps
} from './windows-update-force-release'

const holder = (overrides: Partial<ForceReleaseHolder> = {}): ForceReleaseHolder => ({
  pid: 57012,
  createdAt: 1_700_000_000,
  name: 'hermes.exe',
  cmdline: 'hermes.exe tools',
  source: 'scanner',
  ...overrides
})

function makeDeps(
  overrides: Partial<WindowsUpdateForceReleaseDeps> = {}
): {
  calls: string[]
  deps: WindowsUpdateForceReleaseDeps
  setLocked: (locked: boolean) => void
} {
  const calls: string[] = []
  let locked = true
  let clock = 0

  const deps: WindowsUpdateForceReleaseDeps = {
    now: () => clock,
    wait: async ms => {
      calls.push(`wait:${ms}`)
      clock += Math.max(0, ms)
    },
    isResourceLocked: async () => locked,
    listScannerHolders: async () => {
      calls.push('scan')

      return []
    },
    listRestartManagerHolders: async () => {
      calls.push('rm')

      return []
    },
    terminateHolder: async target => {
      calls.push(`terminate:${target.pid}:${target.createdAt}`)

      return { kind: 'terminated' }
    },
    excludePids: new Set([42]),
    deadlineMs: 5_000,
    settleMs: 0,
    ...overrides
  }

  return {
    calls,
    deps,
    setLocked: next => {
      locked = next
    }
  }
}

describe('orderHoldersLeafFirst', () => {
  it('terminates workers before their wrappers and children before parents', () => {
    const wrapper = holder({ pid: 10, role: 'wrapper' })
    const worker = holder({ pid: 11, wrapperPid: 10, role: 'worker' })
    const child = holder({ pid: 12, parentPid: 11, role: 'worker' })
    const ordered = orderHoldersLeafFirst([wrapper, worker, child])

    assert.deepEqual(
      ordered.map(entry => entry.pid),
      [12, 11, 10]
    )
  })
})

describe('mergeInstallHolders', () => {
  it('dedupes by pid+createdAt and unions scanner with Restart Manager evidence', () => {
    const fromScan = holder({ pid: 7, createdAt: 100, source: 'scanner', resource: 'venv\\Scripts\\hermes.exe' })

    const fromRm = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'venv\\Lib\\site-packages\\foo.pyd'
    })

    const other = holder({ pid: 8, createdAt: 200, source: 'restart-manager' })

    const merged = mergeInstallHolders([fromScan, fromRm, other])

    assert.equal(merged.length, 2)
    const first = merged.find(entry => entry.pid === 7)
    assert.ok(first)
    assert.equal(first.source, 'scanner')
    assert.match(String(first.resource), /foo\.pyd/)
  })

  it('keeps one Restart Manager path as the ownership proof and lists the rest as evidence', () => {
    // 2026-09-04: the merged "hermes.exe; python.exe" string was handed to
    // the exact-terminate script as a path, which could never verify it.
    const fromScan = holder({ pid: 7, createdAt: 100.25, source: 'scanner', role: 'wrapper' })

    const shim = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'C:\\hermes\\venv\\Scripts\\python.exe'
    })

    const pyd = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'C:\\hermes\\venv\\Lib\\site-packages\\yaml\\_yaml.pyd'
    })

    const [merged] = mergeInstallHolders([fromScan, shim, pyd])

    assert.ok(merged)
    assert.equal(merged.resource, 'C:\\hermes\\venv\\Scripts\\python.exe')
    assert.deepEqual(merged.resources, [
      'C:\\hermes\\venv\\Scripts\\python.exe',
      'C:\\hermes\\venv\\Lib\\site-packages\\yaml\\_yaml.pyd'
    ])
    assert.equal(merged.role, 'wrapper')
    assert.equal(merged.createdAt, 100.25)
    assert.match(formatHolderLine(merged), /python\.exe; .*_yaml\.pyd/)
  })

  it('carries the scanner unit routing through a merge', () => {
    const service = holder({ pid: 9, createdAt: 50, source: 'scanner', terminateVia: 'desktop-plugin-service' })
    const rm = holder({ pid: 9, createdAt: 50, source: 'restart-manager', resource: 'C:\\hermes\\venv\\Scripts\\python.exe' })

    const [merged] = mergeInstallHolders([rm, service])

    assert.equal(merged?.terminateVia, 'desktop-plugin-service')
    assert.equal(merged?.resource, 'C:\\hermes\\venv\\Scripts\\python.exe')
  })

  it('drops excluded helper and desktop PIDs', () => {
    const merged = mergeInstallHolders([holder({ pid: 42 }), holder({ pid: 99 })], new Set([42]))
    assert.deepEqual(
      merged.map(entry => entry.pid),
      [99]
    )
  })
})

describe('runWindowsUpdateForceRelease', () => {
  it('returns clear immediately when the install resources are already unlocked', async () => {
    const { calls, deps, setLocked } = makeDeps()
    setLocked(false)

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [])
  })

  it('force-drains the hermes tools|head orphan class via scanner holders within five seconds', async () => {
    const orphan = holder({
      pid: 57012,
      createdAt: 1_700_000_123.5,
      name: 'hermes.exe',
      cmdline: 'hermes.exe tools',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    const child = holder({
      pid: 57099,
      createdAt: 1_700_000_124,
      parentPid: 57012,
      name: 'python.exe',
      cmdline: 'python.exe -m hermes_cli.main tools'
    })

    let locked = true
    const terminated = new Set<number>()

    const { calls, deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => {
        calls.push('scan')

        return [orphan, child].filter(entry => !terminated.has(entry.pid))
      },
      listRestartManagerHolders: async () => {
        calls.push('rm')

        return []
      },
      terminateHolder: async target => {
        calls.push(`terminate:${target.pid}`)
        terminated.add(target.pid)

        if (terminated.has(orphan.pid) && terminated.has(child.pid)) {
          locked = false
        }

        return { kind: 'terminated' }
      },
      settleMs: 10
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.ok(calls.indexOf('terminate:57099') < calls.indexOf('terminate:57012'), 'leaf before root')
    assert.ok(calls.includes('scan'))
    assert.ok((deps.now?.() ?? 0) <= 5_000)
  })

  it('refuses PID reuse when create-time no longer matches', async () => {
    const stale = holder({ pid: 88, createdAt: 111 })

    const { deps } = makeDeps({
      listScannerHolders: async () => [stale],
      terminateHolder: async () => ({ kind: 'create-time-mismatch' })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.match(outcome.message, /create-time|PID reuse|no longer matches/i)
    }
  })

  it('escalates to elevation when OpenProcess/TerminateProcess returns access denied', async () => {
    const elevatedTarget = holder({ pid: 901, createdAt: 222, name: 'python.exe' })

    const { deps } = makeDeps({
      listScannerHolders: async () => [elevatedTarget],
      terminateHolder: async () => ({ kind: 'access-denied', win32Error: 5 })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'needs-elevation')

    if (outcome.kind === 'needs-elevation') {
      assert.equal(outcome.holders[0]?.pid, 901)
      assert.match(outcome.message, /Administrator|elevat/i)
      assert.match(outcome.message, /close the listed processes/i)
      assert.match(outcome.message, /retry the update/i)
      assert.doesNotMatch(outcome.message, /choose force update/i)
    }
  })

  it('surfaces protected-process terminal failure without claiming the venv is clear', async () => {
    const protectedHolder = holder({ pid: 77, createdAt: 333, resource: 'venv\\python.exe' })

    const { deps } = makeDeps({
      listScannerHolders: async () => [protectedHolder],
      terminateHolder: async () => ({ kind: 'protected', win32Error: 5 })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.match(outcome.message, /PID 77/)
      assert.match(outcome.message, /protected|unkillable|Win32/i)
    }
  })

  it('excludes the active updater helper and desktop main process', async () => {
    const helper = holder({ pid: 4242, createdAt: 1 })
    const desktop = holder({ pid: 42, createdAt: 2 })
    const real = holder({ pid: 99, createdAt: 3 })
    let locked = true
    const terminated: number[] = []

    const { deps } = makeDeps({
      excludePids: new Set([42, 4242]),
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [helper, desktop, real],
      terminateHolder: async target => {
        terminated.push(target.pid)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(terminated, [99])
  })

  it('stops the quick path within five seconds even when holders respawn', async () => {
    const zombie = holder({ pid: 55, createdAt: 9 })

    const { deps } = makeDeps({
      listScannerHolders: async () => [zombie],
      terminateHolder: async () => ({ kind: 'terminated' } satisfies ForceReleaseTerminateResult),
      settleMs: 2_000,
      deadlineMs: 5_000
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.ok(outcome.kind === 'timeout' || outcome.kind === 'blocked' || outcome.kind === 'needs-elevation')
    assert.ok((deps.now?.() ?? 0) <= 5_000)
  })

  it('uses Restart Manager holders when the scanner is empty but the shim stays locked', async () => {
    const rmOnly = holder({
      pid: 606,
      createdAt: 404,
      source: 'restart-manager',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    let locked = true
    const terminated: number[] = []

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [],
      listRestartManagerHolders: async () => [rmOnly],
      terminateHolder: async target => {
        terminated.push(target.pid)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(terminated, [606])
  })

  it('queries Restart Manager once and reuses it while the scanner keeps naming holders', async () => {
    // Restart Manager costs about 8 s on a loaded host; re-running it every
    // pass was most of the 20 s budget on 2026-09-04.
    const scanned = holder({ pid: 700, createdAt: 500, source: 'scanner' })
    const rm = holder({ pid: 700, createdAt: 500, source: 'restart-manager', resource: 'C:\\hermes\\venv\\Scripts\\python.exe' })
    let locked = true
    let rmCalls = 0
    let attempts = 0

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [scanned],
      listRestartManagerHolders: async () => {
        rmCalls += 1

        return [rm]
      },
      terminateHolder: async target => {
        attempts += 1
        assert.equal(target.resource, rm.resource, 'RM evidence from the first pass still attributes the holder')

        if (attempts < 2) {
          return { kind: 'failed', detail: 'first attempt lost the race' }
        }

        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease({ ...deps, deadlineMs: 5_000, settleMs: 1 })

    assert.equal(outcome.kind, 'clear')
    assert.equal(attempts, 2)
    assert.equal(rmCalls, 1)
  })

  it('reports each discovery pass and termination outcome for forensics', async () => {
    const scanned = holder({ pid: 701, createdAt: 501, source: 'scanner' })
    let locked = true
    const discoveries: number[] = []
    const outcomes: string[] = []

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [scanned],
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => {
        locked = false

        return { kind: 'terminated' }
      },
      onDiscovery: info => {
        discoveries.push(info.scanner)
      },
      onHolderOutcome: (target, result) => {
        outcomes.push(`${target.pid}:${result.kind}`)
      }
    })

    const outcome = await runWindowsUpdateForceRelease({ ...deps, deadlineMs: 5_000, settleMs: 1 })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(discoveries, [1])
    assert.deepEqual(outcomes, ['701:terminated'])
  })

  it('never reports clear while any verified holder remains', async () => {
    const survivor = holder({ pid: 1, createdAt: 2 })
    const terminate = vi.fn(async (): Promise<ForceReleaseTerminateResult> => ({ kind: 'terminated' }))

    const { deps } = makeDeps({
      listScannerHolders: async () => [survivor],
      terminateHolder: terminate,
      settleMs: 0,
      deadlineMs: 100
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.notEqual(outcome.kind, 'clear')
  })

  it('enforces a hard wall-clock budget when dependencies hang past five seconds', async () => {
    const started = Date.now()

    const hung = () =>
      new Promise<ForceReleaseHolder[]>(resolve => {
        setTimeout(() => resolve([holder({ pid: 1, createdAt: 2 })]), 8_000)
      })

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 400,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => hung(),
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => ({ kind: 'terminated' })
    })

    const elapsed = Date.now() - started
    assert.notEqual(outcome.kind, 'clear')
    assert.ok(elapsed < 2_000, `elapsed ${elapsed}ms must stay near the 400ms budget`)
    assert.ok(elapsed >= 300, `elapsed ${elapsed}ms should wait roughly the budget`)
  })

  it('aborts and drains termination at the outer absolute deadline', async () => {
    const target = holder({ pid: 71, createdAt: 72 })
    let aborted = false
    let settled = false
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 80,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => [target],
      listRestartManagerHolders: async () => [],
      terminateHolder: async (_holder, _budget, signal) => {
        await new Promise<void>(resolve => {
          signal?.addEventListener('abort', () => {
            aborted = true
            resolve()
          }, { once: true })
        })
        settled = true

        return { kind: 'failed', detail: 'aborted-and-drained' }
      }
    })

    const elapsed = Date.now() - started
    assert.equal(outcome.kind, 'timeout')
    assert.equal(aborted, true)
    assert.equal(settled, true, 'termination was terminal before the API returned')
    assert.ok(elapsed < 180, `elapsed ${elapsed}ms exceeded the 80ms absolute deadline envelope`)
  })

  it('bounds a stalled lock probe at the same absolute deadline', async () => {
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 80,
      settleMs: 0,
      isResourceLocked: async () => {
        await new Promise(resolve => setTimeout(resolve, 250))

        return false
      },
      listScannerHolders: async () => [],
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => ({ kind: 'terminated' })
    })

    const elapsed = Date.now() - started
    assert.equal(outcome.kind, 'timeout')
    assert.ok(elapsed < 180, `elapsed ${elapsed}ms exceeded the 80ms absolute deadline envelope`)
  })

  it('passes remaining budget into terminateHolder', async () => {
    const budgets: number[] = []
    const target = holder({ pid: 3, createdAt: 4 })
    let locked = true

    const { deps } = makeDeps({
      deadlineMs: 1_000,
      settleMs: 0,
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [target],
      terminateHolder: async (_holder, budgetMs) => {
        budgets.push(budgetMs)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)
    assert.equal(outcome.kind, 'clear')
    assert.ok(budgets.length >= 1)
    assert.ok(budgets[0]! <= 1_000)
    assert.ok(budgets[0]! > 0)
  })
})

describe('raceWithBudget', () => {
  it('returns the timeout fallback when work exceeds the budget', async () => {
    const started = Date.now()

    const value = await raceWithBudget(
      new Promise<string>(resolve => setTimeout(() => resolve('late'), 1_000)),
      100,
      () => 'fallback'
    )

    const elapsed = Date.now() - started
    assert.equal(value, 'fallback')
    assert.ok(elapsed < 500)
  })
})

describe('restart manager script contract', () => {
  it('emits a literal pipe split that does not over-escape regex', () => {
    const script = buildRestartManagerScript('C:\\h\\rm-list.txt')
    assert.match(script, /\$part\.Split\(\[char\]'\|', 4\)/)
    assert.doesNotMatch(script, /\$part -split '\\\\\|'/)
    assert.doesNotMatch(script, /\$part -split '\\\|'/)
    assert.match(script, /StringBuilder\(CCH_RM_SESSION_KEY \+ 1\)/)
    // The row split used to be "proved" by asserting the exported constant
    // equals its own literal. The generated script is now run end-to-end
    // against a real holder in windows-restart-manager.windows-live.test.ts,
    // where an over-escaped split yields an empty holder list.
  })

  it('parses RM JSON rows into force-release holders', () => {
    const holders = parseRestartManagerOutput(
      JSON.stringify([{ pid: 12, createdAt: 34, name: 'python.exe' }]),
      ['C:\\h\\venv\\Scripts\\hermes.exe']
    )

    assert.equal(holders.length, 1)
    assert.equal(holders[0]?.source, 'restart-manager')
    assert.equal(holders[0]?.pid, 12)
    assert.match(String(holders[0]?.resource), /hermes\.exe/)
  })
})

describe('terminate script output parser', () => {
  it('classifies create-time mismatch, access denied, and protected', () => {
    assert.deepEqual(parseTerminateScriptOutput('CREATE_TIME_MISMATCH actual=1 expected=2', 3), {
      kind: 'create-time-mismatch'
    })
    assert.deepEqual(parseTerminateScriptOutput('ACCESS_DENIED', 5), {
      kind: 'access-denied',
      win32Error: 5
    })
    assert.deepEqual(parseTerminateScriptOutput('PROTECTED win32=5', 5), {
      kind: 'protected',
      win32Error: 5
    })
    assert.deepEqual(parseTerminateScriptOutput('TERMINATED', 0), { kind: 'terminated' })
    assert.deepEqual(parseTerminateScriptOutput('ALREADY_GONE', 0), { kind: 'already-gone' })
    assert.equal(parseTerminateScriptOutput('FAILED win32=87', 1).kind, 'failed')
    assert.equal(parseTerminateScriptOutput('FAILED win32=6', 1).kind, 'failed')
  })
})

describe('target watcher transport boundary', () => {
  it('requires a nonce-bound wrapper marker and fail-closed watcher script handling', () => {
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PHASE_PATH/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PHASE_NONCE/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /marker-published|target-job-created|watcher-READY-observed/)
    assert.match(TERMINATE_JOB_WATCHER_BRIDGE, /spawnSync/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /\$ErrorActionPreference = 'Stop'/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /Add-Type -TypeDefinition @'[\s\S]*'@ -ErrorAction Stop/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /exit \$watchResult/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /FAILED:' \+ \$watcherReadyNonce/)
  })


})

describe('merge create-time tolerance and production leaf-first', () => {
  it('dedupes scanner fractional create-time with RM integer seconds', () => {
    const fromScan = holder({
      pid: 7,
      createdAt: 100.4,
      source: 'scanner',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    const fromRm = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'venv\\Scripts\\python.exe'
    })

    const merged = mergeInstallHolders([fromScan, fromRm])
    assert.equal(merged.length, 1)
    assert.equal(merged[0]?.pid, 7)
    // One exact path for the ownership proof (Restart Manager wins); the
    // scanner claim stays visible as evidence only.
    assert.equal(merged[0]?.resource, 'venv\\Scripts\\python.exe')
    assert.deepEqual(merged[0]?.resources, ['venv\\Scripts\\hermes.exe', 'venv\\Scripts\\python.exe'])
  })

  it('orders production generic holders leaf-first when parentPid evidence is present', () => {
    const root = holder({ pid: 10, createdAt: 1, role: 'other' })
    const child = holder({ pid: 11, createdAt: 2, parentPid: 10, role: 'other' })
    const ordered = orderHoldersLeafFirst(attachHolderTreeRelationships([root, child]))
    assert.deepEqual(
      ordered.map(entry => entry.pid),
      [11, 10]
    )
  })
})

describe('termination cancellation hard boundary', () => {
  it('budget-owned terminate returns without scheduling post-return mutation', async () => {
    const target = holder({ pid: 404, createdAt: 1 })
    let mutated = false
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 400,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => [target],
      listRestartManagerHolders: async () => [],
      terminateHolder: async (_holder, budget) => {
        // Production contract: terminate owns the budget and must not leave
        // pending mutation work that fires after it returns.
        const slice = Math.max(1, Math.min(budget, 80))
        await new Promise(resolve => setTimeout(resolve, slice))

        // A cancelled late arm would mutate if it were left live.
        const late = setTimeout(() => {
          mutated = true
        }, 600)

        clearTimeout(late)

        return { kind: 'failed', detail: 'deadline-exhausted' }
      }
    })

    const elapsed = Date.now() - started
    assert.notEqual(outcome.kind, 'clear')
    await new Promise(resolve => setTimeout(resolve, 700))
    assert.equal(mutated, false)
    assert.ok(elapsed < 2_000, `elapsed ${elapsed}`)
  })

  it(
    'maps multiple near-expiry holders through one production absolute deadline',
    { timeout: 10_000 },
    async () => {
      const holders = [
        holder({ pid: 701, createdAt: 1_701_000_001 }),
        holder({ pid: 702, createdAt: 1_701_000_002 })
      ]

      const terminated = new Set<number>()
      const terminationCalls: Array<{ pid: number; budgetMs: number; deadlineAt?: number }> = []
      let scanPass = 0
      let runCalls = 0
      let lateMutations = 0
      const started = Date.now()

      const fakeRun = async (
        _script: string,
        timeoutMs = 0,
        signal?: AbortSignal,
        nativeDeadlineAt?: number
      ): Promise<{ stdout: string; stderr: string; code: number }> => {
        const call = ++runCalls
        // Leave a real margin for the outer kill/confirmation reserve while
        // still making the second holder genuinely near-expiry.
        const delayMs = call === 1 ? 4_250 : 500
        let aborted = false

        const lateMutation = setTimeout(() => {
          lateMutations += 1
        }, delayMs + 250)

        await new Promise<void>(resolve => {
          const timer = setTimeout(resolve, delayMs)

          const onAbort = () => {
            aborted = true
            clearTimeout(timer)
            resolve()
          }

          if (signal?.aborted) {onAbort()}
          else {signal?.addEventListener('abort', onAbort, { once: true })}
        })
        clearTimeout(lateMutation)

        if (
          aborted ||
          signal?.aborted ||
          (typeof nativeDeadlineAt === 'number' && Date.now() >= nativeDeadlineAt)
        ) {
          return { stdout: '', stderr: `aborted timeout=${timeoutMs}`, code: 1 }
        }

        if (call === 1) {return { stdout: 'TERMINATED', stderr: '', code: 0 }}

        return { stdout: '', stderr: 'near-expiry failure', code: 1 }
      }

      const outcome = await runWindowsUpdateForceRelease({
        deadlineMs: 5_000,
        settleMs: 0,
        isResourceLocked: async () => terminated.size < holders.length,
        listScannerHolders: async () => {
          scanPass += 1

          return scanPass === 1 ? holders : []
        },
        listRestartManagerHolders: async () => [],
        // Keep this callback identical to the production mapping in main.ts:
        // one remaining budget and one absolute deadline reach the native seam.
        terminateHolder: async (target, budgetMs, signal, deadlineAt) => {
          terminationCalls.push({ pid: target.pid, budgetMs, deadlineAt })

          const result = await terminateWindowsHolderWithinDeadline(target, {
            platform: 'win32',
            budgetMs,
            deadlineAt: deadlineAt ?? Date.now(),
            signal,
            run: fakeRun
          })

          if (result.kind === 'terminated' || result.kind === 'already-gone') {
            terminated.add(target.pid)
          }

          return result
        }
      })

      const elapsed = Date.now() - started

      assert.equal(outcome.kind, 'timeout')
      assert.equal(runCalls, 2, 'both near-expiry holders must reach the bounded seam')
      assert.equal(terminationCalls.length, 2)
      assert.equal(
        new Set(terminationCalls.map(call => call.deadlineAt)).size,
        1,
        'holders must share one absolute deadline'
      )
      const first = terminationCalls[0]
      const second = terminationCalls[1]
      assert.ok(first && second)
      assert.ok(first.budgetMs <= 5_000)
      assert.ok(second.budgetMs < 900, `second holder was not near expiry: ${second.budgetMs}`)
      assert.ok(second.budgetMs > 450, `test did not exercise a meaningful second budget: ${second.budgetMs}`)
      assert.ok(typeof first.deadlineAt === 'number')
      assert.ok((first.deadlineAt as number) - started <= 5_100)
      assert.ok(elapsed <= 5_100, `production-mapped force release elapsed ${elapsed}`)

      await new Promise(resolve => setTimeout(resolve, 800))
      assert.equal(lateMutations, 0, 'near-expiry mutation fired after the updater returned')
    }
  )


})

describe('liveness probe classification', () => {
  it('classifies discriminated exit vs error outcomes without host PIDs', async () => {
    const {
      classifyLivenessProbeResult,
      probeProcessLiveness,
      identitiesStillPresent
    } = await import('./windows-process-terminate')

    // Authenticated exits only.
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 0 }), 'live')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 3 }), 'absent')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 1 }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 99 }), 'unknown')

    // Error metadata never proves absence — even numeric/string 3.
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 3 }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: '3' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'ETIMEDOUT' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'EACCES' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'EPERM' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'ENOENT' }), 'unknown')
    assert.equal(
      classifyLivenessProbeResult({ kind: 'error', message: 'spawn powershell ENOENT' }),
      'unknown'
    )

    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'exit', code: 0 })),
      'live'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'exit', code: 3 })),
      'absent'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'ETIMEDOUT' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'EACCES' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'ENOENT' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 3 })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: '3' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 0, async () => ({ kind: 'exit', code: 0 })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(-1, 1_000, async () => ({ kind: 'exit', code: 0 })),
      'absent'
    )

    // identitiesStillPresent retains every unknown; drops only explicit exit-3 absent.
    // Force null create-time so the injectable liveness runner is exercised, not host PIDs.
    const noCreateTime = async () => null
    const unknownRunner = async () => ({ kind: 'error' as const, code: 'ETIMEDOUT' as const })

    const kept = await identitiesStillPresent([{ pid: 11 }, { pid: 12 }], {
      deadlineAt: Date.now() + 5_000,
      livenessRunner: unknownRunner,
      readCreatedAt: noCreateTime
    })

    assert.deepEqual(
      kept.map(entry => entry.pid),
      [11, 12]
    )

    const errorThreeRunner = async () => ({ kind: 'error' as const, code: 3 as const })

    const errorThreeKept = await identitiesStillPresent([{ pid: 31 }, { pid: 32 }], {
      deadlineAt: Date.now() + 5_000,
      livenessRunner: errorThreeRunner,
      readCreatedAt: noCreateTime
    })

    assert.deepEqual(
      errorThreeKept.map(entry => entry.pid),
      [31, 32]
    )

    const mixedRunner = async (pid: number) => {
      if (pid === 21) {return { kind: 'exit' as const, code: 0 }}

      if (pid === 22) {return { kind: 'exit' as const, code: 3 }}

      if (pid === 23) {return { kind: 'error' as const, code: 'EPERM' }}

      if (pid === 24) {return { kind: 'error' as const, code: 3 }}

      return { kind: 'error' as const, code: 'ENOENT' }
    }

    const mixed = await identitiesStillPresent(
      [{ pid: 21 }, { pid: 22 }, { pid: 23 }, { pid: 24 }, { pid: 25 }],
      {
        deadlineAt: Date.now() + 5_000,
        livenessRunner: mixedRunner,
        readCreatedAt: noCreateTime
      }
    )

    assert.deepEqual(
      mixed.map(entry => entry.pid).sort((a, b) => a - b),
      [21, 23, 24, 25]
    )
  })
})

describe('access-denied identity classification', () => {
  it('does not expose ambient test controls in the production termination script', () => {
    const script = buildExactTerminateScript(9, 100, 100)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_FORCE_SNAPSHOT_FAILURE/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PAUSE_PHASE/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PAUSE_PID/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PHASE_MARKER/)
  })

  it('embeds final install-root and current Restart Manager ownership authorization', () => {
    const script = buildExactTerminateScript(4242, 1234, 500, {
      installRoot: 'C:\\Hermes',
      resource: 'C:\\Hermes\\venv\\Scripts\\hermes.exe'
    })

    assert.match(script, /TERMINATION_RESOURCE_OUTSIDE_INSTALL_ROOT/)
    assert.match(script, /TERMINATION_EXECUTABLE_IDENTITY_UNAVAILABLE/)
    assert.match(script, /TERMINATION_CURRENT_LOCK_OWNERSHIP_MISMATCH/)
    assert.match(script, /QueryFullProcessImageName/)
    assert.match(script, /GetFinalPathNameByHandle/)
    assert.match(script, /IsSameOrUnderRoot\(\$resourceFinal, \$installRootFinal\)/)
    assert.match(script, /IsSameOrUnderRoot\(\$imageFinal, \$installRootFinal\)/)
    assert.match(script, /TERMINATION_EXECUTABLE_OUTSIDE_INSTALL_ROOT/)
    assert.match(script, /IsCurrentResourceOwner/)
    assert.match(script, /RmRegisterResources/)
    assert.match(script, /C:\\Hermes\\venv\\Scripts\\hermes\.exe/)
    assert.match(script, /\(\$childExpected \+ 1\.5\) -lt \$parentCreated/)
  })

  it('classifies access-denied identity failures for Administrator routing', () => {
    assert.deepEqual(parseTerminateScriptOutput('ACCESS_DENIED', 5), {
      kind: 'access-denied',
      win32Error: 5
    })
    const script = buildExactTerminateScript(9, 100, 100)
    assert.match(script, /HOLDER_OPEN_FAILED win32=5/)
    assert.match(script, /ACCESS_DENIED/)
    assert.doesNotMatch(script, /\$win32 -eq '5'/)
  })

  it('does not classify generic Job failures as Administrator escalation', () => {
    assert.deepEqual(parseTerminateScriptOutput('BOUNDARY_FAILED TREE_ASSIGN_FAILED win32=5', 5), {
      kind: 'failed',
      detail: 'BOUNDARY_FAILED TREE_ASSIGN_FAILED win32=5',
      win32Error: 5
    })
  })
})
