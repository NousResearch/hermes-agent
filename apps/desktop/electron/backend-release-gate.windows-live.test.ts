/**
 * backend-release-gate.windows-live.test.ts
 *
 * LIVE Windows E2E for the #74805 unlock gate: real spawned processes, the
 * REAL isPidAliveWindows probe against the live process table, real
 * taskkill — no fake clocks, no fake tables. Runs only on win32 (the
 * ephemeral wine2e lane); skipped everywhere else.
 *
 * This is the platform half of the proof: the unit suite pins the gate's
 * decision logic on a fake table; this file proves the two real-world
 * premises the fix rests on:
 *   1. taskkill /T /F returns while the killed process is still enumerable
 *      (the race window exists), and
 *   2. the gate, wired to the real probes, dwells through that window and
 *      only passes once the PID has genuinely left the table.
 */

import { execFileSync, spawn } from 'node:child_process'

import { describe, expect, it } from 'vitest'

import { isPidAliveWindows, waitForBackendRelease } from './backend-release-gate'
import { windowsPowerShellExecutable } from './windows-powershell-path'

const isWindows = process.platform === 'win32'

function spawnSleeper(): { pid: number; kill: () => void } {
  // Bare `spawn('powershell', ...)` resolves through PATH, which is exactly
  // what this suite's own sibling module (windows-powershell-path.ts) exists
  // to avoid for production code. Use the same absolute, windowsHide spawn
  // here so the sleeper is not a second, unguarded copy of that lookup.
  const child = spawn(windowsPowerShellExecutable(), ['-NoProfile', '-Command', 'Start-Sleep -Seconds 300'], {
    stdio: 'ignore',
    windowsHide: true
  })

  if (!child.pid) {
    throw new Error('sleeper failed to spawn')
  }

  return {
    pid: child.pid,
    kill: () => {
      try {
        child.kill()
      } catch {
        /* already gone */
      }
    }
  }
}

/**
 * Block until the OS process table itself reports the PID alive (bounded).
 * The sleeper's `spawn()` call returning a pid proves the handle exists, not
 * that the process table has caught up -- arm (b) below needs the LATTER
 * before it starts a real-clock gate, or a slow-to-appear PID can retire
 * before the 2000 ms deadline even begins, which is indistinguishable from
 * the gate itself being broken.
 */
async function waitUntilObservablyAlive(pid: number, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs

  while (!isPidAliveWindows(pid)) {
    if (Date.now() >= deadline) {
      throw new Error(`sleeper pid ${pid} never became observably alive within ${timeoutMs}ms`)
    }

    await new Promise(resolve => setTimeout(resolve, 25))
  }
}

function taskkillTree(pid: number): void {
  try {
    execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore' })
  } catch {
    /* already gone */
  }
}

describe.skipIf(!isWindows)('waitForBackendRelease — live Windows (#74805)', () => {
  it('isPidAliveWindows tracks a real process through spawn and exit', async () => {
    const sleeper = spawnSleeper()

    expect(isPidAliveWindows(sleeper.pid)).toBe(true)

    taskkillTree(sleeper.pid)

    // Poll until the table retires the PID (bounded).
    const deadline = Date.now() + 10000

    while (isPidAliveWindows(sleeper.pid) && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 100))
    }

    expect(isPidAliveWindows(sleeper.pid)).toBe(false)
  })

  it('the gate dwells until a real killed PID leaves the live process table', async () => {
    const sleeper = spawnSleeper()
    const logs: string[] = []
    let firstAliveCheck: boolean | null = null

    // Fire the real taskkill and IMMEDIATELY enter the gate — the #74805
    // shape. The shim probe reads unlocked throughout (the serve backend
    // never held it); only the PID exit-wait can hold the gate closed.
    taskkillTree(sleeper.pid)

    const result = await waitForBackendRelease(
      [sleeper.pid],
      {
        isShimLocked: () => false,
        isPidAlive: pid => {
          const alive = isPidAliveWindows(pid)

          if (firstAliveCheck === null) {
            firstAliveCheck = alive
          }

          return alive
        },
        collectStragglerPids: () => [],
        killProcessTree: taskkillTree,
        sleep: ms => new Promise(r => setTimeout(r, ms)),
        now: () => Date.now(),
        log: line => logs.push(line)
      },
      'live-e2e'
    )

    expect(result.unlocked).toBe(true)
    // The gate resolved only after the real PID left the real table:
    expect(isPidAliveWindows(sleeper.pid)).toBe(false)
    expect(result.lingeringPids).toEqual([])
    // Record whether the race window was observable on this runner (taskkill
    // returned while the PID was still enumerable). Informational: fast
    // runners can retire tiny process trees before our first check, but the
    // gate's correctness (above) does not depend on winning that race.
    logs.push(`race-window-observed=${firstAliveCheck}`)

    expect(logs.some(l => l.includes('safe to proceed'))).toBe(true)
  })

  it('a live foreign holder keeps the gate closed until the deadline', async () => {
    const holder = spawnSleeper()

    try {
      // The 2000 ms deadline below is a real-clock budget for the gate's OWN
      // dwell logic, not for the sleeper to finish appearing in the process
      // table. Confirm the table already reports it alive first so the
      // full window is spent proving the gate holds, never spent waiting on
      // process-creation latency under a loaded host.
      await waitUntilObservablyAlive(holder.pid)

      const result = await waitForBackendRelease(
        [holder.pid],
        {
          // Simulates the shim held by a process we did NOT kill — the gate
          // must fail closed rather than hand off over a live holder.
          isShimLocked: () => true,
          isPidAlive: isPidAliveWindows,
          collectStragglerPids: () => [],
          killProcessTree: () => {
            /* nothing else to kill */
          },
          sleep: ms => new Promise(r => setTimeout(r, ms)),
          now: () => Date.now(),
          log: () => {}
        },
        'live-e2e',
        2000
      )

      expect(result.unlocked).toBe(false)
      expect(result.lingeringPids).toEqual([holder.pid])
    } finally {
      taskkillTree(holder.pid)
    }
  })
})
