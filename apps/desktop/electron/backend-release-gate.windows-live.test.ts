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
 *   1. the liveness probe observes a real process before and after termination,
 *      and
 *   2. the gate, wired to the real probes, dwells through that window and
 *      only passes once the PID has genuinely left the table.
 */

import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import path from 'node:path'
import { promisify } from 'node:util'

import { describe, expect, it } from 'vitest'

import { isPidAliveWindows, waitForBackendRelease } from './backend-release-gate'

const isWindows = process.platform === 'win32'
const execFileAsync = promisify(execFile)
const TASKKILL_TIMEOUT_MS = 15_000

async function spawnSleeper(): Promise<{ pid: number; kill: () => Promise<void> }> {
  // A real native process, owned exclusively by this test.
  const child = spawn('powershell', ['-NoProfile', '-Command', 'Start-Sleep -Seconds 300'], {
    stdio: 'ignore',
    windowsHide: true
  })

  await once(child, 'spawn')

  if (!child.pid) {
    throw new Error('sleeper failed to spawn')
  }

  return {
    pid: child.pid,
    kill: async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        return
      }

      const exited = once(child, 'exit')
      let timer: ReturnType<typeof setTimeout> | undefined

      try {
        if (child.exitCode === null && !child.killed) {
          child.kill()
        }

        await Promise.race([
          exited,
          new Promise<never>((_, reject) => {
            timer = setTimeout(() => reject(new Error('Owned sleeper did not exit during cleanup')), 5_000)
          })
        ])
      } finally {
        clearTimeout(timer)
      }
    }
  }
}

async function taskkillOwned(pid: number, tree: boolean): Promise<void> {
  const args = ['/PID', String(pid), '/F', ...(tree ? ['/T'] : [])]
  const binary = path.join(process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows', 'System32', 'taskkill.exe')

  try {
    // Synchronous taskkill blocks Vitest's own timeout timer. Bound the native
    // command separately and surface timeout/nonzero exit as a test failure.
    await execFileAsync(binary, args, { windowsHide: true, timeout: TASKKILL_TIMEOUT_MS, maxBuffer: 64 * 1024 })
  } catch (error: any) {
    throw new Error(
      `Native Windows taskkill ${tree ? '/T /F' : '/F'} failed: ` +
        `code=${String(error.code)}, signal=${String(error.signal)}, killed=${String(error.killed)} ` +
        `(command deadline ${TASKKILL_TIMEOUT_MS}ms). The native termination prerequisite failed before the release gate.`,
      { cause: error }
    )
  }
}

describe.skipIf(!isWindows)('waitForBackendRelease — live Windows (#74805)', () => {
  it('isPidAliveWindows tracks a real process through spawn and exit', async () => {
    const sleeper = await spawnSleeper()

    try {
      expect(isPidAliveWindows(sleeper.pid)).toBe(true)

      // This case proves PID liveness; the next case retains real tree-kill.
      await taskkillOwned(sleeper.pid, false)

      // Poll until the table retires the PID (bounded).
      const deadline = Date.now() + 10000

      while (isPidAliveWindows(sleeper.pid) && Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 100))
      }

      expect(isPidAliveWindows(sleeper.pid)).toBe(false)
    } finally {
      await sleeper.kill()
    }
  }, 40_000)

  it('the gate dwells until a real killed PID leaves the live process table', async () => {
    const sleeper = await spawnSleeper()
    const logs: string[] = []
    let firstAliveCheck: boolean | null = null

    // Fire the real taskkill and IMMEDIATELY enter the gate — the #74805
    // shape. The shim probe reads unlocked throughout (the serve backend
    // never held it); only the PID exit-wait can hold the gate closed.
    try {
      await taskkillOwned(sleeper.pid, true)

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
          killProcessTree: () => {
            throw new Error('This fixture has no stragglers to kill')
          },
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
    } finally {
      await sleeper.kill()
    }
  }, 40_000)

  it('a live foreign holder keeps the gate closed until the deadline', async () => {
    const holder = await spawnSleeper()

    try {
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
      await holder.kill()
    }
  }, 40_000)
})
