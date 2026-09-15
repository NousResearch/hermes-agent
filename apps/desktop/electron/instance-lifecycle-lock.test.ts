/**
 * instance-lifecycle-lock.test.ts
 *
 * #107671: relaunching Hermes Desktop while a prior instance is still in
 * before-quit teardown can win Electron's single-instance lock (Chromium
 * releases it when quit *starts*) and become a second GNOME-visible app.
 * On GNOME/Wayland that overlapping START/STOP races shell_app_dispose and
 * can crash the session.
 *
 * The extracted helper must park a would-be primary until the pidfile holder
 * is gone — before createWindow / startHermes — and fail open on stale or
 * unreadable pidfiles so a crash cannot brick the next launch.
 */

import path from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  awaitPriorDesktopInstance,
  INSTANCE_LIFECYCLE_POLL_MS,
  INSTANCE_PID_FILENAME,
  instancePidPath,
  lifecycleAllowsCreateWindow,
  type LifecycleDecision,
  parseInstancePid
} from './instance-lifecycle-lock'

/** Fake clock: sleep() advances time instantly (same pattern as backend-release-gate). */
function fakeClock() {
  let t = 0

  return {
    now: () => t,
    sleep: async (ms: number) => {
      t += ms
    }
  }
}

function makeHarness(opts: {
  raw?: string | null
  readError?: Error
  writeError?: Error
  isPidAlive?: (pid: number) => boolean
  timeoutMs?: number
  pollMs?: number
}) {
  const clock = fakeClock()
  const files = new Map<string, string>()
  const userDataDir = '/tmp/hermes-userdata-107671'
  const pidPath = instancePidPath(userDataDir)
  const writes: Array<{ path: string; contents: string }> = []
  const logs: string[] = []

  if (opts.raw !== undefined && opts.raw !== null) {
    files.set(pidPath, opts.raw)
  }

  return {
    clock,
    files,
    pidPath,
    writes,
    logs,
    userDataDir,
    opts: {
      userDataDir,
      timeoutMs: opts.timeoutMs ?? 5_000,
      pollMs: opts.pollMs ?? INSTANCE_LIFECYCLE_POLL_MS,
      now: clock.now,
      sleep: clock.sleep,
      readFile: (path: string) => {
        if (opts.readError) {
          throw opts.readError
        }

        if (!files.has(path)) {
          return null
        }

        return files.get(path) ?? null
      },
      writeFile: (path: string, contents: string) => {
        if (opts.writeError) {
          throw opts.writeError
        }

        files.set(path, contents)
        writes.push({ path, contents })
      },
      isPidAlive: opts.isPidAlive ?? (() => false),
      log: (line: string) => logs.push(line)
    }
  }
}

describe('parseInstancePid', () => {
  it('accepts a positive integer, including a trailing newline', () => {
    expect(parseInstancePid('140616\n')).toBe(140616)
    expect(parseInstancePid('  140616  ')).toBe(140616)
  })

  it('rejects missing, malformed, non-integer, and non-positive values', () => {
    expect(parseInstancePid(null)).toBeNull()
    expect(parseInstancePid(undefined)).toBeNull()
    expect(parseInstancePid('')).toBeNull()
    expect(parseInstancePid('not-a-pid')).toBeNull()
    expect(parseInstancePid('12.5')).toBeNull()
    expect(parseInstancePid('0')).toBeNull()
    expect(parseInstancePid('-4')).toBeNull()
    expect(parseInstancePid('140616 leftover')).toBeNull()
  })
})

describe('lifecycleAllowsCreateWindow', () => {
  it('blocks createWindow while Electron is ready but the pidfile wait has not proceeded (THE P0)', () => {
    // second-instance at module load: app.isReady() can be true and
    // mainWindow still null during awaitPriorDesktopInstance.
    expect(lifecycleAllowsCreateWindow(true, false)).toBe(false)
  })

  it('allows createWindow only after both Electron ready and lifecycle proceed', () => {
    expect(lifecycleAllowsCreateWindow(true, true)).toBe(true)
  })

  it('blocks createWindow when Electron is not ready, even if the latch is set', () => {
    expect(lifecycleAllowsCreateWindow(false, false)).toBe(false)
    expect(lifecycleAllowsCreateWindow(false, true)).toBe(false)
  })
})

describe('instancePidPath', () => {
  it(`joins the desktop userData dir with ${INSTANCE_PID_FILENAME}`, () => {
    expect(instancePidPath('/home/user/.config/hermes-desktop')).toBe(
      path.join('/home/user/.config/hermes-desktop', INSTANCE_PID_FILENAME)
    )
  })
})

describe('awaitPriorDesktopInstance (#107671 teardown overlap)', () => {
  it('#107671 overlap (THE RED): does not proceed while holder PID 140616 is still alive', async () => {
    // journalctl shape: new instance 140566 won the Electron lock; old
    // instance 140616 is still in before-quit teardown for ~1.2s of fake time.
    const holderPid = 140616
    const selfPid = 140566
    const aliveUntil = 1_200

    const harness = makeHarness({
      raw: `${holderPid}\n`,
      isPidAlive: pid => pid === holderPid && harness.clock.now() < aliveUntil
    })

    let resolved: LifecycleDecision | null = null

    const pending = awaitPriorDesktopInstance({
      ...harness.opts,
      selfPid
    }).then(decision => {
      resolved = decision

      return decision
    })

    // A naive "lock won ⇒ proceed immediately" resolves at t=0 while 140616
    // is still alive. The helper must stay pending until the holder exits.
    await Promise.resolve()
    expect(resolved).toBeNull()
    expect(harness.clock.now()).toBeLessThan(aliveUntil)

    const decision = await pending

    expect(decision).toBe('proceed')
    expect(resolved).toBe('proceed')
    expect(harness.clock.now()).toBeGreaterThanOrEqual(aliveUntil)
    expect(harness.writes.at(-1)?.contents.trim()).toBe(String(selfPid))
  })

  it('fail-open stale: dead holder PID proceeds immediately (no wait)', async () => {
    const harness = makeHarness({
      raw: '140616\n',
      isPidAlive: () => false
    })

    const decision = await awaitPriorDesktopInstance({
      ...harness.opts,
      selfPid: 140566
    })

    expect(decision).toBe('proceed')
    expect(harness.clock.now()).toBe(0)
    expect(harness.writes.at(-1)?.contents.trim()).toBe('140566')
  })

  it('fail-open missing/malformed pidfile proceeds immediately', async () => {
    const missing = makeHarness({})
    const malformed = makeHarness({ raw: 'not-a-pid\n' })
    const nonPositive = makeHarness({ raw: '0\n' })

    await expect(awaitPriorDesktopInstance({ ...missing.opts, selfPid: 9 })).resolves.toBe('proceed')
    await expect(awaitPriorDesktopInstance({ ...malformed.opts, selfPid: 9 })).resolves.toBe('proceed')
    await expect(awaitPriorDesktopInstance({ ...nonPositive.opts, selfPid: 9 })).resolves.toBe('proceed')

    expect(missing.clock.now()).toBe(0)
    expect(malformed.clock.now()).toBe(0)
    expect(nonPositive.clock.now()).toBe(0)
  })

  it('fail-open filesystem errors reading or writing the pidfile proceed (do not brick launch)', async () => {
    const readBoom = makeHarness({
      readError: Object.assign(new Error('permission denied'), { code: 'EACCES' })
    })

    const writeBoom = makeHarness({
      raw: '12.5\n',
      writeError: Object.assign(new Error('ro fs'), { code: 'EROFS' })
    })

    await expect(awaitPriorDesktopInstance({ ...readBoom.opts, selfPid: 7 })).resolves.toBe('proceed')
    await expect(awaitPriorDesktopInstance({ ...writeBoom.opts, selfPid: 7 })).resolves.toBe('proceed')
    expect(readBoom.clock.now()).toBe(0)
    expect(writeBoom.clock.now()).toBe(0)
  })

  it('timeout still-alive: holder stays past the deadline → exit-as-secondary (no pidfile overwrite)', async () => {
    const holderPid = 140616

    const harness = makeHarness({
      raw: `${holderPid}\n`,
      isPidAlive: () => true,
      timeoutMs: 1_000,
      pollMs: 250
    })

    const decision = await awaitPriorDesktopInstance({
      ...harness.opts,
      selfPid: 140566
    })

    expect(decision).toBe('exit-as-secondary')
    expect(harness.clock.now()).toBeGreaterThanOrEqual(1_000)
    expect(harness.writes).toEqual([])
    expect(harness.files.get(harness.pidPath)).toBe(`${holderPid}\n`)
  })

  it('self PID in pidfile proceeds immediately', async () => {
    const selfPid = 140566

    const harness = makeHarness({
      raw: `${selfPid}\n`,
      isPidAlive: pid => pid === selfPid
    })

    const decision = await awaitPriorDesktopInstance({
      ...harness.opts,
      selfPid
    })

    expect(decision).toBe('proceed')
    expect(harness.clock.now()).toBe(0)
  })
})
