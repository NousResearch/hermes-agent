import { describe, expect, it, vi } from 'vitest'

import { createInstalledRuntimeGate } from './backend-resolution'

const ROOT = '/home/u/.hermes/hermes-agent'
const active = { label: 'active runtime' }

function gate(env: NodeJS.ProcessEnv) {
  const log = vi.fn()

  return { log, runtime: createInstalledRuntimeGate(env, log) }
}

describe('installed runtime gate', () => {
  it('uses the installed runtime without the flag', async () => {
    const { runtime } = gate({})
    const probe = vi.fn(async () => active)

    await expect(runtime.resolve(ROOT, probe)).resolves.toBe(active)
    expect(probe).toHaveBeenCalledOnce()
  })

  it('skips the installed runtime without probing it when HERMES_DESKTOP_IGNORE_EXISTING=1', async () => {
    const { log, runtime } = gate({ HERMES_DESKTOP_IGNORE_EXISTING: '1' })
    const probe = vi.fn(async () => active)

    await expect(runtime.resolve(ROOT, probe)).resolves.toBeNull()
    expect(probe).not.toHaveBeenCalled()
    expect(log).toHaveBeenCalledWith(expect.stringContaining(ROOT))
  })

  it('honours only the value the CLI exports', async () => {
    for (const value of ['0', 'true', '']) {
      const { runtime } = gate({ HERMES_DESKTOP_IGNORE_EXISTING: value })

      await expect(runtime.resolve(ROOT, async () => active)).resolves.toBe(active)
    }
  })

  it('uses the runtime this launch installed so the post-install re-resolve does not reinstall', async () => {
    const { runtime } = gate({ HERMES_DESKTOP_IGNORE_EXISTING: '1' })
    const probe = vi.fn(async () => active)
    // main.ts: resolveHermesBackend's ACTIVE_HERMES_ROOT rung, else bootstrap-needed.
    const resolveHermesBackend = async () => (await runtime.resolve(ROOT, probe)) ?? 'bootstrap-needed'

    await expect(resolveHermesBackend()).resolves.toBe('bootstrap-needed')
    expect(probe).not.toHaveBeenCalled()
    // main.ts ensureRuntime: the re-resolve after a successful bootstrap.
    await expect(runtime.afterInstall(resolveHermesBackend)).resolves.toBe(active)
    // Later resolves (pool backends, TUI resume) keep the new runtime too.
    await expect(resolveHermesBackend()).resolves.toBe(active)
    expect(probe).toHaveBeenCalledTimes(2)
  })

  it('reads the flag at resolve time', async () => {
    const env: NodeJS.ProcessEnv = {}
    const { runtime } = gate(env)

    env.HERMES_DESKTOP_IGNORE_EXISTING = '1'
    await expect(runtime.resolve(ROOT, async () => active)).resolves.toBeNull()
  })
})

describe('installed runtime gate under a stalled probe', () => {
  // The observed failure: a re-resolve during a gateway restart ran the 15s
  // `--version` probe on a stalled box, got nothing back, and answered
  // "no usable Hermes install" -> the first-run setup card over a healthy
  // install (desktop.log 2026-11-07 11:49:45, 2026-11-08 01:00:50).
  it('keeps a runtime this launch already proved when a later probe fails', async () => {
    const { log, runtime } = gate({})
    let healthy = true
    const probe = vi.fn(async () => (healthy ? active : null))

    await expect(runtime.resolve(ROOT, probe)).resolves.toBe(active)

    healthy = false
    await expect(runtime.resolve(ROOT, probe)).resolves.toBe(active)
    expect(log).toHaveBeenCalledWith(expect.stringContaining('already proved it usable'))
  })

  it('still reports no install when the first probe fails', async () => {
    const { runtime } = gate({})
    const probe = vi.fn(async () => null)

    await expect(runtime.resolve(ROOT, probe)).resolves.toBeNull()
  })

  it('does not carry the proved runtime over to another root', async () => {
    const { runtime } = gate({})

    await expect(runtime.resolve(ROOT, async () => active)).resolves.toBe(active)
    await expect(runtime.resolve('/other/hermes-agent', async () => null)).resolves.toBeNull()
  })

  it('does not revive the proved runtime when the flag turns on later', async () => {
    const env: NodeJS.ProcessEnv = {}
    const { runtime } = gate(env)

    await expect(runtime.resolve(ROOT, async () => active)).resolves.toBe(active)
    env.HERMES_DESKTOP_IGNORE_EXISTING = '1'
    await expect(runtime.resolve(ROOT, async () => active)).resolves.toBeNull()
  })

  it('reports whether installed runtimes are being skipped', async () => {
    expect(gate({ HERMES_DESKTOP_IGNORE_EXISTING: '1' }).runtime.skipsInstalledRuntimes()).toBe(true)
    expect(gate({}).runtime.skipsInstalledRuntimes()).toBe(false)
  })
})
