import { describe, expect, it, vi } from 'vitest'

import { createInstalledRuntimeGate } from './backend-resolution'
import type { InstallStamp } from './install-stamp'

const ROOT = '/home/u/.hermes/hermes-agent'
const active = { label: 'active runtime' }

function stamp(payload: InstallStamp['payload']): InstallStamp {
  return {
    schemaVersion: 2,
    commit: null,
    commitDate: null,
    branch: null,
    builtAt: null,
    dirty: false,
    source: 'build',
    distribution: 'desktop-app',
    updateMechanism: 'self',
    baseVersion: null,
    displayVersion: null,
    distance: null,
    payload,
    tag: null
  }
}

function gate(env: NodeJS.ProcessEnv, artifact: InstallStamp | null = null) {
  const log = vi.fn()

  return { log, runtime: createInstalledRuntimeGate(env, log, artifact) }
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

  it('always skips an installation already on the machine on a remote-only artifact', async () => {
    // The target PC has a full %LOCALAPPDATA%\\hermes install. A light build
    // must not adopt it even without the flag: it has no local runtime.
    for (const env of [{}, { HERMES_DESKTOP_IGNORE_EXISTING: '0' }]) {
      const { log, runtime } = gate(env, stamp('light'))
      const probe = vi.fn(async () => active)

      await expect(runtime.resolve(ROOT, probe)).resolves.toBeNull()
      expect(probe).not.toHaveBeenCalled()
      expect(log).toHaveBeenCalledWith(expect.stringContaining('remote-only'))
    }
  })

  it('keeps using the runtime a launch installed on a remote-only artifact', async () => {
    const { runtime } = gate({}, stamp('light'))
    const probe = vi.fn(async () => active)

    await expect(runtime.afterInstall(() => runtime.resolve(ROOT, probe))).resolves.toBe(active)
    expect(probe).toHaveBeenCalledOnce()
  })

  it('does not skip the installed runtime for bundled/checkout artifacts', async () => {
    for (const artifact of [stamp('bundled'), stamp('bootstrap'), null]) {
      const { runtime } = gate({}, artifact)

      await expect(runtime.resolve(ROOT, async () => active)).resolves.toBe(active)
    }
  })
})
