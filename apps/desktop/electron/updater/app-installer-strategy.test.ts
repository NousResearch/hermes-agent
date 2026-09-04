// updater/app-installer-strategy.test.ts — the apply-flow contract: the
// one-shot relaunch marker is written BEFORE the OS hand-off, teardown runs
// before the trigger, and quit is unconditional (no relaunch question).

import { describe, expect, it } from 'vitest'

import { AppInstallerStrategy } from './app-installer'
import type { AppInstallerStrategyDeps } from './app-installer'

function makeDeps(over: Partial<AppInstallerStrategyDeps> = {}) {
  const calls: string[] = []

  const deps: AppInstallerStrategyDeps = {
    python: 'python.exe',
    script: 'check.py',
    run: async () => ({ code: 0, stdout: '{"available": true}' }),
    channel: 'stable',
    light: false,
    feedBaseUrl: 'https://updates.example/hermes-desktop',
    shell: { openExternal: async () => { calls.push('openExternal') } },
    teardownBundledBackend: async () => { calls.push('teardown') },
    emitUpdateProgress: () => {},
    appVersion: '0.18.2',
    quit: () => { calls.push('quit') },
    registerPendingRelaunch: () => { calls.push('relaunch-marker');

 return true },
    ...over
  }

  return { deps, calls }
}

describe('AppInstallerStrategy.apply', () => {
  it('writes the relaunch marker before teardown, trigger, and quit — order is the contract', async () => {
    const { deps, calls } = makeDeps()
    const strategy = new AppInstallerStrategy(deps)
    const result = await strategy.apply({})

    expect(result).toEqual({ ok: true, manual: false, bundled: true, mechanism: 'app-installer' })
    expect(calls).toEqual(['relaunch-marker', 'teardown', 'openExternal', 'quit'])
  })

  it('fails open: a marker-write failure never blocks the update', async () => {
    const { deps, calls } = makeDeps({ registerPendingRelaunch: () => false })
    const result = await new AppInstallerStrategy(deps).apply({})
    expect(result.ok).toBe(true)
    expect(calls).toContain('quit')
  })

  it('no feed URL → manual card, no teardown, no quit', async () => {
    const { deps, calls } = makeDeps({ feedBaseUrl: '' })
    const result = await new AppInstallerStrategy(deps).apply({})
    expect(result).toEqual({ ok: true, manual: true, bundled: true, mechanism: 'app-installer' })
    expect(calls).toEqual([])
  })
})

describe('AppInstallerStrategy.check', () => {
  it('threads the OS checker result onto the mechanism wire', async () => {
    const { deps } = makeDeps({ run: async () => ({ code: 0, stdout: '{"available": true}' }) })
    const status = await new AppInstallerStrategy(deps).check()
    expect(status.mechanism).toBe('app-installer')
    expect(status.updateAvailable).toBe(true)
    expect(status.error).toBeUndefined()
  })

  it('unknown availability is an error on the wire, never "no update"', async () => {
    const { deps } = makeDeps({ run: async () => ({ code: 1, stdout: '{"available": null, "error": "winrt missing"}' }) })
    const status = await new AppInstallerStrategy(deps).check()
    expect(status.updateAvailable).toBe(false)
    expect(status.error).toBe('winrt missing')
  })
})
