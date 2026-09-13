// updater/app-installer-strategy.test.ts — the apply-flow contract: the
// relaunch registration (marker + detached waiter) completes BEFORE the OS
// hand-off. Teardown finishes before the descriptor opens.

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
    installer: {
      prepare: async () => { calls.push('prepare');

 return 'update.appinstaller' },
      open: async () => { calls.push('open');

 return '' }
    },
    teardownBundledBackend: async () => { calls.push('teardown') },
    restoreBundledBackend: async () => { calls.push('restore') },
    emitUpdateProgress: () => {},
    appVersion: '0.18.2',
    quit: () => { calls.push('quit') },
    registerPendingRelaunch: async () => { calls.push('relaunch-marker');

 return { automatic: true, cancel: async () => {} } },
    ...over
  }

  return { deps, calls }
}

describe('AppInstallerStrategy.apply', () => {
  it('writes the relaunch marker before teardown, trigger, and quit — order is the contract', async () => {
    const { deps, calls } = makeDeps()
    const strategy = new AppInstallerStrategy(deps)
    const result = await strategy.apply()

    expect(result).toEqual({ ok: true, manual: false, bundled: true, handedOff: true, mechanism: 'app-installer' })
    expect(calls).toEqual(['prepare', 'relaunch-marker', 'teardown', 'open', 'quit'])
  })

  it('fails open: a marker-write failure never blocks the update', async () => {
    const progress: string[] = []

    const { deps, calls } = makeDeps({
      registerPendingRelaunch: async () => ({ automatic: false, cancel: async () => {} }),
      emitUpdateProgress: event => { progress.push(event.message) }
    })

    const result = await new AppInstallerStrategy(deps).apply()
    expect(result.ok).toBe(true)
    expect(calls).toContain('quit')
    expect(progress.some(message => message.includes('Reopen Hermes'))).toBe(true)
  })

  it('uses the package registered source when no feed override is configured', async () => {
    const prepared: string[] = []

    const { deps, calls } = makeDeps({
      feedBaseUrl: '',
      run: async () => ({ code: 2, stdout: JSON.stringify({ available: true, source_uri: 'https://registered.example/channel.appinstaller' }) }),
      installer: {
        prepare: async url => { prepared.push(url);

 return 'registered.appinstaller' },
        open: async () => ''
      }
    })

    expect((await new AppInstallerStrategy(deps).apply()).manual).toBe(false)
    expect(prepared).toEqual(['https://registered.example/channel.appinstaller'])
    expect(calls).toEqual(['relaunch-marker', 'teardown', 'quit'])
  })

  it('no feed URL → manual card, no teardown, no quit', async () => {
    const { deps, calls } = makeDeps({ feedBaseUrl: '' })
    const result = await new AppInstallerStrategy(deps).apply()
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
