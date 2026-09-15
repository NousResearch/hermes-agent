// updater/app-installer-strategy.test.ts — the apply-flow contract: the
// relaunch registration (marker + detached waiter) completes BEFORE the OS
// hand-off. Teardown finishes before the descriptor opens.

import { describe, expect, it } from 'vitest'

import { AppInstallerStrategy } from './app-installer'
import type { AppInstallerStrategyDeps } from './app-installer'

interface StrategyFixture { deps: AppInstallerStrategyDeps; calls: string[] }

function makeDeps(over: Partial<AppInstallerStrategyDeps> = {}): StrategyFixture {
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

  it('uses an exact descriptor without a Python check, and rejects unverified prepared packages before teardown', async (): Promise<void> => {
    const { deps, calls } = makeDeps({
      run: async (): Promise<never> => { throw new Error('Dynamic channels do not query the registered moving feed') },
      feed: { url: 'https://updates.example/releases/channel-builds/abc/win32/update.appinstaller', version: '0.0.2.0',
        verifyPrepared: async (): Promise<void> => { calls.push('verify'); throw new Error('wrong signature') }
      },
      appVersion: '0.0.1.0', feedBaseUrl: 'https://updates.example',
      installer: { prepare: async (url: string): Promise<string> => { calls.push(url); return 'pinned.appinstaller' }, open: async (): Promise<string> => { calls.push('open'); return '' } }
    })
    const strategy = new AppInstallerStrategy(deps)
    expect(await strategy.check()).toMatchObject({ updateAvailable: true })
    await expect(strategy.apply()).rejects.toThrow('wrong signature')
    expect(calls).toContain('https://updates.example/releases/channel-builds/abc/win32/update.appinstaller')
    expect(calls).not.toContain('teardown')
    expect(calls).not.toContain('open')
  })

  it('no feed URL → manual card, no teardown, no quit', async () => {
    const { deps, calls } = makeDeps({ feedBaseUrl: '' })
    const result = await new AppInstallerStrategy(deps).apply()
    expect(result).toEqual({ ok: true, manual: true, bundled: true, mechanism: 'app-installer' })
    expect(calls).toEqual([])
  })
})

it.each([
  [0, '{"available":true,"availability":"Available"}', true, undefined],
  [0, '{"available":false}', false, undefined],
  [2, '{"available":null,"error":"winrt missing"}', undefined, 'winrt missing'],
  [0, '', undefined, 'checker returned no availability'],
  [1, 'boom', undefined, 'checker exited 1'],
  [0, '{"available":"yes"}', undefined, 'checker returned no availability']
] as const)('checker %s %s → available=%s error=%s', async (code: number, stdout: string, available: boolean | undefined, error: string | undefined): Promise<void> => {
  const { deps }: ReturnType<typeof makeDeps> = makeDeps({
    run: async (python: string, script: string): Promise<{ code: number; stdout: string }> => {
      expect([python, script]).toEqual(['python.exe', 'check.py'])

      return { code, stdout }
    }
  })

  expect(await new AppInstallerStrategy(deps).check()).toMatchObject({
    supported: true, mechanism: 'app-installer', currentVersion: '0.18.2', updateAvailable: available, error
  })
})

it.each([
  ['stable', false, 'win32/stable/stable.appinstaller'],
  ['canary', false, 'win32/canary/canary.appinstaller'],
  ['stable', true, 'win32/light/stable/stable.appinstaller'],
  ['canary', true, 'win32/light/canary/canary.appinstaller']
] as const)('stages %s light=%s from its feed and reports open errors', async (channel: 'stable' | 'canary', light: boolean, suffix: string): Promise<void> => {
  const { deps, calls }: ReturnType<typeof makeDeps> = makeDeps({ channel, light,
    installer: {
      prepare: async (url: string): Promise<string> => {
        expect(url).toBe(`https://updates.example/hermes-desktop/${suffix}`)

        return 'update.appinstaller'
      },
      open: async (file: string): Promise<string> => {
        expect(file).toBe('update.appinstaller')

        return 'No file association'
      }
    }
  })

  await expect(new AppInstallerStrategy(deps).apply()).rejects.toThrow('No file association')
  expect(calls).toEqual(['relaunch-marker', 'teardown', 'restore'])
})
