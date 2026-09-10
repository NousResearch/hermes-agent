import { describe, expect, it } from 'vitest'

import { StoreStrategy, type StoreStrategyDeps } from './store'

function dependencies(failAt?: string): { deps: StoreStrategyDeps; calls: string[] } {
  const calls: string[] = []

  const record = async (name: string): Promise<void> => {
    calls.push(name)

    if (name === failAt) {
      throw new Error(`${name} failed`)
    }
  }

  return {
    calls,
    deps: {
      run: async mode => {
        await record(mode)

        return { available: true, packages: ['package-v2'], ok: true }
      },
      appVersion: '1.0.0',
      registerPendingRelaunch: async () => {
        await record('register')

        return { automatic: true, cancel: () => record('cancel') }
      },
      teardown: () => record('teardown'),
      restore: () => record('restore'),
      quit: () => {
        calls.push('quit')
      },
      emitProgress: () => {}
    }
  }
}

describe('Microsoft Store update lifecycle', () => {
  it('downloads before shutdown, owns relaunch before install, and keeps no-update non-destructive', async () => {
    const { deps, calls } = dependencies()
    const strategy = new StoreStrategy(deps)
    expect((await strategy.check()).updateAvailable).toBe(true)
    calls.length = 0
    expect(await strategy.apply()).toMatchObject({ ok: true, handedOff: true, mechanism: 'microsoft-store' })
    expect(calls).toEqual(['download', 'register', 'teardown', 'install', 'quit'])

    const idle = new StoreStrategy({ ...deps, run: async () => ({ available: false, packages: [], ok: true }) })
    calls.length = 0
    expect(await idle.apply()).toMatchObject({ ok: true, updateAvailable: false })
    expect(calls).toEqual([])
  })

  it.each(['download', 'register', 'teardown', 'install'])(
    'a %s failure never quits and restores only after shutdown began',
    async failAt => {
      const { deps, calls } = dependencies(failAt)
      await expect(new StoreStrategy(deps).apply()).rejects.toThrow(`${failAt} failed`)
      expect(calls).not.toContain('quit')
      expect(calls.includes('restore')).toBe(['teardown', 'install'].includes(failAt))
      expect(calls.includes('cancel')).toBe(['teardown', 'install'].includes(failAt))
    }
  )

  it('refuses unknown availability and preserves cancellation plus restore errors', async () => {
    const { deps, calls } = dependencies('install')

    const unknown = new StoreStrategy({
      ...deps,
      run: async () => ({ available: null, packages: [], ok: false, error: 'not Store acquired' })
    })

    expect(await unknown.check()).toMatchObject({ error: 'not Store acquired', updateAvailable: false })
    await expect(unknown.apply()).rejects.toThrow('not Store acquired')
    expect(calls).toEqual([])

    const broken = new StoreStrategy({
      ...deps,
      registerPendingRelaunch: async () => ({
        automatic: true,
        cancel: async () => {
          throw new Error('cancel failed')
        }
      }),
      restore: async () => {
        throw new Error('restore failed')
      }
    })

    await expect(broken.apply()).rejects.toThrow('install failed; cancel failed; restore failed')
  })
})
