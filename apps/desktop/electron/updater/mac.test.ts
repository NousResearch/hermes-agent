import { EventEmitter } from 'node:events'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { MacStrategy, type MacStrategyDeps, prepareMacInstall } from './mac'

function fixture() {
  const events: string[] = []
  const emitter = new EventEmitter()
  const info = { version: '0.29.0', files: [], releaseDate: '', path: '', sha512: '' }

  const deps: MacStrategyDeps = {
    updater: {
      checkForUpdates: vi.fn(async () => {
        events.push('check')

        return { isUpdateAvailable: true, updateInfo: info, versionInfo: info }
      }),
      downloadUpdate: vi.fn(async () => { events.push('download');

 return [] }),
      quitAndInstall: vi.fn(() => { events.push('install') }),
      on: emitter.on.bind(emitter) as MacStrategyDeps['updater']['on'],
      removeListener: emitter.removeListener.bind(emitter) as MacStrategyDeps['updater']['removeListener']
    },
    channel: 'canary',
    appVersion: '0.28.0',
    prepareInstall: vi.fn(async () => { events.push('verify') }),
    beforeInstall: vi.fn(async () => { events.push('stop') }),
    onInstallFailure: vi.fn(async () => { events.push('restore') }),
    emitProgress: vi.fn()
  }

  return { deps, events, emitter, strategy: new MacStrategy(deps) }
}

afterEach(() => vi.useRealTimers())

describe('macOS strategy', () => {
  it('checks the release, verifies before teardown, and installs once', async () => {
    const { strategy, events, emitter } = fixture()
    expect(await strategy.check()).toMatchObject({ channel: 'canary', latestTag: 'v0.29.0', updateAvailable: true })
    events.length = 0
    expect(await strategy.apply()).toMatchObject({ ok: true, handedOff: true })
    expect(events).toEqual(['check', 'download', 'verify', 'stop', 'install'])
    expect(emitter.listenerCount('download-progress')).toBe(0)
  })

  it.each(['downloadUpdate', 'prepareInstall'] as const)('keeps backends alive on %s failure', async failure => {
    const { deps, strategy, events, emitter } = fixture()
    vi.mocked(failure === 'downloadUpdate' ? deps.updater.downloadUpdate : deps.prepareInstall)
      .mockRejectedValueOnce(new Error('invalid update'))
    await expect(strategy.apply()).rejects.toThrow('invalid update')
    expect(events).not.toContain('stop')
    expect(events).not.toContain('install')
    expect(emitter.listenerCount('download-progress')).toBe(0)
  })

  it('does not install when the provider reports no newer release', async () => {
    const { deps, strategy, events } = fixture()
    const info = { version: '0.27.0', files: [], releaseDate: '', path: '', sha512: '' }
    vi.mocked(deps.updater.checkForUpdates).mockResolvedValue({
      isUpdateAvailable: false, updateInfo: info, versionInfo: info
    })
    await strategy.apply()
    expect(events).toEqual([])
    expect(deps.updater.downloadUpdate).not.toHaveBeenCalled()
  })

  it('restores the backend if install handoff throws', async () => {
    const { deps, strategy, events } = fixture()
    vi.mocked(deps.updater.quitAndInstall).mockImplementation(() => { throw new Error('handoff failed') })
    await expect(strategy.apply()).rejects.toThrow('handoff failed')
    expect(events.slice(-2)).toEqual(['stop', 'restore'])
  })

  it('rejects simultaneous apply calls', async () => {
    const { deps, strategy } = fixture()
    let release!: () => void
    vi.mocked(deps.prepareInstall).mockImplementation(() => new Promise(resolve => { release = resolve }))
    const applying = strategy.apply()
    await vi.waitFor(() => expect(deps.prepareInstall).toHaveBeenCalledOnce())
    await expect(strategy.apply()).rejects.toThrow('already in progress')
    await expect(strategy.check()).rejects.toThrow('already in progress')
    release()
    await applying
  })
})

describe('native signature verification', () => {
  it('waits for native readiness and removes both listeners', async () => {
    const native = Object.assign(new EventEmitter(), { checkForUpdates: vi.fn() })
    let ready = false
    const pending = prepareMacInstall(native).then(() => { ready = true })
    await Promise.resolve()
    expect(ready).toBe(false)
    native.emit('update-downloaded')
    await pending
    expect(native.listenerCount('error')).toBe(0)
    expect(native.listenerCount('update-downloaded')).toBe(0)
  })

  it('surfaces native rejection and bounds a missing readiness event', async () => {
    vi.useFakeTimers()
    const native = Object.assign(new EventEmitter(), { checkForUpdates: vi.fn() })
    const rejected = expect(prepareMacInstall(native)).rejects.toThrow('bad signature')
    native.emit('error', new Error('bad signature'))
    await rejected
    const timeout = expect(prepareMacInstall(native, 2000)).rejects.toThrow('timed out')
    await vi.advanceTimersByTimeAsync(2000)
    await timeout
    expect(native.listenerCount('error')).toBe(0)
    expect(native.listenerCount('update-downloaded')).toBe(0)
  })
})
