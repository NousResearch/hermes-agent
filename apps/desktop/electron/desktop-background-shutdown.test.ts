import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { stopDesktopBackgroundServices } from './desktop-background-shutdown'

describe('Desktop background-service shutdown', () => {
  it('runs the resolved Hermes gateway stop --all command and waits for exit', async () => {
    const kill = vi.fn<(signal?: NodeJS.Signals | number) => boolean>(() => true)
    const child = Object.assign(new EventEmitter(), { kill })
    const spawnFn = vi.fn(() => child)
    const resolveBackend = vi.fn(args => ({
      command: '/runtime/bin/hermes',
      args,
      root: '/runtime',
      env: { RUNTIME_MARKER: '1' },
      shell: false
    }))

    const stopped = stopDesktopBackgroundServices({
      resolveBackend,
      spawnFn,
      env: { HERMES_HOME: '/profiles' },
      platform: 'linux',
      timeoutMs: 1_000
    })
    child.emit('exit', 0, null)

    await expect(stopped).resolves.toBe(true)
    expect(resolveBackend).toHaveBeenCalledWith(['gateway', 'stop', '--all'])
    expect(spawnFn).toHaveBeenCalledWith(
      '/runtime/bin/hermes',
      ['gateway', 'stop', '--all'],
      expect.objectContaining({
        cwd: '/runtime',
        env: expect.objectContaining({ HERMES_HOME: '/profiles', RUNTIME_MARKER: '1' }),
        stdio: 'ignore'
      })
    )
    expect(kill).not.toHaveBeenCalled()
  })

  it('bounds a hung stop command and reports failure', async () => {
    vi.useFakeTimers()
    const kill = vi.fn<(signal?: NodeJS.Signals | number) => boolean>(() => true)
    const child = Object.assign(new EventEmitter(), { kill })

    const stopped = stopDesktopBackgroundServices({
      resolveBackend: args => ({ command: 'hermes', args }),
      spawnFn: () => child,
      env: {},
      platform: 'linux',
      timeoutMs: 25
    })
    await vi.advanceTimersByTimeAsync(25)

    await expect(stopped).resolves.toBe(false)
    expect(kill).toHaveBeenCalledWith('SIGTERM')
    vi.useRealTimers()
  })

  it('boots out the exact Hermes companion launchd job on macOS', async () => {
    const children: EventEmitter[] = []
    const spawnFn = vi.fn(() => {
      const child = Object.assign(new EventEmitter(), {
        kill: vi.fn(() => true)
      })
      children.push(child)
      return child
    })

    const stopped = stopDesktopBackgroundServices({
      resolveBackend: args => ({ command: '/runtime/bin/hermes', args }),
      spawnFn,
      env: { HERMES_HOME: '/profiles' },
      platform: 'darwin',
      uid: 501,
      timeoutMs: 1_000
    })
    children[0].emit('exit', 0, null)
    children[1].emit('exit', 0, null)

    await expect(stopped).resolves.toBe(true)
    expect(spawnFn).toHaveBeenNthCalledWith(
      2,
      '/bin/launchctl',
      ['bootout', 'gui/501/com.local.hermes.companion-backend'],
      expect.objectContaining({ shell: false, stdio: 'ignore' })
    )
  })
})
