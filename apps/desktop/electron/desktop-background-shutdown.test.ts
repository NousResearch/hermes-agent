import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { stopDesktopBackgroundServices } from './desktop-background-shutdown'

describe('Desktop background-service shutdown', () => {
  it('runs the resolved Hermes gateway stop --all command and waits for exit', async () => {
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()
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
    expect(child.kill).not.toHaveBeenCalled()
  })

  it('bounds a hung stop command and reports failure', async () => {
    vi.useFakeTimers()
    const child = new EventEmitter() as EventEmitter & { kill: ReturnType<typeof vi.fn> }
    child.kill = vi.fn()

    const stopped = stopDesktopBackgroundServices({
      resolveBackend: args => ({ command: 'hermes', args }),
      spawnFn: () => child,
      env: {},
      timeoutMs: 25
    })
    await vi.advanceTimersByTimeAsync(25)

    await expect(stopped).resolves.toBe(false)
    expect(child.kill).toHaveBeenCalledWith('SIGTERM')
    vi.useRealTimers()
  })
})
