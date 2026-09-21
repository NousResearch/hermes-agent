import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { stopDesktopBackgroundServices } from './desktop-background-shutdown'

describe('Desktop background-service shutdown', () => {
  it('does not stop messaging gateways when Desktop quits', async () => {
    const spawnFn = vi.fn()

    await expect(stopDesktopBackgroundServices({
      resolveBackend: vi.fn(),
      spawnFn,
      env: { HERMES_HOME: '/profiles' },
      platform: 'linux'
    })).resolves.toBe(true)

    expect(spawnFn).not.toHaveBeenCalled()
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

    expect(spawnFn).toHaveBeenCalledTimes(1)
    children[0].emit('exit', 0, null)

    await expect(stopped).resolves.toBe(true)
    expect(spawnFn).toHaveBeenNthCalledWith(
      1,
      '/bin/launchctl',
      ['bootout', 'gui/501/com.local.hermes.companion-backend'],
      expect.objectContaining({ shell: false, stdio: 'ignore' })
    )
  })
})
