import { describe, expect, it, vi } from 'vitest'

import {
  dropPlugin,
  pluginStatus,
  publishPlugin,
  setPluginEnabled,
  subscribePluginStatus
} from './plugins-store'

const uniqueId = (name: string) => `status-test-${name}-${crypto.randomUUID()}`

describe('public plugin status contract', () => {
  it('reports unavailable for absent and inherited record names', () => {
    const id = uniqueId('missing')

    expect(pluginStatus(id)).toEqual({ id, state: 'unavailable' })
    expect(pluginStatus('toString')).toEqual({ id: 'toString', state: 'unavailable' })
    expect(pluginStatus('__proto__')).toEqual({ id: '__proto__', state: 'unavailable' })
  })

  it('maps internal states to a closed non-secret public vocabulary', () => {
    const id = uniqueId('states')
    const base = { id, name: 'Private path must not leak', kind: 'disk' as const, file: 'C:/secret/plugin.js' }

    publishPlugin({ ...base, status: 'disabled' })
    expect(pluginStatus(id)).toEqual({ id, state: 'disabled' })
    publishPlugin({ ...base, status: 'loading' })
    expect(pluginStatus(id)).toEqual({ id, state: 'loading' })
    publishPlugin({ ...base, status: 'loaded' })
    expect(pluginStatus(id)).toEqual({ id, state: 'enabled' })
    publishPlugin({ ...base, status: 'error', error: 'token=must-not-leak' })
    expect(pluginStatus(id)).toEqual({ id, state: 'failed' })
    expect(JSON.stringify(pluginStatus(id))).not.toMatch(/secret|token|path|error/i)
  })

  it('publishes loading then enabled around asynchronous activation', async () => {
    const id = uniqueId('enable')
    let resolveActivation: (() => void) | undefined
    const activation = new Promise<void>(resolve => {
      resolveActivation = resolve
    })

    publishPlugin(
      { id, name: id, kind: 'runtime', status: 'disabled' },
      {
        activate: async () => {
          await activation
          publishPlugin({ id, name: id, kind: 'runtime', status: 'loaded' })
        },
        deactivate: vi.fn()
      }
    )

    const enabling = setPluginEnabled(id, true)
    expect(pluginStatus(id)).toEqual({ id, state: 'loading' })
    resolveActivation?.()
    await enabling
    expect(pluginStatus(id)).toEqual({ id, state: 'enabled' })
  })

  it('publishes failed while preserving the private activation rejection', async () => {
    const id = uniqueId('failure')
    const failure = new Error('private activation failure')

    publishPlugin(
      { id, name: id, kind: 'runtime', status: 'disabled' },
      { activate: async () => Promise.reject(failure), deactivate: vi.fn() }
    )

    await expect(setPluginEnabled(id, true)).rejects.toBe(failure)
    expect(pluginStatus(id)).toEqual({ id, state: 'failed' })
  })

  it('emits only scoped transitions, immediately, and supports unsubscribe', () => {
    const id = uniqueId('events')
    const other = uniqueId('other')
    const states: string[] = []
    const unsubscribe = subscribePluginStatus(id, status => states.push(status.state))

    publishPlugin({ id: other, name: other, kind: 'runtime', status: 'loaded' })
    publishPlugin({ id, name: id, kind: 'runtime', status: 'disabled' })
    publishPlugin({ id, name: id, kind: 'runtime', status: 'loading' })
    publishPlugin({ id, name: id, kind: 'runtime', status: 'loaded' })
    dropPlugin(id)
    unsubscribe()
    publishPlugin({ id, name: id, kind: 'runtime', status: 'loaded' })

    expect(states).toEqual(['unavailable', 'disabled', 'loading', 'enabled', 'unavailable'])
  })

  it('isolates one subscriber exception from lifecycle and sibling observers', () => {
    const id = uniqueId('listener-failure')
    const states: string[] = []
    const stopBroken = subscribePluginStatus(id, () => {
      throw new Error('broken observer')
    })
    const stopHealthy = subscribePluginStatus(id, status => states.push(status.state))

    expect(() => publishPlugin({ id, name: id, kind: 'runtime', status: 'loaded' })).not.toThrow()
    expect(pluginStatus(id).state).toBe('enabled')
    expect(states).toEqual(['unavailable', 'enabled'])
    stopBroken()
    stopHealthy()
  })
})
