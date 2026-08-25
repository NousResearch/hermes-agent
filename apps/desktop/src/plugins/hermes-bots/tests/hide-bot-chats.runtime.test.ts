import type * as HermesSdk from '@hermes/plugin-sdk'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

const gatewayState = atom<'closed' | 'open'>('closed')

const listPersistedSessions = vi.fn(
  async (_route: unknown, options: { profile: string }) => ({
    sessions: [{ id: `${options.profile}-bot`, profile: options.profile, started_at: 1, title: 'Bot Chat' }]
  })
)

const setPersistedSessionHidden = vi.fn(
  async (_route: unknown, _options: { hidden: boolean; profile: string; sessionId: string }) => ({ hidden: true, ok: true })
)

const request = vi.fn(async (method: string) =>
  method === 'profiles.list' ? { profiles: [{ name: 'alpha' }, { name: 'beta' }] } : {}
)

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: {
      ...sdk.host,
      listPersistedSessions,
      onEvent: vi.fn(() => () => undefined),
      profileRoutes: vi.fn(async () => []),
      request,
      setPersistedSessionHidden,
      state: {
        ...sdk.host.state,
        gateway: gatewayState,
        profile: atom('default')
      }
    }
  }
})

const { createPluginContext } = await import('@/contrib/plugin')
// @ts-expect-error The bundled Bot Mode entry intentionally remains plain ESM JavaScript for older runtime loaders.
const { default: plugin } = await import('../plugin.js')

const flushSweep = async () => {
  await vi.advanceTimersByTimeAsync(0)
  await Promise.resolve()
}

afterEach(() => {
  gatewayState.set('closed')
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('Bot Mode hidden-session reconciliation lifecycle', () => {
  it('uses persisted REST on load/reconnect and stops with plugin disposal', async () => {
    vi.useFakeTimers()
    const disposers: Array<() => void> = []

    plugin.register(createPluginContext(plugin.id, dispose => disposers.push(dispose)))

    gatewayState.set('open')
    await flushSweep()
    gatewayState.set('closed')
    gatewayState.set('open')
    await flushSweep()
    gatewayState.set('closed')
    gatewayState.set('open')
    await flushSweep()

    expect(listPersistedSessions).toHaveBeenCalledTimes(6)
    expect(listPersistedSessions.mock.calls.map(([, options]) => options.profile)).toEqual([
      'alpha',
      'beta',
      'alpha',
      'beta',
      'alpha',
      'beta'
    ])
    expect(setPersistedSessionHidden).toHaveBeenCalledTimes(6)
    expect(setPersistedSessionHidden.mock.calls.map(([, options]) => options)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ hidden: true, profile: 'alpha', sessionId: 'alpha-bot' }),
        expect.objectContaining({ hidden: true, profile: 'beta', sessionId: 'beta-bot' })
      ])
    )
    expect(request.mock.calls.some(([method]) => method === 'session.list' || method === 'session.set_hidden')).toBe(false)

    disposers.forEach(dispose => dispose())
    const readsAtDispose = listPersistedSessions.mock.calls.length

    gatewayState.set('closed')
    gatewayState.set('open')
    await flushSweep()

    expect(listPersistedSessions).toHaveBeenCalledTimes(readsAtDispose)
  })
})