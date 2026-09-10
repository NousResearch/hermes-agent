import type * as HermesSdk from '@hermes/plugin-sdk'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

const gatewayState = atom<'closed' | 'open'>('closed')

const listPersistedSessions = vi.fn(async (_route: unknown, options: { profile: string }) => ({
  sessions: [{ id: `${options.profile}-bot`, profile: options.profile, started_at: 1, title: 'Bot Chat' }]
}))

const setPersistedSessionHidden = vi.fn(
  async (_route: unknown, _options: { hidden: boolean; profile: string; sessionId: string }) => ({
    hidden: true,
    ok: true
  })
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
const { default: plugin } = await import('./plugin')

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
  it('keeps profile REST asleep across load and reconnects', async () => {
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

    expect(listPersistedSessions).not.toHaveBeenCalled()
    expect(setPersistedSessionHidden).not.toHaveBeenCalled()
    expect(request.mock.calls.some(([method]) => method === 'session.list' || method === 'session.set_hidden')).toBe(
      false
    )

    disposers.forEach(dispose => dispose())
    gatewayState.set('closed')
    gatewayState.set('open')
    await flushSweep()

    expect(listPersistedSessions).not.toHaveBeenCalled()
  })
})
