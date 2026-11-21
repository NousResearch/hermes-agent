import { act, renderHook } from '@testing-library/react'
import { createElement, type ReactElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getStatus } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { noteServedProfiles } from '@/store/dead-session-prune'

import { useStatusSnapshot } from './use-status-snapshot'

vi.mock('@/hermes', () => ({
  getStatus: vi.fn()
}))

// The hook loads the prune lazily; the mock stands in for that dynamic import too.
vi.mock('@/store/dead-session-prune', () => ({
  noteServedProfiles: vi.fn()
}))

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

const wrapper = ({ children }: { children: ReactNode }): ReactElement =>
  createElement(I18nProvider, { configClient: null, initialLocale: 'en', children })

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.mocked(getStatus).mockReset()
  vi.mocked(noteServedProfiles).mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('useStatusSnapshot feeds the dead-session prune the served list', () => {
  const requestGateway = vi.fn().mockResolvedValue({}) as unknown as GatewayRequester

  it('passes the gateway\'s served profiles from an accepted status', async () => {
    vi.mocked(getStatus).mockResolvedValue({ active_sessions: 0, gateway_shared_with: ['default', 'work'] } as never)

    renderHook(() => useStatusSnapshot('open', requestGateway, 'local-default'), { wrapper })
    await flushAsync()
    await flushAsync()

    expect(noteServedProfiles).toHaveBeenCalledWith(['default', 'work'])
  })

  it('passes null for a standalone gateway, so every profile still counts as served', async () => {
    vi.mocked(getStatus).mockResolvedValue({ active_sessions: 0, gateway_shared_with: null } as never)

    renderHook(() => useStatusSnapshot('open', requestGateway, 'local-default'), { wrapper })
    await flushAsync()
    await flushAsync()

    expect(noteServedProfiles).toHaveBeenCalledWith(null)
  })

  it('does not pass anything from a signed-out status body (it carries no served list)', async () => {
    vi.mocked(getStatus).mockResolvedValue({ auth_required: true } as never)

    renderHook(() => useStatusSnapshot('open', requestGateway, 'local-default'), { wrapper })
    await flushAsync()
    await flushAsync()

    expect(noteServedProfiles).not.toHaveBeenCalled()
  })
})
