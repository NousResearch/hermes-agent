import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CodingWorkspaceSetting } from './coding-workspace-setting'

const mocks = vi.hoisted(() => ({ read: vi.fn(), save: vi.fn(), error: vi.fn() }))
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: (scope: unknown) => mocks.read(scope),
  saveHermesConfigRecord: (config: unknown, scope: unknown) => mocks.save(config, scope),
  profileScopeKey: (scope: unknown) => JSON.stringify(scope)
}))
vi.mock('@/store/notifications', () => ({ notifyError: (...args: unknown[]) => mocks.error(...args) }))
vi.mock('@/i18n', async () => {
  const { en } = await import('@/i18n/en')

  return { useI18n: () => ({ t: en }) }
})

afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('coding controls setting', () => {
  it('rolls back a refused write without changing another connection cache', async () => {
    const scope = { connectionId: 'remote-a', profile: 'coder' }
    const other = { connectionId: 'remote-b', profile: 'coder' }
    const original = { desktop: { coding: { show_controls: false } } }
    mocks.read.mockResolvedValue(original)
    mocks.save.mockResolvedValue({ ok: false })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const otherKey = ['hermes-config-record', JSON.stringify(other)]
    client.setQueryData(otherKey, { desktop: { coding: { show_controls: true } } })
    render(<QueryClientProvider client={client}><CodingWorkspaceSetting profile={scope} /></QueryClientProvider>)
    const toggle = await screen.findByRole('switch', { name: 'Show coding controls' })
    await waitFor(() => expect(toggle).toHaveProperty('disabled', false))
    await act(async () => fireEvent.click(toggle))
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    expect(mocks.error).toHaveBeenCalled()
    expect(client.getQueryData(otherKey)).toEqual({ desktop: { coding: { show_controls: true } } })
  })

  it('defaults off and writes only the captured profile/connection patch', async () => {
    const scope = { connectionId: 'remote', profile: 'coder' }
    mocks.read.mockResolvedValue({ desktop: { repo_scan_enabled: false }, model: 'unchanged' })
    mocks.save.mockResolvedValue({ ok: true })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}><CodingWorkspaceSetting profile={scope} /></QueryClientProvider>)
    const toggle = await screen.findByRole('switch', { name: 'Show coding controls' })
    await waitFor(() => expect(toggle).toHaveProperty('disabled', false))
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    await act(async () => fireEvent.click(toggle))
    expect(mocks.save).toHaveBeenCalledWith({ desktop: { coding: { show_controls: true } } }, scope)
    expect(mocks.read).toHaveBeenCalledWith(scope)
  })
})
