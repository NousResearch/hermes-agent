import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: (
    _connectionId: null | string,
    _profile: string,
    method: string,
    params?: Record<string, unknown>
  ) => requestGateway(method, params ?? {})
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { queryClient } from '@/lib/query-client'
import { $gatewayState } from '@/store/session'

import { VaultSettings } from './vault-settings'

stubResizeObserver()

beforeEach(() => {
  requestGateway.mockReset()
  queryClient.clear()
  $gatewayState.set('open')
})

afterEach(() => {
  cleanup()
  queryClient.clear()
})

it('refreshes revoked Settings access on reconnect and allows an immediate new unlock', async () => {
  let unlocked = true

  const login = {
    id: 'bw:work',
    kind: 'login',
    label: 'Work login',
    origin: 'https://example.com',
    created_at: '2026-08-01T12:00:00+00:00',
    backend: 'bitwarden'
  }

  requestGateway.mockImplementation(async (method: string) => {
    if (method === 'vault.sources') {
      return {
        sources: [
          { name: 'bitwarden', display_name: 'Bitwarden', enabled: true, needs_unlock: true, installed: true, unlocked }
        ]
      }
    }

    if (method === 'vault.list') {
      return { items: unlocked ? [login] : [] }
    }

    if (method === 'vault.unlock') {
      unlocked = true

      return { unlocked: true }
    }

    return {}
  })
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <VaultSettings />
      </QueryClientProvider>
    </MemoryRouter>
  )

  expect(await screen.findByRole('button', { name: 'Lock' })).toBeTruthy()
  expect(await screen.findByText('Work login')).toBeTruthy()

  act(() => $gatewayState.set('closed'))
  unlocked = false // The backend revokes the old Settings transport owner on disconnect.
  act(() => $gatewayState.set('open'))

  fireEvent.click(await screen.findByRole('button', { name: 'Unlock' }))
  await waitFor(() => expect(screen.queryByText('Work login')).toBeNull())
  const dialog = await screen.findByRole('dialog')
  fireEvent.change(within(dialog).getByPlaceholderText('Master password'), { target: { value: 'test-password' } })
  fireEvent.click(within(dialog).getByRole('button', { name: 'Unlock' }))

  await waitFor(() =>
    expect(requestGateway).toHaveBeenCalledWith('vault.unlock', { name: 'bitwarden', password: 'test-password' })
  )
  expect(await screen.findByRole('button', { name: 'Lock' })).toBeTruthy()
  expect(await screen.findByText('Work login')).toBeTruthy()
})
