import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

const { requestGateway } = vi.hoisted(() => ({
  requestGateway: vi.fn()
}))

// The panel routes every RPC through the owner connection/profile socket (never the ambient gateway);
// this legacy fixture ignores owner identity and keeps its method/params assertions focused.
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: (
    _connectionId: null | string,
    _profile: string,
    method: string,
    params?: Record<string, unknown>
  ) => requestGateway(method, params ?? {})
}))

import { queryClient } from '@/lib/query-client'
import { $gatewayState } from '@/store/session'

import { VaultSettings } from './vault-settings'

stubResizeObserver()

const renderVault = (route = '/settings?tab=vault') =>
  render(
    <MemoryRouter initialEntries={[route]}>
      <QueryClientProvider client={queryClient}>
        <VaultSettings />
      </QueryClientProvider>
    </MemoryRouter>
  )

const LOGIN_ITEM = {
  id: 'vault_abc123',
  kind: 'login',
  label: 'GitHub work',
  origin: 'https://github.com',
  created_at: '2026-08-01T12:00:00+00:00',
  identifier: 'me@example.com',
  identifier_type: 'email'
}

beforeEach(() => {
  requestGateway.mockReset()
  queryClient.clear()
  $gatewayState.set('open')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('VaultSettings', () => {
  it('shows the empty state when the vault has no items', async () => {
    requestGateway.mockResolvedValue({ items: [] })
    renderVault()

    await waitFor(() => expect(screen.getByText('Nothing saved yet')).toBeTruthy())
    expect(requestGateway).toHaveBeenCalledWith('vault.list', { profile: 'default' })
  })

  it('lists items with label, kind badge, identifier, and origin — never passwords', async () => {
    requestGateway.mockResolvedValue({ items: [LOGIN_ITEM] })
    renderVault()

    await waitFor(() => expect(screen.getByText('GitHub work')).toBeTruthy())
    expect(screen.getByText('Login')).toBeTruthy()
    // Identifier is agent-visible metadata and now shows in the row.
    expect(screen.getByText('me@example.com')).toBeTruthy()
    expect(screen.getByText('https://github.com')).toBeTruthy()
  })

  it('opens the Add dialog pre-filled from deep-link query params (never secrets)', async () => {
    requestGateway.mockResolvedValue({ items: [] })
    renderVault('/settings?tab=vault&kind=login&label=github&origin=https://github.com')

    await waitFor(() => expect(screen.getByLabelText('Label')).toBeTruthy())
    expect((screen.getByLabelText('Label') as HTMLInputElement).value).toBe('github')
    expect((screen.getByLabelText('Site origin') as HTMLInputElement).value).toBe('https://github.com')
    // The password field always starts empty — a secret can never arrive via link.
    expect((screen.getByLabelText('Password') as HTMLInputElement).value).toBe('')
  })

  it('validates the origin before submitting a login item', async () => {
    requestGateway.mockResolvedValue({ items: [] })
    renderVault()

    fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
    fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'x' } })
    fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'not-a-url' } })
    fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'me@example.com' } })
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'pw' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(screen.getByText('Enter a valid URL like https://example.com.')).toBeTruthy())
    expect(requestGateway).not.toHaveBeenCalledWith('vault.add', expect.anything())
  })

  it('submits vault.add and refetches the list on success', async () => {
    requestGateway.mockImplementation(async (method: string) =>
      method === 'vault.list' ? { items: [] } : { id: 'vault_new' }
    )
    renderVault()

    fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
    fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'GitHub work' } })
    fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'https://github.com' } })
    fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'me@example.com' } })
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 's3cret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('vault.add', {
        profile: 'default',
        kind: 'login',
        label: 'GitHub work',
        origin: 'https://github.com',
        secret: {
          identifier_type: 'email',
          identifier: 'me@example.com',
          password: 's3cret'
        }
      })
    )
  })

  it('deletes an item through the confirm dialog', async () => {
    requestGateway.mockImplementation(async (method: string) =>
      method === 'vault.list' ? { items: [LOGIN_ITEM] } : { removed: true }
    )
    renderVault()

    await waitFor(() => expect(screen.getByText('GitHub work')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Remove saved item' }))
    await waitFor(() => expect(screen.getByText('Delete this item?')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('vault.remove', { profile: 'default', id: 'vault_abc123' })
    )
  })

  it('unlocks a password manager from Settings; the master password leaves only via vault.unlock', async () => {
    const sources = [
      {
        name: 'onepassword',
        display_name: '1Password',
        enabled: true,
        needs_unlock: true,
        unlocked: false,
        installed: true
      },
      {
        name: 'bitwarden',
        display_name: 'Bitwarden',
        enabled: false,
        needs_unlock: true,
        unlocked: false,
        installed: false
      }
    ]

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'vault.list') {
        // An external item has no delete affordance; its manager is shown as a source badge instead.
        return { items: [{ ...LOGIN_ITEM, id: 'op:xyz', label: 'GitHub via 1Password', backend: 'onepassword' }] }
      }

      if (method === 'vault.sources') {
        return { sources: sources.map(source => ({ ...source })) }
      }

      if (method === 'vault.unlock') {
        sources[0] = { ...sources[0], unlocked: true }

        return { unlocked: true }
      }

      return {}
    })
    renderVault()

    await waitFor(() => expect(screen.getByText('GitHub via 1Password')).toBeTruthy())
    expect(screen.queryByRole('button', { name: 'Remove saved item' })).toBeNull()
    // A manager that isn't installed has nothing to switch (detection is automatic); the installed one can be unlocked.
    expect(screen.queryByRole('switch', { name: 'Bitwarden' })).toBeNull()
    expect(screen.getByRole('switch', { name: '1Password' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Unlock' }))
    await waitFor(() => expect(screen.getByText('Unlock 1Password')).toBeTruthy())

    fireEvent.change(screen.getByPlaceholderText('Master password'), { target: { value: 'correct horse' } })
    fireEvent.click(
      screen.getByRole('button', { name: 'Unlock' }).closest('form')!.querySelector('button[type=submit]')!
    )

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('vault.unlock', {
        profile: 'default',
        name: 'onepassword',
        password: 'correct horse'
      })
    )
    await waitFor(() => expect(screen.getByText('Unlocked')).toBeTruthy())
    expect(screen.queryByPlaceholderText('Master password')).toBeNull()
    expect(screen.getByRole('button', { name: 'Lock' })).toBeTruthy()
  })

  it('refreshes password-manager detection when the page is reopened', async () => {
    let installed = false
    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'vault.sources') {
        return {
          sources: [
            {
              name: 'onepassword',
              display_name: '1Password',
              enabled: false,
              needs_unlock: true,
              unlocked: false,
              installed
            }
          ]
        }
      }

      return { items: [] }
    })

    const first = renderVault()
    await screen.findByText('Not detected')
    first.unmount()

    installed = true
    renderVault()

    await waitFor(() => expect(screen.getByRole('switch', { name: '1Password' })).toBeTruthy())
    expect(requestGateway.mock.calls.filter(([method]) => method === 'vault.sources')).toHaveLength(2)
  })

  it('refreshes password-manager detection after remounting while the gateway is closed', async () => {
    let installed = false
    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'vault.sources') {
        return {
          sources: [
            {
              name: 'onepassword',
              display_name: '1Password',
              enabled: false,
              needs_unlock: true,
              unlocked: false,
              installed
            }
          ]
        }
      }

      return { items: [] }
    })

    const first = renderVault()
    await screen.findByText('Not detected')
    first.unmount()

    installed = true
    $gatewayState.set('closed')
    renderVault()
    act(() => $gatewayState.set('open'))

    await waitFor(() => expect(screen.getByRole('switch', { name: '1Password' })).toBeTruthy())
    expect(requestGateway.mock.calls.filter(([method]) => method === 'vault.sources')).toHaveLength(2)
  })
})
