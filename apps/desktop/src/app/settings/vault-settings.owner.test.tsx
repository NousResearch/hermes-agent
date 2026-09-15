import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

// Every vault RPC is routed to the OWNER connection/profile socket; the mock records both
// identities so the tests can prove a profile name cannot cross connection owners.
const { calls } = vi.hoisted(() => ({
  calls: [] as { method: string; params: Record<string, unknown>; profile: string; connectionId: null | string }[]
}))

let respond: (connectionId: null | string, profile: string, method: string) => Promise<unknown> = async () => ({})

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: (
    connectionId: null | string,
    profile: string,
    method: string,
    params?: Record<string, unknown>
  ) => {
    calls.push({ method, params: params ?? {}, profile, connectionId })

    return respond(connectionId, profile, method)
  },
  requestGatewayForProfile: (profile: string, method: string, params?: Record<string, unknown>) => {
    calls.push({ method, params: params ?? {}, profile, connectionId: null })

    return respond(null, profile, method)
  }
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { useStore } from '@nanostores/react'

import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { vaultOwnerKey, VaultSettings } from './vault-settings'

stubResizeObserver()

const sources = [
  { name: 'bitwarden', display_name: 'Bitwarden', enabled: true, needs_unlock: true, unlocked: false, installed: true }
]

// Mirrors the production mount site (settings/index.tsx): the panel is keyed by its owner, so an
// owner change remounts it and every dialog/draft is gone by construction.
function KeyedVault() {
  const profile = useStore($settingsScopeProfile)

  const connectionId = useStore($activeConnectionId)

  return <VaultSettings key={vaultOwnerKey(connectionId, profile)} />
}

function mount() {
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <KeyedVault />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  calls.length = 0
  queryClient.clear()
  $connection.set(null)
  $activeGatewayProfile.set('default')
  $gatewayState.set('open')
  respond = async (_connectionId, _profile, method) =>
    method === 'vault.sources' ? { sources } : method === 'vault.list' ? { items: [] } : { ok: true }
})

afterEach(() => {
  cleanup()
  queryClient.clear()
})

it('a master-password draft is wiped on a profile switch and never submitted to the new owner', async () => {
  mount()
  fireEvent.click(await screen.findByRole('button', { name: 'Unlock' }))
  fireEvent.change(screen.getByPlaceholderText('Master password'), { target: { value: 'password-for-A' } })

  act(() => $activeGatewayProfile.set('other-profile'))

  await waitFor(() => expect(screen.queryByPlaceholderText('Master password')).toBeNull())
  expect(calls.filter(c => c.method === 'vault.unlock')).toHaveLength(0)
  // Reads for the new owner target the new profile, not the old one.
  await waitFor(() => expect(calls.some(c => c.profile === 'other-profile' && c.method === 'vault.list')).toBe(true))
  expect(calls.filter(c => c.profile === 'other-profile').every(c => c.params.profile === 'other-profile')).toBe(true)
})

it('a late list response from profile A never paints under profile B', async () => {
  let resolveA!: (value: unknown) => void
  const held = new Promise(r => (resolveA = r))

  respond = async (_connectionId, profile, method) => {
    if (method === 'vault.sources') {
      return { sources }
    }

    if (profile === 'default' && method === 'vault.list') {
      return held
    }

    return { items: [] }
  }

  mount()
  await waitFor(() => expect(calls.some(c => c.profile === 'default' && c.method === 'vault.list')).toBe(true))

  act(() => $activeGatewayProfile.set('other-profile'))
  await waitFor(() => expect(calls.some(c => c.profile === 'other-profile' && c.method === 'vault.list')).toBe(true))

  await act(async () => {
    resolveA({
      items: [
        {
          id: 'a',
          kind: 'login',
          label: 'A-only private account',
          origin: 'https://a.example',
          identifier: 'a@example.com',
          created_at: ''
        }
      ]
    })
    await held
  })
  expect(screen.queryByText('A-only private account')).toBeNull()
})

it('vault.add secrets never enter the mutation cache', async () => {
  respond = async (_connectionId, _profile, method) =>
    method === 'vault.sources' ? { sources } : method === 'vault.list' ? { items: [] } : { id: 'created' }
  const view = mount()
  fireEvent.click(await screen.findByRole('button', { name: 'Add' }))

  for (const [label, value] of [
    ['Label', 'fixture'],
    ['Site origin', 'https://example.com'],
    ['Identifier', 'fixture@example.com'],
    ['Password', 'fixture-retained-password']
  ] as const) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } })
  }

  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(calls.some(c => c.method === 'vault.add')).toBe(true))
  expect(calls.find(c => c.method === 'vault.add')?.params.profile).toBe('default')
  expect((calls.find(c => c.method === 'vault.add')!.params.secret as Record<string, string>).password).toBe(
    'fixture-retained-password'
  )
  await waitFor(() => expect(screen.queryByLabelText('Password')).toBeNull())
  view.unmount()
  expect(
    JSON.stringify(
      queryClient
        .getMutationCache()
        .getAll()
        .map(m => m.state.variables)
    )
  ).not.toContain('fixture-retained-password')
})

it('routes a remote owner profile that is absent locally for load and save', async () => {
  act(() =>
    $connection.set({
      baseUrl: 'https://remote.example',
      isFullscreen: false,
      nativeOverlayWidth: 0,
      token: 'test-token',
      wsUrl: 'wss://remote.example',
      logs: [],
      windowButtonPosition: null,
      connectionId: 'remote-a',
      registryScoped: true,
      mode: 'remote',
      profile: 'remote-only'
    })
  )
  act(() => $activeGatewayProfile.set('remote-only'))

  respond = async (connectionId, profile, method) => {
    expect(connectionId).toBe('remote-a')
    expect(profile).toBe('remote-only')

    return method === 'vault.sources' ? { sources } : method === 'vault.list' ? { items: [] } : { id: 'remote-item' }
  }

  mount()
  await waitFor(() => expect(calls.some(c => c.method === 'vault.list')).toBe(true))
  fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
  fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'Remote item' } })
  fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'https://remote.example' } })
  fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'remote@example.com' } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'remote-secret' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))

  await waitFor(() => expect(calls.some(c => c.method === 'vault.add')).toBe(true))
  expect(
    calls
      .filter(c => c.method === 'vault.list' || c.method === 'vault.add')
      .every(c => c.connectionId === 'remote-a' && c.profile === 'remote-only')
  ).toBe(true)
  expect(
    calls
      .filter(c => c.method === 'vault.list' || c.method === 'vault.add')
      .every(c => c.params.profile === 'remote-only')
  ).toBe(true)
})

it('keeps same-named profiles isolated when switching connections A to B to A', async () => {
  respond = async (connectionId, profile, method) =>
    method === 'vault.sources'
      ? { sources }
      : method === 'vault.list'
        ? {
            items: connectionId
              ? [{ id: connectionId, kind: 'login', label: connectionId, origin: null, created_at: '' }]
              : []
          }
        : { ok: true }

  const connection = (connectionId: string) => ({
    baseUrl: `https://${connectionId}.example`,
    isFullscreen: false,
    nativeOverlayWidth: 0,
    token: 'test-token',
    wsUrl: `wss://${connectionId}.example`,
    logs: [],
    windowButtonPosition: null,
    connectionId,
    registryScoped: true,
    mode: 'remote' as const,
    profile: 'default'
  })

  $activeGatewayProfile.set('default')

  act(() => $connection.set(connection('source-a')))
  mount()
  await waitFor(() => expect(calls.some(c => c.method === 'vault.list' && c.connectionId === 'source-a')).toBe(true))

  act(() => $connection.set(connection('source-b')))
  await waitFor(() => expect(calls.some(c => c.method === 'vault.list' && c.connectionId === 'source-b')).toBe(true))
  expect(screen.queryByText('source-a')).toBeNull()
  fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
  fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'B item' } })
  fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'https://b.example' } })
  fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'b@example.com' } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'b-secret' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() =>
    expect(calls.some(c => c.method === 'vault.add' && c.connectionId === 'source-b' && c.profile === 'default')).toBe(
      true
    )
  )
  expect(calls.find(c => c.method === 'vault.add' && c.connectionId === 'source-b')?.params.profile).toBe('default')
  await waitFor(() => expect(screen.queryByLabelText('Password')).toBeNull())

  act(() => $connection.set(connection('source-a')))
  await waitFor(() => expect(screen.getByText('source-a')).toBeTruthy())
  expect(screen.getByText('source-a')).toBeTruthy()
  expect(calls.filter(c => c.method === 'vault.list').every(c => c.profile === 'default')).toBe(true)
  expect(calls.filter(c => c.method === 'vault.list').every(c => c.params.profile === 'default')).toBe(true)
})
