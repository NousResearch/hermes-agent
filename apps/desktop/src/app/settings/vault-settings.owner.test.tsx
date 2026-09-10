import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

// Every vault RPC is routed to the OWNER profile's socket; the mock records which profile each
// call targeted so the tests can prove a draft never crosses owners.
const { calls } = vi.hoisted(() => ({
  calls: [] as { method: string; params: Record<string, unknown>; profile: string }[]
}))
let respond: (profile: string, method: string) => Promise<unknown> = async () => ({})

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForProfile: (profile: string, method: string, params?: Record<string, unknown>) => {
    calls.push({ method, params: params ?? {}, profile })

    return respond(profile, method)
  }
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { useStore } from '@nanostores/react'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
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

  return <VaultSettings key={vaultOwnerKey(null, profile)} />
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
  $activeGatewayProfile.set('default')
  $gatewayState.set('open')
  respond = async (_profile, method) =>
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
})

it('a late list response from profile A never paints under profile B', async () => {
  let resolveA!: (value: unknown) => void
  const held = new Promise(r => (resolveA = r))

  respond = async (profile, method) => {
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
  respond = async (_profile, method) =>
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
