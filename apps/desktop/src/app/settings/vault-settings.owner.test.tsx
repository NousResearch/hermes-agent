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

let respond: (profile: string, method: string, params: Record<string, unknown>) => Promise<unknown> = async () => ({})

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: (
    _connectionId: null | string,
    profile: string,
    method: string,
    params?: Record<string, unknown>
  ) => {
    const requestParams = params ?? {}
    calls.push({ method, params: requestParams, profile })

    return respond(profile, method, requestParams)
  }
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
// Partial mock: the profile switch under test also fires module-level
// subscribers elsewhere in the store graph (cron-model-impact dismisses its
// notification), so only the two calls this suite asserts on are stubbed.
vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  notify: vi.fn(),
  notifyError: vi.fn()
}))

import { useStore } from '@nanostores/react'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import { $settingsScopeOverride, $settingsScopeProfile, setSettingsScope } from '@/store/settings-scope'

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
  $settingsScopeOverride.set(null)
  $gatewayState.set('open')
  respond = async (_profile, method) =>
    method === 'vault.sources' ? { sources } : method === 'vault.list' ? { items: [] } : { ok: true }
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
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

it('sends the selected settings profile with vault reads and writes', async () => {
  const selectedProfile = 'credentials-owner'
  setSettingsScope(selectedProfile)
  mount()

  await waitFor(() => expect(calls.some(call => call.method === 'vault.list')).toBe(true))
  fireEvent.click(await screen.findByRole('button', { name: 'Add' }))
  fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'Synthetic fixture' } })
  fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'https://example.invalid' } })
  fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'fixture@example.invalid' } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'synthetic-placeholder' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))

  await waitFor(() => expect(calls.some(call => call.method === 'vault.add')).toBe(true))
  expect.soft(calls.find(call => call.method === 'vault.list')?.params.profile).toBe(selectedProfile)
  expect.soft(calls.find(call => call.method === 'vault.add')?.params.profile).toBe(selectedProfile)
})

it('keeps every synthetic custom-home vault RPC ambient while retaining the custom route owner', async () => {
  let unlocked = false
  let enabled = true
  const customItem = {
    id: 'custom-home-item',
    kind: 'login',
    label: 'Custom-home existing item',
    origin: 'https://custom-home.invalid',
    identifier: 'existing@custom-home.invalid',
    created_at: '2026-09-25T00:00:00Z'
  }

  respond = async (_profile, method, params) => {
    if (method === 'vault.sources') {
      return {
        sources: [
          {
            name: 'bitwarden',
            display_name: 'Bitwarden',
            enabled,
            needs_unlock: true,
            unlocked,
            installed: true
          }
        ]
      }
    }

    if (method === 'vault.list') {
      return { items: [customItem] }
    }

    if (method === 'vault.unlock') {
      unlocked = true
      return { unlocked: true }
    }

    if (method === 'vault.lock') {
      unlocked = false
      return { locked: true }
    }

    if (method === 'vault.source.set') {
      enabled = Boolean(params.enabled)
      return { enabled }
    }

    if (method === 'vault.add') {
      return { id: 'custom-home-created' }
    }

    if (method === 'vault.remove') {
      return { removed: true }
    }

    return { ok: true }
  }

  act(() => $activeGatewayProfile.set('custom'))
  mount()

  await screen.findByText('Custom-home existing item')

  fireEvent.click(screen.getByRole('button', { name: 'Remove saved item' }))
  await screen.findByText('Delete this item?')
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  await waitFor(() => expect(calls.some(call => call.method === 'vault.remove')).toBe(true))

  fireEvent.click(screen.getByRole('button', { name: 'Add' }))
  fireEvent.change(screen.getByLabelText('Label'), { target: { value: 'Custom-home fixture' } })
  fireEvent.change(screen.getByLabelText('Site origin'), { target: { value: 'https://custom-home.invalid' } })
  fireEvent.change(screen.getByLabelText('Identifier'), { target: { value: 'fixture@custom-home.invalid' } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'synthetic-placeholder' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(calls.some(call => call.method === 'vault.add')).toBe(true))

  fireEvent.click(await screen.findByRole('button', { name: 'Unlock' }))
  await screen.findByText('Unlock Bitwarden')
  fireEvent.change(screen.getByPlaceholderText('Master password'), { target: { value: 'synthetic-master-password' } })
  fireEvent.click(
    screen.getByRole('button', { name: 'Unlock' }).closest('form')!.querySelector('button[type=submit]')!
  )
  await waitFor(() => expect(calls.some(call => call.method === 'vault.unlock')).toBe(true))

  fireEvent.click(await screen.findByRole('button', { name: 'Lock' }))
  await waitFor(() => expect(calls.some(call => call.method === 'vault.lock')).toBe(true))

  fireEvent.click(screen.getByRole('switch', { name: 'Bitwarden' }))
  await waitFor(() => expect(calls.some(call => call.method === 'vault.source.set')).toBe(true))

  const methods = [
    'vault.list',
    'vault.sources',
    'vault.source.set',
    'vault.unlock',
    'vault.lock',
    'vault.add',
    'vault.remove'
  ] as const

  for (const method of methods) {
    const methodCalls = calls.filter(call => call.method === method)
    expect.soft(methodCalls.length, method).toBeGreaterThan(0)

    for (const call of methodCalls) {
      expect.soft(call.profile, method + ' route owner').toBe('custom')
      expect.soft(call.params, method + ' payload').not.toHaveProperty('profile')
    }
  }
})
