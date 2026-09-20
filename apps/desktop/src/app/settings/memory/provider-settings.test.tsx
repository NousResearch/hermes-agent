import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { openPluginInstallRequest } from '@/store/plugin-install-request'

import { MemoryProviderSettings } from './provider-settings'

const api = vi.hoisted(() => ({ status: vi.fn(), select: vi.fn(), config: vi.fn(), oauth: vi.fn() }))

vi.mock('@/hermes', () => ({
  getMemoryStatus: api.status,
  setMemoryProvider: api.select,
  getMemoryProviderConfig: api.config,
  getMemoryProviderOAuthStatus: api.oauth,
  startMemoryProviderOAuth: api.oauth
}))
vi.mock('@/store/plugin-install-request', () => ({ openPluginInstallRequest: vi.fn() }))

const owner = { connectionId: 'local', profile: 'alpha' }
const second = { name: 'second', description: 'Second provider', configured: false, status: 'needs_config' }
const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

function LocationProbe() {
  const { pathname, search, state } = useLocation()

  return <output aria-label="location">{`${pathname}${search} ${JSON.stringify(state)}`}</output>
}

function mount(active: string) {
  api.status.mockResolvedValue({ active, builtin_files: { memory: 0, user: 0 }, providers: [second] })
  api.config.mockResolvedValue({ name: 'second', label: 'Second', docs_url: '', fields: [] })
  api.oauth.mockResolvedValue({ supported: false })
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <MemoryProviderSettings profile={owner} />
        <LocationProbe />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  client.clear()
  vi.resetAllMocks()
})

it('lists providers with readiness; Configure opens the declared schema without selecting; Explore hands over the owner', async () => {
  mount('builtin')
  expect(await screen.findByText('Built-in memory')).toBeTruthy()
  expect(screen.getByText('Needs configuration')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Configure second' }))
  await waitFor(() => expect(api.config).toHaveBeenCalledWith('second', owner))
  expect(api.select).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('link', { name: 'Explore memory plugins' }))
  const state = JSON.stringify({ capabilityScope: owner })
  expect(screen.getByLabelText('location').textContent).toBe(`/capabilities?tab=plugins ${state}`)
})

it('a provider that connects becomes selectable without Retry', async () => {
  mount('builtin')
  const idle = { state: 'idle', connected: false, auth: null, detail: '' }
  api.oauth
    .mockResolvedValueOnce(idle)
    .mockResolvedValue({ ...idle, state: 'connected', connected: true, auth: 'oauth' })
  fireEvent.click(await screen.findByRole('button', { name: 'Configure second' }))
  const use = (await screen.findByRole('button', { name: 'Use provider' })) as HTMLButtonElement
  expect(use.disabled).toBe(true)
  api.status.mockResolvedValue({
    active: 'builtin',
    builtin_files: { memory: 0, user: 0 },
    providers: [{ ...second, status: 'ready' }]
  })
  fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))
  await waitFor(() => expect(use.disabled).toBe(false))
  expect(api.select).not.toHaveBeenCalled()
})

it('a missing selected provider offers Install from Git with the memory origin', async () => {
  mount('lost')
  fireEvent.click(await screen.findByRole('button', { name: 'Configure lost' }))
  fireEvent.click(screen.getByRole('button', { name: /Install from Git/ }))
  expect(openPluginInstallRequest).toHaveBeenCalledWith(
    expect.objectContaining({ profile: owner, origin: { kind: 'memory', providerId: 'lost' } })
  )
})
