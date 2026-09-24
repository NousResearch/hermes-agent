import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import type { MemoryProviderConfig } from '@/types/hermes'

import { MemoryProviderSettings } from './provider-settings'

// The real request layer runs so every path, body and owner pin is what the backend would receive.
const owner = { connectionId: 'local', profile: 'alpha' }
const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
let active = 'builtin'

const schema: MemoryProviderConfig = {
  name: 'alpha',
  label: 'Alpha',
  docs_url: '',
  capabilities: { save_without_activation: true },
  fields: [
    { key: 'token', label: 'token', kind: 'secret', value: '', inline: true, is_set: true } as never,
    { key: 'workspace', label: 'workspace', kind: 'text', value: 'initial', inline: true, is_set: true } as never
  ]
}

const api = vi.fn(async (request: HermesApiRequest) => {
  if (request.path === '/api/memory/provider') {
    active = (request.body as { provider: string }).provider

    return { ok: true, active }
  }

  if (request.path === '/api/memory') {
    return {
      active,
      builtin_files: { memory: 0, user: 0 },
      providers: [
        { name: 'alpha', description: 'Alpha provider', configured: true, status: 'ready' },
        { name: 'beta', description: 'Beta provider', configured: false, status: 'needs_config' }
      ]
    }
  }

  return request.path.includes('/oauth/') ? { supported: false } : request.method === 'PUT' ? { ok: true } : schema
})

const requests = (path: string, method = 'GET') =>
  api.mock.calls.map(([request]) => request).filter(r => r.path === path && (r.method ?? 'GET') === method)

function LocationProbe() {
  const { pathname, search, state } = useLocation()

  return <output aria-label="location">{`${pathname}${search} ${JSON.stringify(state)}`}</output>
}

afterEach(() => {
  cleanup()
  client.clear()
})

it('lists providers with readiness; Configure saves without selecting; Use provider selects and reads the owner back; built-in switches back', async () => {
  window.hermesDesktop = { api } as never
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <MemoryProviderSettings owner={owner} />
        <LocationProbe />
      </QueryClientProvider>
    </MemoryRouter>
  )

  expect(await screen.findByText('Built-in memory')).toBeTruthy()
  expect(screen.getByText('Active')).toBeTruthy()
  expect(screen.getAllByText('Ready')).toHaveLength(2)
  expect(screen.getByText('Needs configuration')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Configure alpha' }))
  const configPath = '/api/memory/providers/alpha/config?surface=declared'
  await waitFor(() => expect(requests(configPath)).toMatchObject([{ profile: 'alpha', connectionId: 'local' }]))
  expect(requests('/api/memory/provider', 'PUT')).toEqual([])

  const token = (await screen.findByLabelText('token')) as HTMLInputElement
  fireEvent.change(screen.getByLabelText('workspace'), { target: { value: 'edited' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
  await screen.findByRole('status')
  expect(requests(configPath, 'PUT')).toMatchObject([
    { profile: 'alpha', connectionId: 'local', body: { values: { workspace: 'edited' }, activate: false } }
  ])
  expect(token.value).toBe('')

  const readsBeforeSelect = requests('/api/memory').length
  fireEvent.click(screen.getByRole('button', { name: 'Use provider' }))
  expect(await screen.findByRole('button', { name: 'Active' })).toBeTruthy()
  expect(requests('/api/memory/provider', 'PUT')).toMatchObject([
    { profile: 'alpha', connectionId: 'local', body: { provider: 'alpha' } }
  ])
  expect(requests('/api/memory')).toHaveLength(readsBeforeSelect + 1)

  fireEvent.click(screen.getByRole('button', { name: 'Configure builtin' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Use provider' }))
  expect(await screen.findByRole('button', { name: 'Active' })).toBeTruthy()
  expect(requests('/api/memory/provider', 'PUT').at(-1)?.body).toEqual({ provider: 'builtin' })

  fireEvent.click(screen.getByRole('link', { name: 'Explore memory plugins' }))
  const state = JSON.stringify({ capabilityScope: owner })
  expect(screen.getByLabelText('location').textContent).toBe(`/capabilities?tab=plugins ${state}`)
})
