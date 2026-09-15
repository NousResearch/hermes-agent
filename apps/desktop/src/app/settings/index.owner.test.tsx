import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { ConfirmHost } from '@/components/confirm-host'
import type { HermesApiRequest } from '@/global'
import { queryClient } from '@/lib/query-client'
import { $confirmRequest } from '@/store/confirm'
import { $profiles } from '@/store/profile'
import { $settingsScopeOverride, setSettingsScope } from '@/store/settings-scope'

import { ConfigSettings } from './config-settings'

import { SettingsView } from './index'

const calls: HermesApiRequest[] = []
let rejectConfig = false
let readDefaults: () => Promise<unknown> = async () => ({ checkpoints: { enabled: true } })

beforeEach(() => {
  calls.length = 0
  rejectConfig = false
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  $profiles.set([])
  setSettingsScope({ connectionId: 'fixture-lab', profile: 'research' })
  setApiRequestConnection('local')
  vi.stubGlobal('hermesDesktop', {
    api: vi.fn(async (request: HermesApiRequest) => {
      calls.push(request)

      if (request.path === '/api/config/defaults') {
        return readDefaults()
      }

      if (request.path === '/api/config' || request.path === '/api/config/schema') {
        if (rejectConfig) {
          throw new Error('Fixture gateway unavailable')
        }

        return request.path.endsWith('/schema') ? { fields: {} } : { checkpoints: { enabled: false } }
      }

      if (request.path === '/api/profiles') {
        return { profiles: [] }
      }

      return { available: false }
    })
  })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $settingsScopeOverride.set(null)
  $confirmRequest.set(null)
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
})

it('keeps the owner selector available when config/schema fails, without reading another gateway', async () => {
  rejectConfig = true
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ConfigSettings activeSectionId="safety" importInputRef={createRef<HTMLInputElement>()} />
      </QueryClientProvider>
    </MemoryRouter>
  )
  expect(screen.getByRole('button', { name: /Applies to/ })).toBeTruthy()
  await screen.findByRole('button', { name: /Refresh/ })
  expect(screen.getByRole('button', { name: /Applies to.*research.*fixture-lab/ })).toBeTruthy()
  expect(
    calls
      .filter(request => request.path.startsWith('/api/config'))
      .every(request => request.connectionId === 'fixture-lab' && request.profile === 'research')
  ).toBe(true)
})

it('Settings reset reads and writes one owner and cancels a deferred reset after selection changes', async () => {
  render(
    <MemoryRouter initialEntries={['/settings?tab=config:safety']}>
      <QueryClientProvider client={queryClient}>
        <SettingsView onClose={() => {}} />
        <ConfirmHost />
      </QueryClientProvider>
    </MemoryRouter>
  )
  fireEvent.click(screen.getAllByRole('button', { name: 'Reset to defaults' })[0])
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Reset to defaults' }))
  await waitFor(() => expect(calls.some(request => request.method === 'PUT')).toBe(true))
  const resetCalls = calls.filter(request => request.path === '/api/config/defaults' || request.method === 'PUT')
  expect(resetCalls.map(request => [request.connectionId, request.profile])).toEqual([
    ['fixture-lab', 'research'],
    ['fixture-lab', 'research']
  ])
  let resolve!: (value: unknown) => void
  readDefaults = () =>
    new Promise(r => {
      resolve = r
    })
  fireEvent.click(screen.getAllByRole('button', { name: 'Reset to defaults' })[0])
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Reset to defaults' }))
  await waitFor(() => expect(resolve).toBeTypeOf('function'))
  act(() => setSettingsScope({ connectionId: 'local', profile: 'research' }))
  await act(async () => {
    resolve({ checkpoints: { enabled: true } })
  })
  expect(calls.filter(request => request.method === 'PUT')).toHaveLength(1)
})
