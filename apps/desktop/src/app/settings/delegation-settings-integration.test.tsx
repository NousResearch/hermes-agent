import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import { $activeConnectionId } from '@/store/connections'
import { $settingsRequestProfile } from '@/store/settings-scope'
import type { HermesConfigRecord } from '@/types/hermes'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/api/client')),
  ...(await vi.importActual('@/api/config')),
  ...(await vi.importActual('@/api/models')),
  getElevenLabsVoices: async () => ({ available: false }),
  getProfiles: async () => ({ profiles: [] })
}))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom<string | null>('connection-A') }))
vi.mock('@/store/settings-scope', () => ({
  $settingsRequestProfile: atom<string | undefined>('B'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>('B'),
  $settingsScopeProfile: atom('B')
}))
vi.mock('./profile-scope', () => ({ SettingsProfileScope: () => null }))
vi.mock('../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: () => {} }))
vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

const selectedProfile = $settingsRequestProfile as unknown as { set: (value: string) => void }
const activeConnection = $activeConnectionId as unknown as { set: (value: string) => void }
const requests: HermesApiRequest[] = []
const saves: HermesApiRequest[] = []
let config: HermesConfigRecord
let catalogError = false
let delayCatalog: ((request: HermesApiRequest) => Promise<unknown> | undefined) | undefined
let ConfigSettings: typeof ConfigSettingsType
beforeAll(async () => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)
beforeEach(() => {
  requests.length = 0
  saves.length = 0
  selectedProfile.set('B')
  activeConnection.set('connection-A')
  setApiRequestConnection('connection-A')
  setApiRequestProfile('A')
  config = { delegation: { provider: 'openai', model: 'gpt-5.1', max_iterations: 50 } }
  catalogError = false
  delayCatalog = undefined
  window.hermesDesktop = {
    ...window.hermesDesktop,
    api: vi.fn(async (request: HermesApiRequest) => {
      requests.push(request)

      if (request.path === '/api/config/schema') {
        return { fields: {} }
      }

      if (request.path.startsWith('/api/model/options')) {
        if (catalogError) {
          throw new Error('503')
        }

        const delayed = delayCatalog?.(request)

        if (delayed) {
          return delayed
        }

        const owner = `${request.connectionId}-${request.profile}`

        return {
          providers: [
            { name: 'OpenAI', slug: 'openai', models: ['gpt-5.1'] },
            { name: 'Anthropic', slug: 'anthropic', models: ['claude-sonnet-4-6'] },
            { name: owner, slug: owner, models: ['fixture-model'] }
          ]
        }
      }

      if (request.path === '/api/config' && request.method === 'PUT') {
        saves.push(request)

        return { ok: true }
      }

      if (request.path === '/api/config') {
        return structuredClone(config)
      }

      return { ok: true }
    })
  } as typeof window.hermesDesktop
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } })
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId="model" importInputRef={createRef<HTMLInputElement>()} subpage="delegation" />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

async function pick(label: string, name: string) {
  fireEvent.click(await screen.findByLabelText(label))
  fireEvent.click(await screen.findByRole('option', { name }))
}

async function debounce() {
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 750))
  })
}

describe('real settings route and autosave integration', () => {
  it('active A/settings B reads and saves B, never autosaves an unfinished route', async () => {
    mount()
    await pick('Subagent Provider', 'Anthropic')
    await debounce()
    expect(saves).toEqual([])
    await pick('Subagent Model', 'claude-sonnet-4-6')
    await debounce()
    expect(saves).toEqual([])
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(saves).toHaveLength(1))
    expect(saves[0]).toMatchObject({
      profile: 'B',
      connectionId: 'connection-A',
      body: { config: { delegation: { model: 'claude-sonnet-4-6', provider: 'anthropic' } } }
    })
    expect(requests.filter(r => r.path.startsWith('/api/model/options'))).toEqual([
      expect.objectContaining({ profile: 'B', connectionId: 'connection-A' })
    ])
  })
  it.each(['Inherit from main agent', 'Custom model (use parent credentials)', 'Anthropic'])(
    'switches away from a direct endpoint via %s and preserves request settings',
    async choice => {
      config = {
        delegation: {
          model: 'old-model',
          provider: 'openai',
          base_url: 'https://endpoint.invalid/v1',
          api_key: 'fixture-only',
          api_mode: 'chat_completions',
          request_overrides: { temperature: 0.2 },
          max_iterations: 50
        }
      }
      mount()
      expect((await screen.findByLabelText('Subagent Provider')).textContent).toBe('Direct endpoint override')
      await pick('Subagent Provider', choice)

      if (choice === 'Anthropic') {
        await pick('Subagent Model', 'claude-sonnet-4-6')
      }

      fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
      await waitFor(() => expect(saves).toHaveLength(1))
      const route = (saves[0].body as { config: HermesConfigRecord }).config.delegation as Record<string, unknown>
      expect(route.base_url).toBe('')
      expect(route.api_key).toBe('')
      expect(route).not.toHaveProperty('request_overrides')
      expect(route).not.toHaveProperty('api_mode')
      expect(route.model ?? 'old-model').toBe(
        choice === 'Inherit from main agent' ? '' : choice === 'Anthropic' ? 'claude-sonnet-4-6' : 'old-model'
      )
    }
  )
  it('keeps scopes separate across B to C to B and equal names on two connections', async () => {
    mount()
    await pick('Subagent Provider', 'connection-A-B')
    await act(async () => selectedProfile.set('C'))
    await pick('Subagent Provider', 'connection-A-C')
    expect(screen.queryByRole('option', { name: 'connection-A-B' })).toBeNull()
    await act(async () => selectedProfile.set('B'))
    await pick('Subagent Provider', 'connection-A-B')
    await act(async () => {
      setApiRequestConnection('connection-B')
      activeConnection.set('connection-B')
    })
    await pick('Subagent Provider', 'connection-B-B')
    await pick('Subagent Model', 'fixture-model')
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(saves).toHaveLength(1))
    expect(saves[0]).toMatchObject({ connectionId: 'connection-B', profile: 'B' })
    expect(requests.filter(r => r.path.startsWith('/api/model/options')).map(r => [r.connectionId, r.profile])).toEqual(
      [
        ['connection-A', 'B'],
        ['connection-A', 'C'],
        ['connection-B', 'B']
      ]
    )
  })
  it('saves a manual route after catalog failure and preserves it across retry', async () => {
    catalogError = true
    mount()
    await screen.findByText('Model catalog unavailable. Enter a provider and model manually, or retry.')
    await pick('Subagent Provider', 'Custom provider...')
    fireEvent.change(screen.getByLabelText('Custom subagent provider'), { target: { value: 'custom:lab' } })
    await pick('Subagent Model', 'Custom model…')
    fireEvent.change(screen.getByLabelText('Subagent Model'), { target: { value: 'lab-model' } })
    catalogError = false
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(requests.filter(r => r.path.startsWith('/api/model/options'))).toHaveLength(2))
    expect((screen.getByLabelText('Custom subagent provider') as HTMLInputElement).value).toBe('custom:lab')
    expect((screen.getByLabelText('Subagent Model') as HTMLInputElement).value).toBe('lab-model')
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(saves).toHaveLength(1))
    expect(saves[0]).toMatchObject({
      profile: 'B',
      connectionId: 'connection-A',
      body: { config: { delegation: { model: 'lab-model', provider: 'custom:lab' } } }
    })
  })
  it('ignores a late old-connection catalog after the settings owner changes', async () => {
    let resolveOld!: (value: unknown) => void
    delayCatalog = request =>
      request.connectionId === 'connection-A'
        ? new Promise(resolve => {
            resolveOld = resolve
          })
        : undefined
    mount()
    await waitFor(() => expect(requests.some(r => r.path.startsWith('/api/model/options'))).toBe(true))
    await act(async () => {
      setApiRequestConnection('connection-B')
      activeConnection.set('connection-B')
    })
    await pick('Subagent Provider', 'connection-B-B')
    await act(async () => resolveOld({ providers: [{ name: 'Old connection only', slug: 'old', models: ['old'] }] }))
    fireEvent.click(screen.getByLabelText('Subagent Provider'))
    expect(screen.queryByRole('option', { name: 'Old connection only' })).toBeNull()
    expect(screen.getByRole('option', { name: 'connection-B-B' })).toBeTruthy()
    expect(saves).toEqual([])
  })
  it('cancels a draft on unmount without sending it to autosave', async () => {
    mount()
    await pick('Subagent Provider', 'Anthropic')
    cleanup()
    await debounce()
    expect(saves).toEqual([])
  })
})
