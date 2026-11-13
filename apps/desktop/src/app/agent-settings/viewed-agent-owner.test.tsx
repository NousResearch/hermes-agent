import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import type * as GatewayStore from '@/store/gateway'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { setGatewayState } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'
import { $shellChoice } from '@/store/shell-variant'
import type { ProfileInfo } from '@/types/hermes'

import { SettingsView } from './entry'

// The window is on `default`; the page shows `research`, another agent on the SAME machine. Every
// request below must name research, and nothing the window is on may move.
const OWNER = { connectionId: 'local', profile: 'research' }

const AMBIENT = { connectionId: 'local', profile: 'default' }

interface Request {
  path: string
  connectionId?: string
  profile?: string
  method?: string
  body?: unknown
}

const session = {
  id: 'history-1',
  title: 'Research notes',
  source: 'desktop',
  started_at: 1_700_000_000,
  last_active: 1_700_000_000,
  message_count: 3,
  preview: ''
}

const rpc = vi.fn(async (method: string, params?: Record<string, unknown>): Promise<unknown> => {
  if (method === 'connectors.list') {
    return { connectors: [] }
  }

  if (method === 'plugins.manage') {
    return params?.action === 'list'
      ? { plugins: [{ key: 'advanced-delegation', name: 'advanced-delegation', status: 'disabled', source: 'user' }] }
      : { ok: true }
  }

  if (method === 'session.active_list') {
    return { sessions: [] }
  }

  return {}
})

// The window's own socket: an embedded agent control reaching it is the bug under test.
const ambientRequest = vi.fn(async (method: string, params?: Record<string, unknown>) => rpc(method, params))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: ambientRequest })
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: vi.fn(
    async (_connectionId: string, _profile: string, method: string, params?: Record<string, unknown>) =>
      rpc(method, params)
  )
}))

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({ initialValue, onChange }: { initialValue: string; onChange: (value: string) => void }) => (
    <textarea
      aria-label="Instructions editor"
      defaultValue={initialValue}
      onChange={event => onChange(event.target.value)}
    />
  )
}))

const config = {
  model: { default: 'model-a', provider: 'nous' },
  display: { show_reasoning: false },
  browser: { use_real_profile: false },
  delegation: { backend: 'default' }
}

const routing = {
  backend: 'default',
  configured: 'default',
  backends: [
    { name: 'default', builtin: true, available: true },
    { name: 'model_map', builtin: false, available: false }
  ],
  model_map: {},
  profile_routes: {},
  profiles: []
}

function answer({ path, method = 'GET' }: Request): unknown {
  const pathname = path.split('?')[0]

  const routes: Record<string, unknown> = {
    '/api/skills': [],
    '/api/tools/toolsets': [],
    '/api/cron/jobs': method === 'POST' ? { id: 'saved', enabled: true } : [],
    '/api/messaging/platforms': {
      platforms: [
        {
          id: 'telegram',
          name: 'Telegram',
          description: '',
          docs_url: '',
          configured: true,
          enabled: false,
          env_vars: [],
          gateway_running: true,
          state: 'disabled'
        }
      ]
    },
    '/api/messaging/platforms/telegram': { ok: true, hot_served: false },
    '/api/pairing': { approved: [], pending: [] },
    '/api/profiles/sessions': { sessions: [session], total: 1, limit: 25, offset: 0 },
    '/api/profiles/research/soul': { content: 'Owner instructions' },
    '/api/profiles': {
      current: 'default',
      profiles: ['default', 'research'].map(name => ({ name, is_default: name === 'default', path: `/t/${name}` }))
    },
    '/api/config': method === 'PUT' ? { ok: true } : config,
    '/api/config/schema': { fields: {} },
    '/api/env': {
      FAL_KEY: { is_set: false, redacted_value: null, description: 'Image generation', category: 'tool' }
    },
    '/api/delegation/routing': routing,
    '/api/model/moa': { presets: {}, default_preset: 'custom', reference_models: [] },
    '/api/model/auxiliary': { tasks: [] },
    '/api/model/options': { providers: [] },
    [`/api/sessions/${session.id}`]: method === 'PATCH' ? { ok: true } : session,
    [`/api/sessions/${session.id}/mutation-snapshot`]: {
      session_id: session.id,
      exists: true,
      runtime_revision: 1,
      runtime_generation: 1
    },
    '/api/gateway/restart': { name: 'gateway-restart', ok: true },
    '/api/status': {}
  }

  return routes[pathname] ?? {}
}

const api = vi.fn(async (request: Request) => answer(request))

const originalDesktop = window.hermesDesktop

beforeAll(async () => {
  await import('./index')
}, 60_000)

beforeEach(() => {
  queryClient.clear()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  $profiles.set(
    ['default', 'research'].map(name => ({ name, is_default: name === 'default', path: `/t/${name}` })) as ProfileInfo[]
  )
  $activeGatewayProfile.set(AMBIENT.profile)
  $apiRequestScope.set({ ...AMBIENT })
  $settingsScopeOverride.set(null)
  $shellChoice.set(true)
  setGatewayState('open')
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
  window.hermesDesktop = originalDesktop
  $apiRequestScope.set({ profile: null, connectionId: null })
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $shellChoice.set(false)
  setGatewayState('idle')
})

function mount(search: string, connection = OWNER.connectionId) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[`/settings?agentProfile=research&connection=${connection}${search}`]}>
        <SettingsView onClose={() => {}} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

const restCalls = () => api.mock.calls.map(([request]) => request)

/** Every request carried the owner, nothing reached the window's socket, and the window did not move. */
function expectOwnerOnly() {
  expect(restCalls().length).toBeGreaterThan(0)

  for (const request of restCalls()) {
    expect({ path: request.path, connectionId: request.connectionId, profile: request.profile }).toEqual({
      path: request.path,
      ...OWNER
    })
  }

  for (const [connectionId, profile, method] of vi.mocked(requestGatewayForAgent).mock.calls) {
    expect({ method, connectionId, profile }).toEqual({ method, ...OWNER })
  }

  expect(ambientRequest.mock.calls).toEqual([])
  expect($activeGatewayProfile.get()).toBe(AMBIENT.profile)
  expect($apiRequestScope.get()).toEqual(AMBIENT)
  expect($settingsScopeOverride.get()).toBeNull()
  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
}

const write = (method: string, path: string) =>
  waitFor(() => {
    const found = restCalls().find(request => request.method === method && request.path.split('?')[0] === path)

    expect(found, restCalls().map(request => `${request.method ?? 'GET'} ${request.path}`).join(', ')).toBeDefined()

    return found!
  })

it('shows another agent on this machine without opening it, and saves a routine to it', async () => {
  mount('')
  await screen.findByText('Owner instructions')
  expect(await screen.findByText('Research notes')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: en.cron.newCron }))
  fireEvent.change(screen.getByPlaceholderText('0 9 * * *'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText(en.cron.promptLabel), { target: { value: 'Review notes' } })
  fireEvent.click(screen.getByRole('button', { name: en.common.save }))
  await write('POST', '/api/cron/jobs')

  for (const path of ['/api/config', '/api/skills', '/api/tools/toolsets', '/api/profiles/research/soul']) {
    expect(restCalls().some(request => request.path === path)).toBe(true)
  }

  expect(vi.mocked(requestGatewayForAgent).mock.calls.some(([, , method]) => method === 'connectors.list')).toBe(true)
  expectOwnerOnly()
})

it('reads and edits the viewed agent history while another agent is active', async () => {
  mount('&panel=history')
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Selected: Research notes' }))
  fireEvent.click(screen.getByRole('button', { name: 'Pin' }))
  const pin = await write('PATCH', `/api/sessions/${session.id}`)

  expect(pin.body).toMatchObject({ pinned: true, profile: 'research' })
  expect(requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'session.active_list', {
    profile: 'research'
  })
  expectOwnerOnly()
})

it('saves instructions to the viewed agent', async () => {
  mount('&panel=settings&tab=agent:instructions')
  fireEvent.change(await screen.findByRole('textbox', { name: 'Instructions editor' }), {
    target: { value: 'New research instructions' }
  })
  fireEvent.click(screen.getByRole('button', { name: en.profiles.saveSoul }))
  const save = await write('PUT', '/api/profiles/research/soul')

  expect(save.body).toEqual({ content: 'New research instructions' })
  expectOwnerOnly()
})

it('reads and autosaves a config page on the viewed agent', async () => {
  mount('&panel=settings&tab=agent:chat')
  fireEvent.click(await screen.findByRole('switch'))
  const save = await write('PUT', '/api/config')

  expect(JSON.stringify(save.body)).toContain('"show_reasoning":true')
  expect(restCalls().some(request => request.path === '/api/config/schema')).toBe(true)
  expectOwnerOnly()
})

it('asks the viewed agent, not the window, about its delegation plugin and enables it there', async () => {
  mount('&panel=settings&tab=config:model&page=delegation')
  fireEvent.click(await screen.findByRole('button', { name: en.settings.model.subagents.routing.enablePlugin }))
  await waitFor(() =>
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'local',
      'research',
      'plugins.manage',
      expect.objectContaining({ action: 'toggle', profile: 'research' }),
      undefined
    )
  )
  expect(requestGatewayForAgent).toHaveBeenCalledWith(
    'local',
    'research',
    'plugins.manage',
    expect.objectContaining({ action: 'list', profile: 'research' }),
    undefined
  )
  expectOwnerOnly()
})

it('reads and saves tool keys on the viewed agent', async () => {
  const { baseElement } = mount('&panel=settings&tab=agent:tools-advanced')
  // The rows regroup once the toolsets land, so the field is looked up again for every event.
  const key = () => within(baseElement.querySelector<HTMLElement>('#credential-key-FAL_KEY')!).getByRole('textbox')
  await waitFor(key)
  await waitFor(() => expect(requestGatewayForAgent).toHaveBeenCalled())
  fireEvent.focus(key())
  fireEvent.change(key(), { target: { value: 'fal-secret' } })
  fireEvent.keyDown(key(), { key: 'Enter' })
  const save = await write('PUT', '/api/env')

  expect(save.body).toMatchObject({ key: 'FAL_KEY', value: 'fal-secret' })
  expectOwnerOnly()
})

it('toggles a channel on the viewed agent and restarts its gateway, not the window one', async () => {
  mount('&panel=settings&tab=agent:messaging&platform=telegram')
  fireEvent.click(await screen.findByRole('switch', { name: en.messaging.enableAria('Telegram') }))
  const toggle = await write('PUT', '/api/messaging/platforms/telegram')

  expect(toggle.body).toEqual({ enabled: true })
  fireEvent.click(await screen.findByRole('button', { name: en.messaging.restartNow }))
  await write('POST', '/api/gateway/restart')
  expectOwnerOnly()
})

it('switches real-profile browsing on the viewed agent', async () => {
  mount('&panel=settings&tab=config:browser&page=network')
  fireEvent.click(await screen.findByRole('switch', { name: 'Use My Real Browser Profile' }))
  const save = await write('PUT', '/api/config')

  expect(JSON.stringify(save.body)).toContain('"use_real_profile":true')
  expectOwnerOnly()
})

it('keeps an agent on another machine behind Open agent', async () => {
  mount('', 'remote')
  expect(await screen.findByRole('button', { name: 'Open agent' })).toBeTruthy()
  expect(screen.queryByText('Owner instructions')).toBeNull()
  expect(api).not.toHaveBeenCalled()
  act(() => $apiRequestScope.setKey('connectionId', 'remote'))
  expect(await screen.findByText('Owner instructions')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
})
