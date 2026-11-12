import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { pluginSettingsRouteHref } from '@/contrib/settings-pages'
import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import type * as GatewayStore from '@/store/gateway'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import { $shellChoice } from '@/store/shell-variant'

import { SettingsView } from './entry'

const plugin = {
  description: '',
  key: 'image_gen/fal',
  name: 'Fal',
  source: 'user',
  status: 'enabled',
  version: '1',
  settings_schema: [{ description: '', key: 'region', label: 'Region', required: false, type: 'string', value: 'us' }]
}

const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
  if (method === 'plugins.manage') {
    return params?.action === 'list' ? { plugins: [plugin] } : { ok: true, plugin }
  }

  return {}
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: vi.fn(
    async (_connectionId: string, _profile: string, method: string, params?: Record<string, unknown>) =>
      requestGateway(method, params)
  )
}))

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    disabled,
    initialValue,
    onChange
  }: {
    disabled?: boolean
    initialValue: string
    onChange: (value: string) => void
  }) => (
    <textarea
      aria-label="Instructions editor"
      defaultValue={initialValue}
      disabled={disabled}
      onChange={event => onChange(event.target.value)}
    />
  )
}))

interface Request {
  path: string
  connectionId?: string
  profile?: string
  method?: string
  body?: unknown
}

const api = vi.fn(async ({ path, method }: Request): Promise<unknown> => {
  const pathname = path.split('?')[0]

  if (pathname === '/api/skills' || pathname === '/api/tools/toolsets') {
    return []
  }

  if (pathname === '/api/cron/jobs') {
    return method === 'POST' ? { id: 'saved', enabled: true } : []
  }

  if (pathname === '/api/messaging/platforms') {
    return { platforms: [] }
  }

  if (pathname === '/api/profiles/sessions') {
    return { sessions: [], total: 0, limit: 5, offset: 0 }
  }

  if (pathname.endsWith('/soul')) {
    return { content: 'Owner instructions' }
  }

  if (pathname === '/api/profiles') {
    return {
      current: 'default',
      profiles: [
        {
          name: 'default',
          is_default: true,
          path: '/test/hermes',
          has_env: false,
          model: null,
          provider: null,
          skill_count: 0
        }
      ]
    }
  }

  return {}
})

const defaultApi = api.getMockImplementation()!

const originalDesktop = window.hermesDesktop

// Exercise the real lazy-route boundary after mocks are initialized, before the
// interaction clock starts; a static import would bypass that loading boundary.
beforeAll(async () => {
  await import('./index')
}, 60_000)

beforeEach(() => {
  queryClient.clear()
  api.mockClear()
  api.mockImplementation(defaultApi)
  requestGateway.mockClear()
  vi.mocked(requestGatewayForAgent).mockClear()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  $profiles.set([])
  $activeGatewayProfile.set('default')
  $apiRequestScope.set({ connectionId: 'beta', profile: 'default' })
  $agentPlugins.set([])
  $agentPluginsStatus.set('idle')
  $shellChoice.set(true)
})
afterEach(() => {
  cleanup()
  queryClient.clear()
  window.hermesDesktop = originalDesktop
  $apiRequestScope.set({ profile: null, connectionId: null })
  $settingsScopeOverride.set(null)
  $shellChoice.set(false)
})

function mount(path: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <SettingsView onClose={() => {}} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it.each(['', '&connection=beta'])(
  'pins every real overview request and routine write to one native owner (%s)',
  async connection => {
    mount(`/settings?agentProfile=default${connection}&tab=plugins&agent=image_gen%2Ffal`)
    await screen.findByText('Owner instructions')
    const paths = ['/api/config', '/api/skills', '/api/tools/toolsets', '/api/cron/jobs?profile=default']

    await waitFor(() => {
      for (const path of paths) {
        const request = api.mock.calls.find(([request]) => request.path === path)?.[0]

        expect(request).toBeDefined()
        // An absent connection would dispatch through the native primary instead.
        expect(request?.connectionId).toBe('beta')
        expect(request?.profile).toBe('default')
      }
    })
    fireEvent.click(screen.getByRole('button', { name: en.cron.newCron }))
    fireEvent.change(screen.getByPlaceholderText('0 9 * * *'), { target: { value: '0 9 * * *' } })
    fireEvent.change(screen.getByLabelText(en.cron.promptLabel), { target: { value: 'Review notes' } })
    fireEvent.click(screen.getByRole('button', { name: en.common.save }))
    await waitFor(() => expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(true))
    const write = api.mock.calls.find(([request]) => request.method === 'POST')![0]

    expect(write.connectionId).toBe('beta')
    expect(write.profile).toBe('default')
    act(() => $shellChoice.set(false))
    expect(await screen.findByLabelText(/Region/)).toBeTruthy()
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'beta',
      'default',
      'plugins.manage',
      { action: 'list', profile: 'default' },
      undefined
    )
    expect(screen.queryByText('Owner instructions')).toBeNull()
    act(() => $shellChoice.set(true))
    expect(await screen.findByText('Owner instructions')).toBeTruthy()
    act(() => $apiRequestScope.setKey('connectionId', 'alpha'))
    expect(await screen.findByRole('button', { name: 'Open agent' })).toBeTruthy()
    expect(screen.queryByText('Owner instructions')).toBeNull()
    act(() => $shellChoice.set(false))
    expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
    act(() => $shellChoice.set(true))
    expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
  }
)

it.each([false, true])('opens and saves the existing plugin route with redesign=%s', async redesigned => {
  $shellChoice.set(redesigned)
  mount(pluginSettingsRouteHref({ agent: plugin.key }, 'default'))
  const region = await screen.findByLabelText(/Region/)

  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
  fireEvent.change(region, { target: { value: 'eu' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))
  await waitFor(() =>
    expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
      action: 'settings',
      key: 'image_gen/fal',
      profile: 'default',
      values: { region: 'eu' }
    })
  )
})

it('starts a newly selected settings section at the top without moving its navigation', async () => {
  mount('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:instructions')
  await screen.findByRole('textbox', { name: 'Instructions editor' })
  const body = screen.getByRole('region', { name: en.commandCenter.settings })
  const navigation = screen.getByRole('navigation', { name: en.commandCenter.settings })

  body.scrollTop = 164
  navigation.scrollTop = 120
  fireEvent.click(screen.getByRole('button', { name: 'General' }))
  await screen.findByRole('heading', { name: 'General' })
  expect(body.scrollTop).toBe(0)
  expect(navigation.scrollTop).toBe(120)
})

it('expands tool configuration in place without scrolling away from its disclosure', async () => {
  let search = ''

  function LocationProbe() {
    search = useLocation().search

    return null
  }

  const location = () => ({ search })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:tools']}>
        <SettingsView onClose={() => {}} />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )

  const advanced = (await screen.findByText(en.settings.sections.advanced, { selector: 'summary *' })).closest(
    'details'
  )!

  const body = advanced.closest('section')!

  body.scrollTop = 240
  advanced.open = true
  fireEvent(advanced, new Event('toggle'))
  await waitFor(() => expect(location().search).toContain('tools-advanced'))
  expect(advanced.open).toBe(true)
  expect(body.scrollTop).toBe(240)
  advanced.open = false
  fireEvent(advanced, new Event('toggle'))
  await waitFor(() => expect(location().search).not.toContain('tools-advanced'))
  expect(advanced.open).toBe(false)
  expect(body.scrollTop).toBe(240)
})

it('keeps unsaved instructions on their owner across redesign and classic switches', async () => {
  mount('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:instructions')
  const editor = await screen.findByRole('textbox', { name: 'Instructions editor' })
  fireEvent.change(editor, { target: { value: 'Unsaved owner instructions' } })
  act(() => $shellChoice.set(false))
  expect(((await screen.findByRole('textbox', { name: 'Instructions editor' })) as HTMLTextAreaElement).value).toBe(
    'Unsaved owner instructions'
  )
  act(() => $shellChoice.set(true))
  expect(((await screen.findByRole('textbox', { name: 'Instructions editor' })) as HTMLTextAreaElement).value).toBe(
    'Unsaved owner instructions'
  )
  fireEvent.click(screen.getByRole('button', { name: en.profiles.saveSoul }))
  await waitFor(() =>
    expect(api.mock.calls.some(([request]) => request.method === 'PUT' && request.path.endsWith('/soul'))).toBe(true)
  )
  const write = api.mock.calls.find(([request]) => request.method === 'PUT' && request.path.endsWith('/soul'))![0]
  expect(write.connectionId).toBe('beta')
  expect(write.profile).toBe('default')
  expect(write.body).toEqual({ content: 'Unsaved owner instructions' })
})

it('retains an in-flight instructions save across switching and clears only its submitted revision', async () => {
  const pending = Promise.withResolvers<unknown>()
  api.mockImplementation(request =>
    request.method === 'PUT' && request.path.endsWith('/soul') ? pending.promise : defaultApi(request)
  )
  mount('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:instructions')
  fireEvent.change(await screen.findByRole('textbox', { name: 'Instructions editor' }), {
    target: { value: 'Submitted instructions' }
  })
  fireEvent.click(screen.getByRole('button', { name: en.profiles.saveSoul }))
  await waitFor(() => expect(api.mock.calls.filter(([request]) => request.method === 'PUT')).toHaveLength(1))

  act(() => $shellChoice.set(false))
  const editor = await screen.findByRole('textbox', { name: 'Instructions editor' })
  expect((editor as HTMLTextAreaElement).disabled).toBe(true)
  const save = screen.getByRole('button', { name: en.profiles.saveSoul })
  expect((save as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(save)
  expect(api.mock.calls.filter(([request]) => request.method === 'PUT')).toHaveLength(1)

  // A queued editor change can arrive after Save; its newer revision must survive.
  fireEvent.change(editor, { target: { value: 'Later instructions' } })
  await act(async () => {
    pending.resolve({ ok: true })
  })
  await waitFor(() => expect((save as HTMLButtonElement).disabled).toBe(false))
  act(() => $shellChoice.set(true))
  expect(((await screen.findByRole('textbox', { name: 'Instructions editor' })) as HTMLTextAreaElement).value).toBe(
    'Later instructions'
  )
  fireEvent.click(screen.getByRole('button', { name: en.profiles.saveSoul }))
  await waitFor(() => expect(api.mock.calls.filter(([request]) => request.method === 'PUT')).toHaveLength(2))
  expect(
    api.mock.calls
      .filter(([request]) => request.method === 'PUT')
      .map(([request]) => ({
        connectionId: request.connectionId,
        profile: request.profile,
        body: request.body
      }))
  ).toEqual([
    { connectionId: 'beta', profile: 'default', body: { content: 'Submitted instructions' } },
    { connectionId: 'beta', profile: 'default', body: { content: 'Later instructions' } }
  ])
})
