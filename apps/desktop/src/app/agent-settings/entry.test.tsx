import { host } from '@hermes/plugin-sdk'
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { pluginSettingsRouteHref } from '@/contrib/settings-pages'
import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import { botsText } from '@/plugins/hermes-bots/i18n'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import type * as GatewayStore from '@/store/gateway'
import { requestGatewayForAgent } from '@/store/gateway'
import { $notifications } from '@/store/notifications'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import { $shellChoice } from '@/store/shell-variant'

import { routinesCopy } from '../routines/routines-copy'
import { $routineSelection, $routineSheet, closeRoutineSheet, selectRoutine } from '../routines/routines-store'

import { SettingsView } from './entry'
import { $lastViewedAgent } from './landing'

const routines = routinesCopy('en')

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

  // The agent page counts connectors from the same list the Connectors page reads.
  if (method === 'connectors.list') {
    return { connectors: [] }
  }

  return {}
})

const screenPlugin = vi.hoisted(() => ({ tone: 'unknown' as string, open: vi.fn(), start: vi.fn() }))

vi.mock('@/plugins/hermes-bots/screen-open', async original => ({
  ...(await original<object>()),
  openBotScreen: screenPlugin.open,
  startBotScreen: screenPlugin.start
}))

vi.mock('@/app/chat/sidebar/simplified-bot-chat-row', async original => ({
  ...(await original<object>()),
  useBotModeLoaded: () => true
}))
vi.mock('@/plugins/hermes-bots/i18n', async original => (await import('@/test/bots-screen-mocks')).botsI18nMock(original))
vi.mock('@/plugins/hermes-bots/screen-portal', async original =>
  (await import('@/test/bots-screen-mocks')).screenPortalMock(original, screenPlugin)
)

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
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  await import('./index')
}, 60_000)

beforeEach(() => {
  screenPlugin.tone = 'unknown'
  screenPlugin.open.mockReset()
  screenPlugin.start.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
  closed.mockReset()
  $notifications.set([])
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
  closeRoutineSheet()
  selectRoutine(null)
})

function LocationProbe() {
  const location = useLocation()

  return <span data-testid="location">{`${location.pathname}${location.search}`}</span>
}

const closed = vi.fn()

function mount(path: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <SettingsView onClose={closed} />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it.each(['', '&connection=beta'])(
  'pins every real overview request to one native owner (%s)',
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
    act(() => $apiRequestScope.setKey('connectionId', 'alpha'))
    expect(await screen.findByRole('button', { name: 'Open agent' })).toBeTruthy()
    expect(screen.queryByText('Owner instructions')).toBeNull()
  }
)

it('sends the classic shell from an agent-only route to the Profiles page', async () => {
  $shellChoice.set(false)
  mount('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:instructions')
  // Classic has no agent page: it must not land on Model settings with a stale agentProfile.
  await waitFor(() => expect(screen.getByTestId('location').textContent).toMatch(/^\/profiles(\?|$)/))
  expect(screen.getByTestId('location').textContent).not.toContain('agentProfile')
  expect(screen.queryByRole('textbox', { name: 'Instructions editor' })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
  expect(api.mock.calls.some(([request]) => request.path.endsWith('/soul'))).toBe(false)
})

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

/** Leave Instructions for General and come back: the editor unmounts in between. */
async function revisitInstructions() {
  fireEvent.click(screen.getByRole('button', { name: 'General' }))
  await screen.findByRole('heading', { name: 'General' })
  fireEvent.click(screen.getByRole('button', { name: 'Instructions' }))
}

it('keeps unsaved instructions on their owner across leaving the page and coming back', async () => {
  mount('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:instructions')
  const editor = await screen.findByRole('textbox', { name: 'Instructions editor' })
  fireEvent.change(editor, { target: { value: 'Unsaved owner instructions' } })
  await revisitInstructions()
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

it('retains an in-flight instructions save across leaving and clears only its submitted revision', async () => {
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

  await revisitInstructions()
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
  await revisitInstructions()
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

it('keeps one settings menu, in the page panel, and Back to agents returns the panel to the agents', async () => {
  let search = ''

  function LocationProbe() {
    search = useLocation().search

    return null
  }

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter
        initialEntries={['/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:general']}
      >
        <SettingsView onClose={() => {}} />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )

  const panel = await screen.findByRole('complementary', { name: en.commandCenter.settings })
  // 1036:7189: the stage draws no second menu beside the page.
  expect(screen.getAllByRole('navigation', { name: en.commandCenter.settings })).toHaveLength(1)
  fireEvent.click(within(panel).getByRole('button', { name: 'Model' }))
  await waitFor(() => expect(new URLSearchParams(search).get('tab')).toBe('config:model'))

  fireEvent.click(within(panel).getByRole('button', { name: /Back to agents/ }))
  await waitFor(() => expect(new URLSearchParams(search).get('panel')).toBeNull())
  expect(screen.queryByRole('complementary', { name: en.commandCenter.settings })).toBeNull()
  expect(screen.queryByRole('navigation', { name: en.commandCenter.settings })).toBeNull()
})

function profileRow(name: string, extra: Record<string, unknown> = {}) {
  return {
    name,
    is_default: name === 'default',
    path: `/test/${name}`,
    has_env: false,
    model: null,
    provider: null,
    skill_count: 0,
    ...extra
  }
}

/** A backend whose profile list follows its renames and deletes. */
function profilesBackend(initial: ReturnType<typeof profileRow>[]) {
  let rows = initial

  api.mockImplementation(async (request: Request) => {
    const [pathname] = request.path.split('?')
    const target = decodeURIComponent(pathname.replace('/api/profiles/', ''))

    if (request.method === 'DELETE') {
      rows = rows.filter(row => row.name !== target)

      return { ok: true, path: `/test/${target}` }
    }

    if (request.method === 'PATCH') {
      const { new_name } = request.body as { new_name: string }
      rows = rows.map(row => (row.name === target ? profileRow(new_name) : row))

      return { name: new_name, ok: true, path: `/test/${new_name}` }
    }

    return pathname === '/api/profiles' ? { current: 'default', profiles: rows } : defaultApi(request)
  })
}

function mountWithRoute(path: string) {
  let route = ''

  function RouteProbe() {
    const location = useLocation()
    route = location.pathname + location.search

    return null
  }

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <SettingsView onClose={() => {}} />
        <RouteProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )

  return () => route
}

it('leaves a deleted agent from its General settings and drops it from every list', async () => {
  profilesBackend([profileRow('default'), profileRow('research')])
  const route = mountWithRoute('/settings?agentProfile=research&connection=beta&panel=settings&tab=agent:general')
  fireEvent.click(await screen.findByRole('button', { name: 'Delete agent' }))
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: en.common.delete }))

  // Not a "Deleted" notice on the gone agent's page, and the Agents destination must not land back on it.
  await waitFor(() => expect(route()).toBe('/profiles'))
  expect($lastViewedAgent.get()).toBeNull()
  await waitFor(() => expect($profiles.get().map(row => row.name)).toEqual(['default']))
})

it('lists a renamed agent once, under its new name, as soon as the rename lands', async () => {
  profilesBackend([profileRow('default'), profileRow('research')])
  $profiles.set([profileRow('default'), profileRow('research')])
  const route = mountWithRoute('/settings?agentProfile=research&connection=beta&panel=settings&tab=agent:general')
  fireEvent.click(await screen.findByRole('button', { name: /^Rename: / }))
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'scholar' } })
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: en.profiles.rename }))

  await waitFor(() => expect(new URLSearchParams(route().split('?')[1]).get('agentProfile')).toBe('scholar'))
  await waitFor(() => expect($profiles.get().map(row => row.name)).toEqual(['default', 'scholar']))
})

it('renames the default agent from the name it has, not the bot title that masks it', async () => {
  profilesBackend([profileRow('default', { display_name: 'Home', bot_title: 'Concierge' })])
  mountWithRoute('/settings?agentProfile=default&connection=beta&panel=settings&tab=agent:general')
  fireEvent.click(await screen.findByRole('button', { name: /^Rename: / }))

  expect((screen.getByLabelText(en.profiles.displayNameLabel) as HTMLInputElement).value).toBe('Home')
})

it('opens a link naming this machine as "local" in place, without the Open agent gate', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  mount('/settings?agentProfile=default&connection=local')

  expect(await screen.findByText('Owner instructions')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
})

it('opens the Routines page from the Overview’s Routines row, as its neighbours open theirs', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  mount('/settings?agentProfile=default')

  fireEvent.click((await screen.findAllByRole('button', { name: new RegExp(`^${en.cron.title}`) }))[0])

  expect(screen.getByTestId('location').textContent).toBe('/cron')
})

const screenControls = () => screen.findAllByRole('button', { name: new RegExp(`^${botsText().screen.portalTitle}`) })

it.each([
  ['missing', botsText().screen.portalNotInstalled],
  ['off', botsText().screen.portalStopped]
] as const)('says on the Overview where a Screen that is not running stands (%s), and opens it', async (tone, says) => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = tone
  mount('/settings?agentProfile=default')

  const [row, ...others] = await screenControls()
  expect(others).toHaveLength(0)
  expect(row.textContent).toContain(says)
  expect(row.querySelector('img')).toBeNull()

  // The Screen is a tab beside the chat: the page steps aside once it has opened, and once only
  // however many clicks were waiting on it.
  fireEvent.click(row)
  fireEvent.click(row)
  await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
  await act(() => new Promise(resolve => setTimeout(resolve, 20)))
  expect(closed).toHaveBeenCalledTimes(1)
  expect(screenPlugin.open.mock.calls[0][0]).toMatchObject({ name: 'default' })
})

it('shows a running Screen as itself on the Overview, in the row’s place, and it opens the same Screen', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'live'
  mount('/settings?agentProfile=default')

  // One control, the tile: its caption says who holds the screen, so no row repeats it.
  const [tile, ...others] = await screenControls()
  expect(others).toHaveLength(0)
  expect(tile.getAttribute('aria-label')).toBe(`${botsText().screen.portalTitle}: ${botsText().screen.portalWatching}`)

  // The row around the tile is not a control: a click beside the tile does nothing.
  fireEvent.click(tile.closest('.agent-overview-row')!)
  expect(screenPlugin.open).not.toHaveBeenCalled()
  expect(closed).not.toHaveBeenCalled()

  fireEvent.click(tile)
  expect(screenPlugin.open).toHaveBeenCalledTimes(1)
  expect(screenPlugin.open.mock.calls[0][0]).toMatchObject({ name: 'default' })
  expect(closed).toHaveBeenCalledTimes(1)
})

const startButton = () => screen.queryByRole('button', { name: botsText().screen.heroStart })

it('starts a stopped Screen from its row, once, for that agent', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'

  let started: () => void = () => {}
  screenPlugin.start.mockImplementation(() => new Promise<void>(resolve => (started = resolve)))
  mount('/settings?agentProfile=default')
  await screenControls()

  fireEvent.click(startButton()!)
  fireEvent.click(startButton()!)

  expect(screenPlugin.start).toHaveBeenCalledTimes(1)
  expect(screenPlugin.start.mock.calls[0][0]).toMatchObject({ name: 'default' })
  expect(startButton()?.hasAttribute('disabled')).toBe(true)
  // Starting is not opening: the page stays, and the row is still there to open the Screen.
  expect(closed).not.toHaveBeenCalled()

  await act(async () => started())
  expect(startButton()?.hasAttribute('disabled')).toBe(false)
})

it('gives Start back, and says so, when the backend never answers', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'
  screenPlugin.start.mockImplementation(() => new Promise<void>(() => {}))
  mount('/settings?agentProfile=default')
  await screenControls()
  vi.useFakeTimers()

  try {
    fireEvent.click(startButton()!)
    expect(startButton()?.hasAttribute('disabled')).toBe(true)

    await act(() => vi.advanceTimersByTimeAsync(5 * 60_000))

    expect(startButton()?.hasAttribute('disabled')).toBe(false)
    expect($notifications.get().some(note => note.kind === 'error')).toBe(true)
  } finally {
    vi.useRealTimers()
  }
})

it('asks for one Screen however often Start is pressed, and tells nobody once the page is gone', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'
  screenPlugin.start.mockClear()
  screenPlugin.start.mockImplementation(() => new Promise<void>(() => {}))
  const view = mount('/settings?agentProfile=default')
  await screenControls()
  vi.useFakeTimers()

  try {
    fireEvent.click(startButton()!)
    await act(() => vi.advanceTimersByTimeAsync(5 * 60_000))
    const told = $notifications.get().length
    fireEvent.click(startButton()!)

    expect(screenPlugin.start).toHaveBeenCalledTimes(1)

    view.unmount()
    await act(() => vi.advanceTimersByTimeAsync(5 * 60_000))

    expect($notifications.get().length).toBe(told)
  } finally {
    vi.useRealTimers()
  }
})

it('starts the agent on the page: another agent\'s request still out is not this one\'s', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'
  screenPlugin.start.mockClear()
  screenPlugin.start.mockImplementation(() => new Promise<void>(() => {}))

  function Switch() {
    const navigate = useNavigate()

    return <button onClick={() => navigate('/settings?agentProfile=ops')}>to ops</button>
  }

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/settings?agentProfile=default']}>
        <SettingsView onClose={closed} />
        <Switch />
      </MemoryRouter>
    </QueryClientProvider>
  )
  await screenControls()
  fireEvent.click(startButton()!)
  expect(screenPlugin.start).toHaveBeenCalledTimes(1)

  fireEvent.click(screen.getByRole('button', { name: 'to ops' }))
  await waitFor(() => expect(startButton()?.hasAttribute('disabled')).toBe(false))
  fireEvent.click(startButton()!)

  expect(screenPlugin.start).toHaveBeenCalledTimes(2)
})

it('says why a Screen did not start, and leaves its row as it was', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'
  screenPlugin.start.mockRejectedValue(new Error('not enough memory to start a screen'))
  mount('/settings?agentProfile=default')
  await screenControls()

  fireEvent.click(startButton()!)

  await waitFor(() =>
    expect($notifications.get().some(note => note.kind === 'error' && /not enough memory/.test(JSON.stringify(note)))).toBe(true)
  )
  expect(startButton()?.hasAttribute('disabled')).toBe(false)
  expect((await screenControls())[0].textContent).toContain(botsText().screen.portalStopped)
})

it.each(['missing', 'unknown', 'live', 'unsupported'])('offers no Start for a Screen that is %s', async tone => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = tone
  mount('/settings?agentProfile=default')

  await screen.findAllByRole('button', { name: new RegExp(`^${en.cron.title}`) })
  expect(startButton()).toBeNull()
})

it('stays on the Overview and says so when the Screen could not be opened', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'off'
  vi.spyOn(host, 'profileRoutes').mockRejectedValue(new Error('no route to that agent'))
  mount('/settings?agentProfile=default')

  fireEvent.click(await screen.findByRole('button', { name: new RegExp(`^${botsText().screen.portalTitle}`) }))

  await waitFor(() => expect($notifications.get().some(note => note.kind === 'error')).toBe(true))
  expect(closed).not.toHaveBeenCalled()
})

it('says the Screen is being checked until its state is known', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'unknown'
  mount('/settings?agentProfile=default')

  const row = await screen.findByRole('button', { name: new RegExp(`^${botsText().screen.portalTitle}`) })
  expect(row.textContent).toContain(botsText().screen.heroConnecting)
})

it('leaves Screen off the Overview where an agent cannot have one', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  screenPlugin.tone = 'unsupported'
  mount('/settings?agentProfile=default')

  await screen.findAllByRole('button', { name: new RegExp(`^${en.cron.title}`) })
  expect(screen.queryByRole('button', { name: new RegExp(`^${botsText().screen.portalTitle}`) })).toBeNull()
})

it('says a routine’s schedule in words on the Overview, as the Routines page does', async () => {
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  api.mockImplementation(async (request: Request) =>
    request.path.split('?')[0] === '/api/cron/jobs'
      ? [{ enabled: true, id: 'review', name: 'Review', profile: 'default', schedule: { expr: '0 9 * * *', kind: 'cron' } }]
      : defaultApi(request)
  )
  mount('/settings?agentProfile=default')

  const row = await screen.findByRole('button', { name: /^Review/ })
  expect(row.textContent).not.toContain('0 9 * * *')
  expect(row.textContent).toMatch(/9:00/)
})

it('lists the agent’s routines read only in Settings › Routines, each opening on the Routines page', async () => {
  // The window's own machine, named by nobody: the links stay on this window's Routines page.
  $apiRequestScope.set({ connectionId: null, profile: 'default' })

  const review = {
    enabled: true,
    id: 'review',
    name: 'Review',
    profile: 'default',
    prompt: 'Review notes',
    schedule: { expr: '0 9 * * *', kind: 'cron' },
    state: 'scheduled'
  }

  api.mockImplementation(async (request: Request) =>
    request.path.split('?')[0] === '/api/cron/jobs' ? [review] : defaultApi(request)
  )
  mount('/settings?agentProfile=default&panel=settings&tab=agent:schedules')

  // No editor here: no instruction to type, no Save.
  fireEvent.click(await screen.findByRole('button', { name: /^Review/ }))
  expect(screen.queryByLabelText(routines.instruction)).toBeNull()
  expect($routineSelection.get()).toEqual({ id: 'review', profile: 'default' })
  expect(screen.getByTestId('location').textContent).toBe('/cron')

  cleanup()
  mount('/settings?agentProfile=default&panel=settings&tab=agent:schedules')
  fireEvent.click(await screen.findByRole('button', { name: en.cron.newCron }))
  expect($routineSheet.get()).toEqual({ mode: 'create', profile: 'default' })
})
