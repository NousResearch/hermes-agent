import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pluginServersQueryKey } from '@/app/capabilities/connectors/data/keys'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { $pluginRecords, publishPlugin } from '@/contrib/plugins-store'
import { type HermesGateway, type ProfileScope, setSkillEnabled } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'

import { CapabilityMenu } from './capability-menu'
import { ContextMenu } from './context-menu'

const mocks = vi.hoisted(() => ({
  skills: [
    { name: 'review-code', description: 'Review code', enabled: true },
    { name: 'inactive', description: 'Disabled', enabled: false }
  ],
  hostedToggle: vi.fn(async () => ({ ok: true })),
  localToggle: vi.fn(async () => undefined),
  gateway: vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<{ ok?: boolean; plugins?: unknown[] }> => ({ plugins: [] })),
  agentRequest: vi.fn(async (_connection: string, _profile: string, _method: string, _params?: Record<string, unknown>): Promise<{ ok?: boolean; plugins?: unknown[] }> => ({ plugins: [] })),
  mcpHook: vi.fn((_options: { profile?: ProfileScope; sessionId?: string | null }) => ({
    names: ['filesystem'],
    servers: { filesystem: { enabled: true } },
    setServerEnabled: mocks.localToggle,
    profilePending: false,
    configFailed: false,
    configLoading: false
  }))
}))

vi.mock('@/app/capabilities/skills/skills-data', async importOriginal => ({
  ...(await importOriginal()),
  useSkillsQuery: () => ({ data: mocks.skills, error: null, isPending: false })
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  setSkillEnabled: vi.fn(async () => undefined)
}))
vi.mock('@/app/capabilities/connectors/data/queries', () => ({
  useHostedConnectors: () => ({
    rows: [{ connected: true, enabled: true, slug: 'gmail', accountLabel: 'Work' }],
    titles: { gmail: 'Gmail' },
    phase: 'ready',
    rulesFailed: false
  }),
  usePluginServers: () => []
}))
vi.mock('@/app/capabilities/connectors/data/mutations', () => ({
  useConnectorSwitch: () => ({ pending: null, setEnabled: mocks.hostedToggle })
}))
vi.mock('@/app/capabilities/mcp/use-mcp-servers', () => ({
  useMcpServers: (options: { profile?: ProfileScope; sessionId?: string | null }) => mocks.mcpHook(options)
}))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ gateway: null, requestGateway: mocks.gateway })
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal()),
  requestGatewayForAgent: mocks.agentRequest
}))

function open(
  kind: 'skills' | 'plugins' | 'connectors',
  profile: ProfileScope = 'byte',
  gateway?: HermesGateway,
  sessionId?: string | null
) {
  const onInsertSkill = vi.fn()
  const onClose = vi.fn()

  render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <DropdownMenu defaultOpen>
          <DropdownMenuTrigger>Open</DropdownMenuTrigger>
          <DropdownMenuContent>
            <CapabilityMenu
              gateway={gateway}
              kind={kind}
              onClose={onClose}
              onInsertSkill={onInsertSkill}
              profile={profile}
              sessionId={sessionId}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </I18nProvider>
    </QueryClientProvider>
  )

  return { onClose, onInsertSkill }
}

beforeEach(() => {
  mocks.gateway.mockReset().mockResolvedValue({ plugins: [] })
  mocks.agentRequest.mockReset().mockResolvedValue({ plugins: [] })
  mocks.hostedToggle.mockClear()
  mocks.localToggle.mockClear()
  mocks.mcpHook.mockClear()
  vi.mocked(setSkillEnabled).mockClear()
  $pluginRecords.set({})
})
afterEach(() => {
  cleanup()
  queryClient.clear()
})

describe('composer capability lists', () => {
  it('keeps the + menu open when toggling an installed skill', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <I18nProvider configClient={null} initialLocale="en">
          <ContextMenu
            onInsertSkill={vi.fn()}
            onInsertText={vi.fn()}
            onOpenUrlDialog={vi.fn()}
            profile="byte"
            state={{
              model: { model: '', provider: '', canSwitch: false },
              tools: { enabled: true, label: 'Add to chat' },
              voice: { enabled: false, active: false }
            }}
          />
        </I18nProvider>
      </QueryClientProvider>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Add to chat' }), { button: 0, ctrlKey: false })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Skills' }))
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'review-code: On' }))

    await waitFor(() => expect(setSkillEnabled).toHaveBeenCalledWith('review-code', false, 'byte'))
    expect(screen.getByRole('searchbox', { name: 'Search Skills' })).toBeTruthy()
  })

  it('lets Escape leave a focused capability search', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <I18nProvider configClient={null} initialLocale="en">
          <ContextMenu
            onInsertSkill={vi.fn()}
            onInsertText={vi.fn()}
            onOpenUrlDialog={vi.fn()}
            state={{ model: { model: '', provider: '', canSwitch: false }, tools: { enabled: true, label: 'Add to chat' }, voice: { enabled: false, active: false } }}
          />
        </I18nProvider>
      </QueryClientProvider>
    )
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Add to chat' }), { button: 0, ctrlKey: false })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Skills' }))
    const search = screen.getByRole('searchbox', { name: 'Search Skills' })
    search.focus()
    fireEvent.keyDown(search, { key: 'Escape' })
    expect(screen.queryByRole('searchbox', { name: 'Search Skills' })).toBeNull()
  })

  it('exposes toggles as keyboard-navigable checked menu items', () => {
    open('skills')
    expect(screen.getByRole('menuitemcheckbox', { name: /review-code/i })).toBeTruthy()
  })

  it('shows installed skills, inserts an enabled skill into the chat, and toggles it for this profile', async () => {
    const { onClose, onInsertSkill } = open('skills')

    expect(screen.getByRole('menuitem', { name: /review-code/ })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: /inactive/ })).toBeNull()
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'review-code: On' }))
    await waitFor(() => expect(setSkillEnabled).toHaveBeenCalledWith('review-code', false, 'byte'))
    fireEvent.click(screen.getByRole('menuitem', { name: /review-code/ }))
    expect(onInsertSkill).toHaveBeenCalledWith('review-code')
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('reloads the menu chat, not whichever chat is active, after an MCP toggle', () => {
    open('connectors', { connectionId: 'remote-b', profile: 'byte' }, undefined, 'tile-session')

    expect(mocks.mcpHook).toHaveBeenCalledWith(
      expect.objectContaining({
        profile: { connectionId: 'remote-b', profile: 'byte' },
        sessionId: 'tile-session'
      })
    )
  })

  it('shows local and connected hosted connectors and toggles their access in place', async () => {
    open('connectors')

    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Gmail: On' }))
    await waitFor(() => expect(mocks.hostedToggle).toHaveBeenCalledWith('gmail', false))
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'filesystem: On' }))
    await waitFor(() => expect(mocks.localToggle).toHaveBeenCalledWith('filesystem', false))
  })

  it('does not show a previous gateway’s plugins in another connection with the same profile', async () => {
    const first = { request: vi.fn(async () => ({ plugins: [{ key: 'first', name: 'First gateway plugin', status: 'enabled', source: 'user' }] })) } as unknown as HermesGateway
    const second = { request: vi.fn(() => new Promise(() => undefined)) } as unknown as HermesGateway

    open('plugins', 'byte', first)
    await screen.findByText('First gateway plugin')
    cleanup()
    open('plugins', 'byte', second)

    expect(screen.queryByText('First gateway plugin')).toBeNull()
  })

  it('routes plugin reads and writes to the chat owner rather than the ambient gateway', async () => {
    const ambient = { request: vi.fn(async () => ({ plugins: [{ key: 'ambient', name: 'Ambient', status: 'enabled', source: 'user' }] })) } as unknown as HermesGateway
    mocks.agentRequest.mockImplementation(async (_connection, _profile, _method, params) =>
      params?.action === 'toggle'
        ? { ok: true }
        : { plugins: [{ key: 'tile', name: 'Tile plugin', status: 'enabled', source: 'user' }] }
    )

    open('plugins', { connectionId: 'remote-b', profile: 'byte' }, ambient)
    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Tile plugin: On' }))

    await waitFor(() => expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-b', 'byte', 'plugins.manage', {
      action: 'toggle', key: 'tile', enable: false, profile: 'byte'
    }))
    expect(ambient.request).not.toHaveBeenCalled()
  })

  it('refreshes the connector plugin-server list after changing an agent plugin', async () => {
    mocks.gateway.mockImplementation(async (_method, params) => params?.action === 'toggle' ? { ok: true } : { plugins: [{ key: 'media/weather', name: 'Weather', status: 'enabled', source: 'user' }] })
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    open('plugins')

    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Weather: On' }))
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: pluginServersQueryKey('byte') }))
    invalidate.mockRestore()
  })

  it('lists installed agent and desktop plugins, and toggles an agent by its canonical key', async () => {
    mocks.gateway.mockResolvedValue({
      plugins: [{ key: 'media/weather', name: 'Weather', status: 'enabled', source: 'user', description: 'Forecasts' }]
    })
    publishPlugin({ id: 'desktop-clock', name: 'Clock', kind: 'disk', status: 'loaded' }, { activate: vi.fn(), deactivate: vi.fn() })
    open('plugins')

    const toggle = await screen.findByRole('menuitemcheckbox', { name: 'Weather: On' })
    fireEvent.click(toggle)
    await waitFor(() =>
      expect(mocks.gateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'media/weather',
        enable: false,
        profile: 'byte'
      })
    )
    const desktopToggle = screen.getByRole('menuitemcheckbox', { name: 'Clock: On' })
    await waitFor(() => expect(desktopToggle.hasAttribute('disabled')).toBe(false))
    fireEvent.click(desktopToggle)
    await waitFor(() => expect($pluginRecords.get()['desktop-clock']?.status).toBe('disabled'))
  })
})
