// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterAll, afterEach, describe, expect, it, vi } from 'vitest'

import { registerPluginLocales } from '@/i18n/plugin-i18n'
import { queryClient } from '@/lib/query-client'

// @ts-expect-error - the runtime plugin is plain ESM outside the TS project
import * as plugin from '../../../../desktop-plugins/pipedream-apps/plugin.js'

const disposeI18n = registerPluginLocales(plugin.ID, plugin.LOCALES)

function renderWithQuery(ui: ReactElement) {
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

function toastSpy() {
  return { success: vi.fn(), warning: vi.fn() }
}

function pipedreamFrame() {
  return window.document.querySelector('iframe[title="Pipedream Connect"]') as HTMLIFrameElement | null
}

function dispatchPipedreamMessage(
  frame: HTMLIFrameElement,
  data: unknown,
  options: { origin?: string; source?: unknown } = {}
) {
  const event = new MessageEvent('message', {
    data,
    origin: options.origin ?? 'https://pipedream.com'
  })

  Object.defineProperty(event, 'source', {
    configurable: true,
    value: 'source' in options ? options.source : frame.contentWindow
  })
  window.dispatchEvent(event)
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
})

afterAll(() => {
  disposeI18n()
})

describe('pipedream apps runtime plugin', () => {
  it('rewrites Connectors deep links to the Apps tab', () => {
    expect(plugin.appsTabHashFor('#/capabilities?tab=connectors')).toBe(
      '#/capabilities?tab=pipedream-apps:apps'
    )
    expect(plugin.appsTabHashFor('#/capabilities?tab=connectors&server=github')).toBe(
      '#/capabilities?tab=pipedream-apps:apps&server=github'
    )
    expect(plugin.appsTabHashFor('#/capabilities?tab=skills')).toBe('#/capabilities?tab=skills')
    expect(plugin.appsTabHashFor('')).toBe('')
  })

  it('maps app and account shapes', () => {
    expect(plugin.appSlugOf({ name_slug: 'slack' })).toBe('slack')
    expect(plugin.accountSlugOf({ app: { name_slug: 'slack' } })).toBe('slack')
    expect(plugin.appNameOf({ name: 'Slack' })).toBe('Slack')
    expect(plugin.accountSubtitle({ email: 'me@example.com', name: 'Me' })).toBe('me@example.com')
    expect(plugin.accountSubtitle({ name: 'Me' })).toBe('Me')
    expect(plugin.accountTone({ healthy: false })).toBe('warn')
    expect(plugin.accountTone({ dead: true })).toBe('bad')
  })

  it('builds namespace-relative REST paths', async () => {
    const calls: Array<[string, unknown]> = []

    const rest = vi.fn(async (path: string, options?: unknown) => {
      calls.push([path, options])

      return {}
    })

    const api = plugin.createPipedreamApi(rest)

    await api.apps({ q: 'slack', limit: 24, category: 'appcat_1', after: 'c1' })
    await api.connect({ externalUserId: 'u1', appSlug: 'slack' })
    await api.accounts('u1')
    await api.disconnect({ externalUserId: 'u1', accountId: 'apn_1' })
    await api.mcpEnable({ externalUserId: 'u1', appSlug: 'slack' })

    expect(calls[0][0]).toBe('/apps?q=slack&limit=24&category=appcat_1&after=c1')
    expect(calls[1]).toEqual(['/connect', { method: 'POST', body: { externalUserId: 'u1', appSlug: 'slack' } }])
    expect(calls[2][0]).toBe('/accounts?externalUserId=u1')
    expect(calls[3]).toEqual(['/accounts/apn_1?externalUserId=u1', { method: 'DELETE' }])
    expect(calls[4]).toEqual(['/mcp/enable', { method: 'POST', body: { externalUserId: 'u1', appSlug: 'slack' } }])
  })

  it('opens the Connect Link and enables MCP once the account appears', async () => {
    let connected = false

    const api = {
      config: vi.fn().mockResolvedValue({
        configured: true,
        environment: 'production',
        mcpBaseUrl: 'https://app.example/api/pipedream/mcp'
      }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack', description: 'Team chat', categories: ['Chat'] }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn(async () =>
        connected
          ? [{ id: 'apn_1', email: 'me@example.com', app: { name_slug: 'slack', name: 'Slack' } }]
          : []
      ),
      connect: vi.fn(async () => {
        connected = true

        return { connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1' }
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn().mockResolvedValue({ ok: true, action: 'added' })
    }

    const toast = toastSpy()
    const openExternal = vi.fn().mockResolvedValue(true)

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn().mockResolvedValue(true)}
        fallbackExternalUserId="user-1"
        openExternal={openExternal}
        toast={toast}
      />
    )

    expect(await screen.findByText('Slack')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() =>
      expect(openExternal).toHaveBeenCalledWith('https://pipedream.com/connect?token=ctok_1')
    )
    await waitFor(() =>
      expect(toast.success).toHaveBeenCalledWith(
        'Connector connected',
        'Connected successfully. Tools will be available in a new chat.'
      )
    )
    await waitFor(() =>
      expect(api.mcpEnable).toHaveBeenCalledWith({ externalUserId: 'user-1', appSlug: 'slack' })
    )
  })

  it('warns when MCP enable fails after the app connects', async () => {
    let connected = false

    const api = {
      config: vi.fn().mockResolvedValue({
        configured: true,
        environment: 'production',
        mcpBaseUrl: 'https://app.example/api/pipedream/mcp'
      }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack' }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn(async () =>
        connected
          ? [{ id: 'apn_1', app: { name_slug: 'slack', name: 'Slack' } }]
          : []
      ),
      connect: vi.fn(async () => {
        connected = true

        return { connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1' }
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn().mockRejectedValue(new Error('dashboard rejected the server'))
    }

    const toast = toastSpy()

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn().mockResolvedValue(true)}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))

    await waitFor(() =>
      expect(toast.warning).toHaveBeenCalledWith(
        'Connector connected, tools not registered',
        'The connector is connected, but its tools could not be registered: dashboard rejected the server'
      )
    )
  })

  it('uses the shared page column and filters through the category dropdown', async () => {
    Element.prototype.scrollIntoView = vi.fn()

    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [{ id: 'appcat_1', name: 'Chat' }] }),
      apps: vi.fn().mockResolvedValue({ apps: [], nextCursor: null, prevCursor: null, totalCount: 0 }),
      accounts: vi.fn().mockResolvedValue([]),
      connect: vi.fn(),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    const { container } = renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toastSpy()}
      />
    )

    expect(await screen.findByRole('combobox', { name: 'Category' })).toBeTruthy()
    expect(container.querySelector('.pd-apps')?.classList.contains('capabilities-page')).toBe(true)

    fireEvent.click(screen.getByRole('combobox', { name: 'Category' }))
    fireEvent.click(await screen.findByText('Chat'))

    await waitFor(() =>
      expect(api.apps).toHaveBeenLastCalledWith(expect.objectContaining({ category: 'appcat_1' }))
    )
  })

  it('shows the setup state when Pipedream credentials are absent', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({ configured: false, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn(),
      apps: vi.fn(),
      accounts: vi.fn(),
      connect: vi.fn(),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toastSpy()}
      />
    )

    expect(await screen.findByText('Pipedream is not configured')).toBeTruthy()
    expect(api.apps).not.toHaveBeenCalled()
  })

  it('lays out All as On this device then Available, keeping the app pager', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [
          { name_slug: 'slack', name: 'Slack', description: 'Team chat' },
          { name_slug: 'github', name: 'GitHub', description: 'Code' }
        ],
        nextCursor: 'c2',
        prevCursor: null,
        totalCount: 3
      }),
      accounts: vi.fn().mockResolvedValue([
        { id: 'apn_1', email: 'me@example.com', healthy: true, app: { name_slug: 'slack', name: 'Slack' } }
      ]),
      connect: vi.fn(),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toastSpy()}
      />
    )

    expect(await screen.findByRole('button', { name: 'All 3' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'On this device 1' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Available 2' })).toBeTruthy()

    expect(await screen.findByRole('heading', { name: /On this device\s*1/ })).toBeTruthy()
    expect(screen.getByRole('heading', { name: /Available\s*2/ })).toBeTruthy()

    // Connected Slack lives in the device section, so the available grid omits it.
    expect(screen.getAllByText('Slack')).toHaveLength(1)
    expect(screen.getByText('GitHub')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Next' })).toBeTruthy()
  })

  it('lists connected accounts and disconnects after confirmation', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({ apps: [], nextCursor: null, prevCursor: null, totalCount: 0 }),
      accounts: vi.fn().mockResolvedValue([
        {
          id: 'apn_1',
          email: 'me@example.com',
          healthy: true,
          dead: false,
          app: { name_slug: 'slack', name: 'Slack' }
        }
      ]),
      connect: vi.fn(),
      disconnect: vi.fn().mockResolvedValue({ ok: true }),
      mcpEnable: vi.fn()
    }

    const toast = toastSpy()

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'On this device 1' }))
    expect(await screen.findByText('me@example.com')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.click(within(dialog).getByRole('button', { name: 'Disconnect' }))

    await waitFor(() =>
      expect(api.disconnect).toHaveBeenCalledWith({ accountId: 'apn_1', externalUserId: 'user-1' })
    )
    expect(api.mcpEnable).not.toHaveBeenCalled()
    expect(toast.success).toHaveBeenCalled()
  })

  it('opens Pipedream in the page and only trusts a validated success message', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({
        configured: true,
        environment: 'production',
        mcpBaseUrl: 'https://app.example/api/pipedream/mcp'
      }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack' }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn().mockResolvedValue([]),
      connect: vi.fn().mockResolvedValue({
        token: 'ctok_1',
        connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1'
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn().mockResolvedValue({ ok: true })
    }

    const toast = toastSpy()
    const openExternal = vi.fn().mockResolvedValue(true)

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        browser
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={openExternal}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(pipedreamFrame()).toBeTruthy())

    const frame = pipedreamFrame() as HTMLIFrameElement

    expect(frame.src).toContain('/_static/connect.html?token=ctok_1&app=slack')
    expect(openExternal).not.toHaveBeenCalled()

    // Wrong origin, wrong source, and malformed success payloads are ignored.
    dispatchPipedreamMessage(frame, { type: 'success', authProvisionId: 'apn_1' }, { origin: 'https://evil.example' })
    dispatchPipedreamMessage(frame, { type: 'success', authProvisionId: 'apn_1' }, { source: window })
    dispatchPipedreamMessage(frame, { type: 'success' })
    dispatchPipedreamMessage(frame, null)

    expect(toast.success).not.toHaveBeenCalled()
    expect(pipedreamFrame()).toBeTruthy()

    dispatchPipedreamMessage(frame, { type: 'success', authProvisionId: 'apn_1' })

    await waitFor(() => expect(pipedreamFrame()).toBeNull())
    await waitFor(() => expect(toast.success).toHaveBeenCalled())
    await waitFor(() =>
      expect(api.mcpEnable).toHaveBeenCalledWith({ externalUserId: 'user-1', appSlug: 'slack' })
    )
  })

  it('waits for the final account check before reporting a close as cancelled', async () => {
    let connected = false

    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack' }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn(async () =>
        connected ? [{ id: 'apn_1', app: { name_slug: 'slack', name: 'Slack' } }] : []
      ),
      connect: vi.fn().mockResolvedValue({
        token: 'ctok_1',
        connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1'
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    const toast = toastSpy()

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        browser
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))
    await waitFor(() => expect(pipedreamFrame()).toBeTruthy())

    connected = true
    dispatchPipedreamMessage(pipedreamFrame() as HTMLIFrameElement, { type: 'close' })

    await waitFor(() => expect(toast.success).toHaveBeenCalled())
    expect(pipedreamFrame()).toBeNull()
    expect(screen.queryByText('The connection was not completed. You can try again.')).toBeNull()
  })

  it('reports a cancellation only when the close and final account check find nothing', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack' }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn().mockResolvedValue([]),
      connect: vi.fn().mockResolvedValue({
        token: 'ctok_1',
        connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1'
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    const toast = toastSpy()

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        browser
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))
    await waitFor(() => expect(pipedreamFrame()).toBeTruthy())

    dispatchPipedreamMessage(pipedreamFrame() as HTMLIFrameElement, { type: 'close' })

    await waitFor(
      () => expect(screen.getByText('The connection was not completed. You can try again.')).toBeTruthy(),
      { timeout: 4_000 }
    )
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('does not report success when Pipedream posts an error', async () => {
    const api = {
      config: vi.fn().mockResolvedValue({ configured: true, environment: 'production', mcpBaseUrl: null }),
      categories: vi.fn().mockResolvedValue({ categories: [] }),
      apps: vi.fn().mockResolvedValue({
        apps: [{ name_slug: 'slack', name: 'Slack' }],
        nextCursor: null,
        prevCursor: null,
        totalCount: 1
      }),
      accounts: vi.fn().mockResolvedValue([]),
      connect: vi.fn().mockResolvedValue({
        token: 'ctok_1',
        connectLinkUrl: 'https://pipedream.com/connect?token=ctok_1'
      }),
      disconnect: vi.fn(),
      mcpEnable: vi.fn()
    }

    const toast = toastSpy()

    renderWithQuery(
      <plugin.PipedreamAppsTab
        api={api}
        browser
        copyText={vi.fn()}
        fallbackExternalUserId="user-1"
        openExternal={vi.fn()}
        toast={toast}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))
    await waitFor(() => expect(pipedreamFrame()).toBeTruthy())

    dispatchPipedreamMessage(pipedreamFrame() as HTMLIFrameElement, { type: 'error', message: 'access denied' })

    await waitFor(() => expect(screen.getByText('access denied')).toBeTruthy())
    expect(toast.success).not.toHaveBeenCalled()
    expect(pipedreamFrame()).toBeNull()
  })
})
