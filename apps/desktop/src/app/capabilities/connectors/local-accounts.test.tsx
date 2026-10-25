import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { McpCatalogEntry } from '@/hermes'
import { I18nProvider } from '@/i18n'
import type * as McpOAuth from '@/lib/mcp-dashboard-oauth'
import type { McpServers } from '@/lib/mcp-servers'

import { LocalAccounts } from './local-accounts'

const rpc = vi.hoisted(() => vi.fn())
const oauth = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: rpc }))
vi.mock('@/lib/mcp-dashboard-oauth', async importOriginal => ({
  ...(await importOriginal<typeof McpOAuth>()),
  completeMcpDesktopOAuth: oauth
}))

const scope = { connectionId: 'synthetic-remote', profile: 'profile-a' }
const catalog = [{ name: 'notion', url: 'https://mcp.notion.com/mcp', auth_type: 'oauth' }] as McpCatalogEntry[]

const original = {
  url: catalog[0].url,
  auth: 'oauth',
  enabled: false,
  tools: { exclude: ['delete-page'] },
  headers: { Authorization: 'do-not-copy' }
}

const servers: McpServers = { notion: original }

function show(entries = servers, profile = scope) {
  const refresh = vi.fn(async () => undefined)
  const open = vi.fn()

  const view = render(
    <I18nProvider>
      <LocalAccounts
        catalog={catalog}
        onChanged={refresh}
        onOpenAccount={open}
        pending={false}
        profile={profile}
        serverName="notion"
        servers={entries}
      />
    </I18nProvider>
  )

  return { ...view, refresh, open }
}

async function connect(alias: string) {
  fireEvent.click(screen.getByRole('button', { name: 'Add another account' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Account name' }), { target: { value: alias } })
  fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('direct MCP additional accounts', () => {
  it('adds a named catalog account in the selected remote profile without copying credentials or changing its sibling', async () => {
    rpc.mockResolvedValue({ ok: true })
    oauth.mockResolvedValue({ tools: [] })
    const { open } = show()
    await connect('work')
    await waitFor(() => expect(open).toHaveBeenCalledWith('notion-work'))
    expect(rpc).toHaveBeenCalledWith(
      'synthetic-remote',
      'profile-a',
      'mcp.servers.add',
      {
        name: 'notion-work',
        preset: 'notion',
        config: { tools: { exclude: ['delete-page'] } }
      },
      60_000
    )
    expect(oauth).toHaveBeenCalledWith(
      expect.objectContaining({ serverName: 'notion-work', profile: expect.objectContaining(scope) })
    )
    expect(servers).toEqual({ notion: original })
  })

  it('retains a failed account and retries authorization without creating it again', async () => {
    rpc.mockResolvedValue({ ok: true })
    oauth.mockRejectedValueOnce(new Error('Consent denied')).mockResolvedValueOnce({ tools: [] })
    const { open } = show()
    await connect('home')
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Consent denied'))
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(open).toHaveBeenCalledWith('notion-home'))
    expect(rpc).toHaveBeenCalledTimes(1)
    expect(oauth).toHaveBeenCalledTimes(2)
  })

  it('passes successful retry tools to the probe controller instead of leaving the failed probe cached', async () => {
    const tools = [{ name: 'list-pages', description: 'List pages' }]
    rpc.mockResolvedValue({ ok: true })
    oauth.mockRejectedValueOnce(new Error('Consent denied')).mockResolvedValueOnce({ tools })
    const { refresh } = show()

    await connect('home')
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Consent denied'))
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(refresh).toHaveBeenCalledWith({ name: 'notion-home', tools }))
  })

  it('publishes a new account only after OAuth succeeds, so its first probe sees authorized credentials', async () => {
    let approve!: (value: { tools: [] }) => void
    rpc.mockResolvedValue({ ok: true })
    oauth.mockReturnValue(new Promise(resolve => { approve = resolve }))
    const { refresh } = show()

    await connect('work')
    await waitFor(() => expect(oauth).toHaveBeenCalledTimes(1))
    expect(refresh).not.toHaveBeenCalled()

    approve({ tools: [] })
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
  })

  it('rejects names that collide with an existing token or tool identity', async () => {
    show({ ...servers, notion_work: { url: 'https://another.example/mcp', auth: 'oauth' } })
    await connect('work')
    expect(screen.getByRole('alert').textContent).toContain('already used')
    expect(rpc).not.toHaveBeenCalled()
    expect(oauth).not.toHaveBeenCalled()
  })

  it('never starts OAuth on the replacement profile after creation completes on an unmounted source', async () => {
    let finish!: (value: { ok: true }) => void
    rpc.mockReturnValueOnce(
      new Promise(resolve => {
        finish = resolve
      })
    )
    const { unmount, refresh } = show()
    await connect('work')
    await waitFor(() => expect(rpc).toHaveBeenCalledTimes(1))
    unmount()
    finish({ ok: true })
    await Promise.resolve()
    expect(oauth).not.toHaveBeenCalled()
    expect(refresh).not.toHaveBeenCalled()
  })

  it('cancels the visible interaction immediately while the source authorization unwinds', async () => {
    let reject!: (reason: Error) => void

    const promise = new Promise<never>((_resolve, fail) => {
      reject = fail
    })

    rpc.mockResolvedValue({ ok: true })
    oauth.mockReturnValueOnce(promise)
    show()
    await connect('cancelled')
    await waitFor(() => expect(oauth).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('textbox', { name: 'Account name' })).toBeNull()
    expect(oauth.mock.calls[0][0].cancelled()).toBe(true)
    reject(new Error('Authorization cancelled'))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Add another account' }).hasAttribute('disabled')).toBe(false)
    )
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('shows a server-side collision without starting OAuth and permits correcting the name', async () => {
    rpc.mockRejectedValueOnce(new Error('That name is already used.')).mockResolvedValueOnce({ ok: true })
    oauth.mockResolvedValue({ tools: [] })
    const { open } = show()
    await connect('work')
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('already used'))
    expect(oauth).not.toHaveBeenCalled()
    fireEvent.change(screen.getByRole('textbox', { name: 'Account name' }), { target: { value: 'home' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    await waitFor(() => expect(open).toHaveBeenCalledWith('notion-home'))
  })

  it('does not reuse a cancelled creation when the user starts another account', async () => {
    let finish!: (value: { ok: true }) => void
    rpc
      .mockReturnValueOnce(
        new Promise(resolve => {
          finish = resolve
        })
      )
      .mockResolvedValue({ ok: true })
    oauth.mockResolvedValue({ tools: [] })
    const { open } = show()
    await connect('cancelled')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    finish({ ok: true })
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Add another account' }).hasAttribute('disabled')).toBe(false)
    )
    await connect('replacement')
    await waitFor(() => expect(open).toHaveBeenCalledWith('notion-replacement'))
  })

  it('shows three separately addressable Notion entries', () => {
    show({
      ...servers,
      'notion-work': { url: catalog[0].url, auth: 'oauth' },
      'notion-home': { url: catalog[0].url, auth: 'oauth' }
    })

    for (const name of ['notion', 'notion-work', 'notion-home']) {
      const account = screen.getByRole('button', { name })
      expect(account.getAttribute('aria-current')).toBe(name === 'notion' ? 'true' : null)
    }
  })

  it('creates a second Attio account through the Attio preset', async () => {
    rpc.mockResolvedValue({ ok: true })
    oauth.mockResolvedValue({ tools: [] })
    const attioCatalog = [{ name: 'attio', url: 'https://mcp.attio.com/mcp', auth_type: 'oauth' }] as McpCatalogEntry[]
    const open = vi.fn()

    render(
      <I18nProvider>
        <LocalAccounts
          catalog={attioCatalog}
          onChanged={vi.fn(async () => undefined)}
          onOpenAccount={open}
          pending={false}
          profile={scope}
          serverName="attio"
          servers={{ attio: { url: attioCatalog[0].url, auth: 'oauth' } }}
        />
      </I18nProvider>
    )
    await connect('second')
    await waitFor(() => expect(open).toHaveBeenCalledWith('attio-second'))
    expect(rpc).toHaveBeenCalledWith('synthetic-remote', 'profile-a', 'mcp.servers.add', {
      name: 'attio-second', preset: 'attio', config: {}
    }, 60_000)
  })
})
