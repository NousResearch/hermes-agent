import { afterEach, describe, expect, it, vi } from 'vitest'

import { type BrowserBootstrap, pluginEventsTicketUrl } from './browser-transport'

// The browser-hosted Desktop dials a plugin's `/events` stream with a one-use ticket minted for
// that stream. Only `fetch` is stubbed: URL building and the verdict on a failed mint run for real.

const bootstrap = (basePath = ''): BrowserBootstrap => ({ authRequired: true, basePath, privateSession: false, token: '' })

function serveMint(status: number, body: unknown = { ticket: 't-1', ttl_seconds: 60 }) {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
    new Response(typeof body === 'string' ? body : JSON.stringify(body), { status })
  )

  vi.stubGlobal('fetch', fetchMock)

  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('pluginEventsTicketUrl', () => {
  it('mints for exactly the plugin stream and dials under the base path', async () => {
    const fetchMock = serveMint(200)

    const url = new URL((await pluginEventsTicketUrl(bootstrap('/hermes'), 'kanban', '/events')) as string)

    expect(url.pathname).toBe('/hermes/api/plugins/kanban/events')
    expect(url.searchParams.get('ticket')).toBe('t-1')

    const [minted, init] = fetchMock.mock.calls[0]
    const mintUrl = new URL(String(minted))
    expect(mintUrl.pathname).toBe('/hermes/api/webapp/socket-ticket')
    expect(mintUrl.searchParams.get('socket')).toBe('plugin-events/kanban')
    expect(init?.method).toBe('POST')
  })

  it('merges the ticket into the stream query without a second "?"', async () => {
    serveMint(200)

    const raw = (await pluginEventsTicketUrl(bootstrap(), 'kanban', '/events?board=ops&since=12')) as string
    const url = new URL(raw)

    expect(raw.match(/\?/g)).toHaveLength(1)
    expect(url.searchParams.get('board')).toBe('ops')
    expect(url.searchParams.get('since')).toBe('12')
    expect(url.searchParams.get('ticket')).toBe('t-1')
  })

  it('keeps a "?" inside the query value instead of dropping the rest', async () => {
    serveMint(200)

    const url = new URL((await pluginEventsTicketUrl(bootstrap(), 'kanban', '/events?board=a?b&since=3')) as string)

    expect(url.searchParams.get('board')).toBe('a?b')
    expect(url.searchParams.get('since')).toBe('3')
  })

  it('has no ticket path for a non-events path, an id the server would not name, or no bootstrap', async () => {
    const fetchMock = serveMint(200)

    expect(await pluginEventsTicketUrl(bootstrap(), 'kanban', '/board')).toBeNull()
    expect(await pluginEventsTicketUrl(bootstrap(), 'kanban', '/events/extra')).toBeNull()
    expect(await pluginEventsTicketUrl(bootstrap(), 'Kanban', '/events')).toBeNull()
    expect(await pluginEventsTicketUrl(bootstrap(), '../x', '/events')).toBeNull()
    expect(await pluginEventsTicketUrl(bootstrap(), 'a'.repeat(129), '/events')).toBeNull()
    expect(await pluginEventsTicketUrl(null, 'kanban', '/events')).toBeNull()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([400, 401, 403, 404])('settles on polling (null) when the mint is refused with %i', async status => {
    serveMint(status, status === 400 ? { detail: 'Unknown socket' } : 'nope')

    expect(await pluginEventsTicketUrl(bootstrap(), 'kanban', '/events')).toBeNull()
  })

  it('throws, so the caller retries, on a 5xx', async () => {
    serveMint(503, 'restarting')

    await expect(pluginEventsTicketUrl(bootstrap(), 'kanban', '/events')).rejects.toThrow(/^503:/)
  })

  it('throws on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))))

    await expect(pluginEventsTicketUrl(bootstrap(), 'kanban', '/events')).rejects.toThrow('Failed to fetch')
  })

  it('throws when the mint answers without a ticket', async () => {
    serveMint(200, {})

    await expect(pluginEventsTicketUrl(bootstrap(), 'kanban', '/events')).rejects.toThrow(/ticket/)
  })
})
