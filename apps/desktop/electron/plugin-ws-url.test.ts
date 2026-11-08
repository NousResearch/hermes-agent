import { expect, test, vi } from 'vitest'

import { createLocalGatewayDials, type GatewayEndpoint } from './local-gateway'
import { createPluginWsUrlHandler, pluginWsPath } from './plugin-ws-url'

// PLG-1: a local gateway's plugin socket rides the native-http grant its REST twin uses. The grant
// is attached as a header to the one dial main prepared, never placed in the renderer's URL.

const endpoint = { api_origin: 'http://127.0.0.1:9', control_home: null, profile_id: '/home/u/.hermes' } as unknown as GatewayEndpoint

test('a plugin dial carries the grant header once and never in its URL', () => {
  const dials = createLocalGatewayDials()
  const url = dials.preparePlugin('http://127.0.0.1:9', '/api/plugins/kanban/events?board=b', 'grant-1', 7)
  const parsed = new URL(url)

  expect(url).not.toContain('grant-1')
  expect(parsed.pathname).toBe('/api/plugins/kanban/events')
  expect(parsed.searchParams.get('board')).toBe('b')
  const details = { url, webContentsId: 7, resourceType: 'webSocket', requestHeaders: { Origin: 'file://', 'x-hermes-gateway-ticket': 'forged' } }
  expect(dials.headers({ ...details, webContentsId: 8 })).toBeNull()
  // A different path carrying the same nonce gets nothing.
  expect(dials.headers({ ...details, url: url.replace('/kanban/events', '/other/events') })).toBeNull()
  // The same nonce and path aimed at another server, scheme or port gets nothing.
  for (const elsewhere of ['ws://evil.example:9', 'wss://127.0.0.1:9', 'ws://127.0.0.1:10', 'ws://localhost:9']) {
    expect(dials.headers({ ...details, url: url.replace('ws://127.0.0.1:9', elsewhere) })).toBeNull()
  }

  expect(dials.headers(details)).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-1' })
  expect(dials.headers(details)).toBeNull()
})

test('the gateway socket dial still carries no grant header', () => {
  const dials = createLocalGatewayDials()
  const dial = new URL(dials.prepare('http://127.0.0.1:9', 'interactive-1', 7))
  dial.searchParams.delete('ticket')

  expect(dials.headers({ url: dial.toString(), webContentsId: 7, resourceType: 'webSocket', requestHeaders: { Origin: 'file://' } })).toEqual({})
})

test('plugin socket paths stay inside the plugin namespace', () => {
  expect(pluginWsPath('kanban', '/events?board=a/../b')).toBe('/api/plugins/kanban/events?board=a/../b')

  for (const [id, path] of [['kanban', '/../../api/ws'], ['kanban', 'events'], ['..', '/x'], ['a/b', '/x'], ['kanban', '/a\\b'], ['kanban', '/./x'], [1, '/x'], ['kanban', null]]) {
    expect(() => pluginWsPath(id, path)).toThrow()
  }
})

test('encoded, dot-segment and backslash paths stay inside the plugin namespace through URL parsing', () => {
  for (const path of ['/%2e%2e/other/events', '/.%2e/other/events', '/%2E%2E/%2e%2e/api/ws', '/x/%2e%2e/%2e%2e/other/events',
    '/..%2fother/events', '/%2f..%2fother', '/a%2Fb', '/%5c..%5cother', '/%252e%252e/other', '/%2e/x', '/a/%2e%2e', '/%zz']) {
    expect(() => pluginWsPath('kanban', path), path).toThrow('Invalid plugin socket path')
  }

  // What the dial records and attaches the grant to is the path that was checked.
  const path = pluginWsPath('kanban', '/boards/b%20c/events?q=%2e%2e%2f')
  const dials = createLocalGatewayDials()
  const url = new URL(dials.preparePlugin('http://127.0.0.1:9', path, 'grant-3', 7))
  expect(url.pathname).toBe('/api/plugins/kanban/boards/b%20c/events')
  expect(path).toBe(`${url.pathname}?q=%2e%2e%2f`)
})

test('mints a native-http grant for a local gateway and null elsewhere', async () => {
  const mint = vi.fn(async () => 'grant-2')
  const prepare = vi.fn((origin: string, path: string, ticket: string, id: number) => `${origin}|${path}|${ticket}|${id}`)

  const handler = createPluginWsUrlHandler({
    ensureBackend: vi.fn(async () => ({ baseUrl: 'http://127.0.0.1:9', gatewayEndpoint: endpoint })),
    ensureRegistryBackend: vi.fn(async () => ({ baseUrl: 'https://remote.invalid' })),
    forgetLocalGatewayDescriptor: vi.fn(),
    forgetRegistryLocalGatewayDescriptor: vi.fn(),
    launchHome: '/home/u/.hermes',
    mint,
    prepare
  })

  await expect(handler({ path: '/events', pluginId: 'kanban', profile: 'default' }, 3)).resolves.toBe('http://127.0.0.1:9|/api/plugins/kanban/events|grant-2|3')
  expect(mint).toHaveBeenCalledWith(expect.objectContaining({ profile_id: '/home/u/.hermes' }), 'native-http')
  await expect(handler({ connectionId: 'homelab', path: '/events', pluginId: 'kanban' }, 3)).resolves.toBeNull()
  await expect(handler({ path: '/../ws', pluginId: 'kanban' }, 3)).rejects.toThrow('Invalid plugin socket path')
  await expect(handler({ path: '/%2e%2e/other/events', pluginId: 'kanban' }, 3)).rejects.toThrow('Invalid plugin socket path')
  expect(mint).toHaveBeenCalledTimes(1)
})

// The renderer cannot tell a refused upgrade from a gateway that is away. Main can name the one case it
// knows is not transient, a grant mint refused for a permission reason, and answers it as plain data.
test('a mint refused for a permission reason is a named reply, any other failure stays an error', async () => {
  const unsafe = new Error('Unsafe gateway control path: \'/home/u/.hermes\' is accessible to other users (mode 755): run chmod go-rwx \'/home/u/.hermes\'')
  const groupWritable = new Error('Profile home is writable by a group other accounts share: run chmod g-w \'/home/u/.hermes\'')
  const bridged = Object.assign(new Error('Gateway ticket bootstrap failed'), { reason: 'unsafe_control_permissions' })
  const mint = vi.fn(async (): Promise<string> => { throw unsafe })

  const handler = createPluginWsUrlHandler({
    ensureBackend: vi.fn(async () => ({ baseUrl: 'http://127.0.0.1:9', gatewayEndpoint: endpoint })),
    ensureRegistryBackend: vi.fn(),
    forgetLocalGatewayDescriptor: vi.fn(),
    forgetRegistryLocalGatewayDescriptor: vi.fn(),
    launchHome: '/home/u/.hermes',
    mint,
    prepare: vi.fn(() => 'unused')
  })

  const request = { path: '/events', pluginId: 'kanban', profile: 'default' }

  for (const error of [unsafe, groupWritable, bridged]) {
    mint.mockRejectedValueOnce(error)
    await expect(handler(request, 3)).resolves.toEqual({ refused: 'permission' })
  }

  // An owner that is gone (redialed once, then still gone) is a connection failure: the renderer retries.
  mint.mockRejectedValue(new Error('Gateway ticket control socket missing'))
  await expect(handler(request, 3)).rejects.toThrow('Gateway ticket control socket missing')
})

test('the shared local dial keeps the gateway socket on an interactive grant', async () => {
  const { localGatewayDial } = await import('./local-gateway')
  const mint = vi.fn(async () => 'interactive-2')
  const url = await localGatewayDial({ ensure: async () => ({ baseUrl: 'http://127.0.0.1:9', gatewayEndpoint: endpoint }), forget: vi.fn(), launchHome: '/home/u/.hermes', mint, prepare: (baseUrl, ticket) => `${baseUrl}|${ticket}`, profile: '' })

  expect(url).toBe('http://127.0.0.1:9|interactive-2')
  expect(mint).toHaveBeenCalledWith(expect.objectContaining({ profile_id: '/home/u/.hermes' }), 'interactive')
})
