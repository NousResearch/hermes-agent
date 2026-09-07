import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = {
  addMcpServer: vi.fn(),
  authMcpServer: vi.fn(),
  cancelMcpOAuthFlow: vi.fn(),
  getActionStatus: vi.fn(),
  getMcpCatalog: vi.fn(),
  getMcpOAuthFlow: vi.fn(),
  installMcpCatalogEntry: vi.fn(),
  listMcpServers: vi.fn(),
  removeMcpServer: vi.fn(),
  setMcpServerEnabled: vi.fn(),
  testMcpServer: vi.fn()
}

vi.mock('@/hermes', () => api)

const searchMcpRegistry = vi.fn()

vi.mock('@/api/mcp', () => ({ searchMcpRegistry }))

const completeMcpDesktopOAuth = vi.fn()

vi.mock('@/lib/mcp-dashboard-oauth', () => ({
  completeMcpDesktopOAuth,
  McpOAuthCancelled: class McpOAuthCancelled extends Error {}
}))

vi.mock('@/lib/mcp-directory', () => ({
  MCP_DIRECTORY: [
    { name: 'figma', description: 'Design files', url: 'https://mcp.figma.com/mcp', docs: 'https://figma.com' }
  ]
}))

const {
  connectConnector,
  ConnectorCancelled,
  ConnectorNeedsAuth,
  connectorState,
  invalidateConnectorCache,
  listLocalConnectors,
  resolveConnectors,
  searchConnectors
} = await import('./mcp-connectors')

const catalogEntry = (name: string, extra: Record<string, unknown> = {}) => ({
  name,
  description: `${name} from the catalog`,
  source: 'https://example.com',
  transport: 'http',
  auth_type: 'none',
  required_env: [],
  command: null,
  args: [],
  url: `https://${name}.example.com/mcp`,
  install_url: null,
  install_ref: null,
  bootstrap: [],
  default_enabled: null,
  post_install: '',
  needs_install: false,
  installed: false,
  enabled: false,
  ...extra
})

const registryEntry = (name: string, extra: Record<string, unknown> = {}) => ({
  name,
  registry_name: `com.${name}/mcp`,
  title: name,
  description: `${name} from the registry`,
  url: `https://mcp.${name}.com/mcp`,
  transport: 'streamable-http',
  trust: 'verified',
  publisher: `${name}.com`,
  website: '',
  version: '1.0.0',
  headers: [],
  ...extra
})

beforeEach(() => {
  vi.clearAllMocks()
  invalidateConnectorCache()
  api.getMcpCatalog.mockResolvedValue({ entries: [], diagnostics: [] })
  searchMcpRegistry.mockResolvedValue({ entries: [] })
  window.hermesDesktop = { openExternal: vi.fn() } as never
})

describe('the resolution ladder', () => {
  it('prefers the reviewed catalog over the registry for the same name', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('notion')] })
    searchMcpRegistry.mockResolvedValue({ entries: [registryEntry('notion')] })

    const [connector] = (await resolveConnectors(['notion'])).connectors

    expect(connector).toMatchObject({ source: 'catalog', trust: 'catalog' })
    // A reviewed manifest short-circuits the ladder — no registry round-trip.
    expect(searchMcpRegistry).not.toHaveBeenCalled()
  })

  it('falls through the curated directory before reaching the registry', async () => {
    const [connector] = (await resolveConnectors(['figma'])).connectors

    expect(connector).toMatchObject({ auth: 'oauth', source: 'directory', trust: 'verified' })
    expect(searchMcpRegistry).not.toHaveBeenCalled()
  })

  it('reaches the registry only for names the local rungs did not answer', async () => {
    searchMcpRegistry.mockResolvedValue({ entries: [registryEntry('sentry')] })

    const [connector] = (await resolveConnectors(['sentry'])).connectors

    expect(connector).toMatchObject({ publisher: 'sentry.com', source: 'registry', trust: 'verified' })
  })

  it('ignores a registry near-miss rather than substituting it', async () => {
    // Searching "notion" surfaces "notion-helper"; offering that under the
    // name the agent asked for would connect something it never named.
    searchMcpRegistry.mockResolvedValue({ entries: [registryEntry('notion-helper')] })

    expect((await resolveConnectors(['notion'])).connectors).toEqual([])
  })

  it('keeps the order the caller asked in and drops unresolvable names', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('notion')] })

    const names = (await resolveConnectors(['figma', 'nonexistent', 'notion'])).connectors.map(entry => entry.name)

    expect(names).toEqual(['figma', 'notion'])
  })

  it('reports a fuzzy hit as found, under the name it was asked for', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('unreal-engine')] })

    // "Unreal" matched `unreal-engine` and rendered a card, yet the card also
    // said "Could not find Unreal": the miss was measured against the name we
    // resolved to rather than the one that was asked.
    const resolution = await resolveConnectors(['Unreal'])

    expect(resolution.connectors.map(entry => entry.name)).toEqual(['unreal-engine'])
    expect(resolution.unresolved).toEqual([])
    expect(searchMcpRegistry).not.toHaveBeenCalled()
  })

  it('collapses two spellings of one connector into a single entry', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('unreal-engine')] })

    const resolution = await resolveConnectors(['Unreal', 'unreal-engine'])

    expect(resolution.connectors).toHaveLength(1)
  })

  it('reports a miss in the caller‘s own spelling', async () => {
    searchMcpRegistry.mockResolvedValue({ entries: [] })

    expect((await resolveConnectors(['Rube Goldberg'])).unresolved).toEqual(['Rube Goldberg'])
  })

  it('degrades to the other rungs when the catalog is unreachable', async () => {
    api.getMcpCatalog.mockRejectedValue(new Error('backend down'))

    expect((await resolveConnectors(['figma'])).connectors[0]).toMatchObject({ source: 'directory' })
  })

  it('treats a documented secret header as an api_key requirement', async () => {
    searchMcpRegistry.mockResolvedValue({
      entries: [
        registryEntry('acme', {
          headers: [{ name: 'X-Api-Key', description: 'Your key', required: true, secret: true }]
        })
      ]
    })

    expect((await resolveConnectors(['acme'])).connectors[0]).toMatchObject({
      auth: 'api_key',
      requiredEnv: [{ name: 'X-Api-Key', prompt: 'Your key', required: true }]
    })
  })

  it('leaves auth unknown when the registry documents no credential', async () => {
    searchMcpRegistry.mockResolvedValue({ entries: [registryEntry('acme')] })

    expect((await resolveConnectors(['acme'])).connectors[0]?.auth).toBe('unknown')
  })
})

describe('search and local listing', () => {
  it('lists catalog and directory entries without querying the registry', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('notion')] })

    const names = (await listLocalConnectors()).map(entry => entry.name)

    expect(names).toEqual(['figma', 'notion'])
    expect(searchMcpRegistry).not.toHaveBeenCalled()
  })

  it('puts catalog hits ahead of registry hits and de-duplicates by name', async () => {
    api.getMcpCatalog.mockResolvedValue({ diagnostics: [], entries: [catalogEntry('notion')] })
    searchMcpRegistry.mockResolvedValue({ entries: [registryEntry('notion'), registryEntry('notion-helper')] })

    const results = await searchConnectors('notion')

    expect(results.map(entry => entry.name)).toEqual(['notion', 'notion-helper'])
    expect(results[0]?.source).toBe('catalog')
  })

  it('does not query on a one-character draft', async () => {
    expect(await searchConnectors('n')).toEqual([])
    expect(searchMcpRegistry).not.toHaveBeenCalled()
  })
})

describe('connectorState', () => {
  it.each([
    [[], 'not_configured'],
    [[{ name: 'linear', enabled: false }], 'disabled'],
    [[{ name: 'linear', enabled: true }], 'connected']
  ])('reads %j as %s', (servers, expected) => {
    expect(connectorState('linear', servers as { name: string; enabled: boolean }[])).toBe(expected)
  })
})

describe('connecting', () => {
  const connector = {
    auth: 'unknown' as const,
    description: '',
    docs: '',
    homepage: '',
    name: 'acme',
    needsInstall: false,
    publisher: 'acme.com',
    keywords: [],
    registryName: 'com.acme/mcp',
    requiredEnv: [],
    setup: [],
    source: 'registry' as const,
    title: 'Acme',
    trust: 'verified' as const,
    url: 'https://mcp.acme.com/mcp'
  }

  const never = () => false

  it('finds a connector by what people call it, not only by its name', async () => {
    // "google docs" shares no word with `google-workspace`; the manifest's own
    // suggestion keywords are the only thing that connects them.
    api.getMcpCatalog.mockResolvedValue({
      entries: [catalogEntry('google-workspace', { suggest: { keywords: ['google docs'], hosts: [] } })],
      diagnostics: []
    })

    const { connectors, unresolved } = await resolveConnectors(['Google Docs'])

    expect(connectors.map(entry => entry.name)).toEqual(['google-workspace'])
    expect(unresolved).toEqual([])
  })

  it('a switched-off connector is switched on, then made to prove it', async () => {
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })

    const phases: string[] = []

    const result = await connectConnector(connector, 'disabled', { cancelled: never, onPhase: p => phases.push(p) })

    expect(api.setMcpServerEnabled).toHaveBeenCalledWith('acme', true)
    expect(phases).toEqual(['enabling', 'probing'])
    expect(result.tools).toEqual(['search'])
    expect(api.addMcpServer).not.toHaveBeenCalled()
  })

  it('a no-auth connector connects without a browser round-trip', async () => {
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [] })

    await connectConnector({ ...connector, auth: 'none' }, 'not_configured', { cancelled: never })

    expect(api.addMcpServer).toHaveBeenCalledWith({ name: 'acme', url: 'https://mcp.acme.com/mcp' })
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
  })

  it('a connector that answers nothing is not a connected connector', async () => {
    // Writing config is not evidence. Without this the card goes green, the
    // model calls a tool three messages later, and the user reads the error.
    api.testMcpServer.mockResolvedValue({ ok: false, error: 'connection refused', tools: [] })

    await expect(
      connectConnector({ ...connector, auth: 'none' }, 'not_configured', { cancelled: never })
    ).rejects.toThrow('connection refused')
  })

  it('a refused credential is told apart from a dead endpoint', async () => {
    api.testMcpServer.mockResolvedValue({ ok: false, error: '401 unauthorized', tools: [] })

    await expect(
      connectConnector({ ...connector, auth: 'none' }, 'not_configured', { cancelled: never })
    ).rejects.toBeInstanceOf(ConnectorNeedsAuth)
  })

  it('a sign-in that reports no tools is checked rather than believed', async () => {
    completeMcpDesktopOAuth.mockResolvedValue({ tools: [] })
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })

    const result = await connectConnector({ ...connector, auth: 'oauth' }, 'needs_auth', { cancelled: never })

    expect(result.tools).toEqual(['search'])
  })

  it('a grant too narrow to use surfaces, even though the sign-in worked', async () => {
    // The insufficient-scope case: consent completed, the tools still refuse.
    completeMcpDesktopOAuth.mockResolvedValue({ tools: [] })
    api.testMcpServer.mockResolvedValue({ ok: false, error: '403 insufficient scope', tools: [] })

    await expect(
      connectConnector({ ...connector, auth: 'oauth' }, 'needs_auth', { cancelled: never })
    ).rejects.toBeInstanceOf(ConnectorNeedsAuth)
  })

  it('retrying a key-based connector re-checks it instead of opening a browser', async () => {
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })

    const result = await connectConnector({ ...connector, auth: 'api_key' }, 'connected', { cancelled: never })

    expect(result.tools).toEqual(['search'])
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
  })

  it('writes a corrected key before re-testing it', async () => {
    // Otherwise a wrong credential is permanent: the stored value is what
    // gets probed, so every retry fails on the same character.
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })
    api.installMcpCatalogEntry.mockResolvedValue({ background: false })

    const entry = { ...connector, auth: 'api_key' as const, source: 'catalog' as const }

    const result = await connectConnector(entry, 'needs_auth', { cancelled: never, env: { API_KEY: 'corrected' } })

    expect(api.installMcpCatalogEntry).toHaveBeenCalledWith(entry.name, { API_KEY: 'corrected' })
    expect(result.tools).toEqual(['search'])
  })

  it('probes an unknown-auth endpoint and stops there when it answers', async () => {
    // The reason a public server can be a plain switch: we find out by
    // asking it, not by assuming every remote needs OAuth.
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })

    const result = await connectConnector(connector, 'not_configured', { cancelled: never })

    expect(result.tools).toEqual(['search'])
    expect(completeMcpDesktopOAuth).not.toHaveBeenCalled()
  })

  it('falls through to sign-in only when the probe is refused', async () => {
    api.testMcpServer.mockResolvedValue({ ok: false, error: '401', tools: [] })
    completeMcpDesktopOAuth.mockResolvedValue({ tools: [{ name: 'search' }] })

    const phases: string[] = []

    const result = await connectConnector(connector, 'not_configured', {
      cancelled: never,
      onPhase: p => phases.push(p)
    })

    expect(phases).toEqual(['adding', 'probing', 'signing_in'])
    expect(result.tools).toEqual(['search'])
  })

  it('rolls the config write back when sign-in fails', async () => {
    api.testMcpServer.mockResolvedValue({ ok: false, tools: [] })
    completeMcpDesktopOAuth.mockRejectedValue(new Error('nope'))

    await expect(connectConnector(connector, 'not_configured', { cancelled: never })).rejects.toThrow('nope')

    // A half-configured entry would fail every later probe and read as a bug.
    expect(api.removeMcpServer).toHaveBeenCalledWith('acme')
  })

  it('rolls back on cancellation too', async () => {
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [] })

    await expect(connectConnector(connector, 'not_configured', { cancelled: () => true })).rejects.toBeInstanceOf(
      ConnectorCancelled
    )

    expect(api.removeMcpServer).toHaveBeenCalledWith('acme')
  })

  it('re-authorizing an existing connector never removes it on failure', async () => {
    completeMcpDesktopOAuth.mockRejectedValue(new Error('nope'))

    await expect(connectConnector(connector, 'needs_auth', { cancelled: never })).rejects.toThrow('nope')

    expect(api.removeMcpServer).not.toHaveBeenCalled()
    expect(api.addMcpServer).not.toHaveBeenCalled()
  })

  it('sends a catalog entry with credentials through the reviewed install path', async () => {
    const catalog = {
      ...connector,
      auth: 'api_key' as const,
      name: 'notion',
      requiredEnv: [{ name: 'NOTION_TOKEN', prompt: 'Token', required: true }],
      source: 'catalog' as const,
      trust: 'catalog' as const
    }

    api.installMcpCatalogEntry.mockResolvedValue({ ok: true })
    api.testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'search', description: '' }] })

    const result = await connectConnector(catalog, 'not_configured', {
      cancelled: never,
      env: { NOTION_TOKEN: 'secret' }
    })

    expect(api.installMcpCatalogEntry).toHaveBeenCalledWith('notion', { NOTION_TOKEN: 'secret' })
    expect(api.addMcpServer).not.toHaveBeenCalled()
    // The key the user typed is checked while they can still fix it.
    expect(result.tools).toEqual(['search'])
  })

  it('a wrong key is caught at the card, not on the first tool call', async () => {
    const catalog = {
      ...connector,
      auth: 'api_key' as const,
      name: 'notion',
      requiredEnv: [{ name: 'NOTION_TOKEN', prompt: 'Token', required: true }],
      source: 'catalog' as const,
      trust: 'catalog' as const
    }

    api.installMcpCatalogEntry.mockResolvedValue({ ok: true })
    api.testMcpServer.mockResolvedValue({ ok: false, error: '401 invalid token', tools: [] })

    await expect(
      connectConnector(catalog, 'not_configured', { cancelled: never, env: { NOTION_TOKEN: 'wrong' } })
    ).rejects.toBeInstanceOf(ConnectorNeedsAuth)

    // The install is reviewed and the key may be one character out — starting
    // over would throw away a correct config for a typo.
    expect(api.removeMcpServer).not.toHaveBeenCalled()
  })

  it('waits out a backgrounded catalog install and fails on a bad exit', async () => {
    api.installMcpCatalogEntry.mockResolvedValue({ ok: true, background: true, action: 'act-1' })
    api.getActionStatus.mockResolvedValue({ running: false, exit_code: 1 })

    await expect(
      connectConnector(
        { ...connector, name: 'notion', needsInstall: true, source: 'catalog', trust: 'catalog' },
        'not_configured',
        { cancelled: never }
      )
    ).rejects.toThrow(/Install failed/)
  })
})
