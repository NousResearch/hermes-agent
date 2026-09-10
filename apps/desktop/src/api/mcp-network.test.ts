import { afterEach, describe, expect, it, vi } from 'vitest'

import { parseMcpImport } from '@/lib/mcp-import'
import { getServers } from '@/lib/mcp-servers'

import { addMcpServer, installMcpCatalogEntry, saveMcpServers } from './mcp'

afterEach(() => vi.unstubAllGlobals())

describe('MCP network intent at the Desktop API boundary', () => {
  it('keeps the explicit owner and network through create, bulk save and catalog install', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    vi.stubGlobal('hermesDesktop', { api })
    const scope = { connectionId: 'local', profile: 'unity' }
    const config = { url: 'http://localhost:8080/sse', network: 'windows' as const, transport: 'sse' as const }
    await addMcpServer({ name: 'unity', ...config }, scope)
    expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ ...scope, body: { name: 'unity', ...config } }))
    await addMcpServer({ name: 'new-default', url: 'http://localhost:8080/mcp' }, scope)
    expect(api).toHaveBeenLastCalledWith(
      expect.objectContaining({ ...scope, body: { name: 'new-default', url: 'http://localhost:8080/mcp', network: 'auto' } })
    )
    await saveMcpServers({ unity: config }, scope)
    expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ ...scope, body: { servers: { unity: config } } }))
    await installMcpCatalogEntry('unity', {}, scope, 'windows')
    expect(api).toHaveBeenLastCalledWith(
      expect.objectContaining({ ...scope, body: { name: 'unity', env: {}, enable: true, network: 'windows' } })
    )
  })

  it('round-trips pasted cross-environment SSE entries through the shared config reader', () => {
    const imported = parseMcpImport(
      JSON.stringify({ mcpServers: { unity: { type: 'sse', network: 'windows', url: 'http://localhost:8080/sse' } } })
    )

    expect(imported).toEqual([
      { name: 'unity', config: { transport: 'sse', network: 'windows', url: 'http://localhost:8080/sse' } }
    ])
    expect(
      getServers({ mcp_servers: Object.fromEntries(imported!.map(entry => [entry.name, entry.config])) }).unity
    ).toEqual(imported![0].config)
  })
})
