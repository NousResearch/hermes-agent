import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from './client'
import { connectDiscoveredMcp, discoverMcpServers } from './mcp-discovery'

afterEach(() => {
  setApiRequestConnection(null)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('routes both requests to the selected connection and profile without sending configuration', async () => {
  const api = vi.fn().mockResolvedValue({ candidates: [], warnings: [] })
  vi.stubGlobal('hermesDesktop', { api })
  setApiRequestConnection('remote-a')

  for (const scope of ['designer', { connectionId: 'local', profile: 'designer' }]) {
    const connectionId = typeof scope === 'string' ? 'remote-a' : 'local'
    await discoverMcpServers(scope)
    expect(api).toHaveBeenLastCalledWith({
      connectionId,
      profile: 'designer',
      path: '/api/mcp/discovery',
      method: 'POST',
      timeoutMs: 60_000
    })
    await connectDiscoveredMcp('opaque-candidate', scope)
    expect(api).toHaveBeenLastCalledWith({
      connectionId,
      profile: 'designer',
      path: '/api/mcp/discovery/connect',
      method: 'POST',
      body: { candidate_id: 'opaque-candidate' },
      timeoutMs: 60_000
    })
  }
})
