import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile, STARTUP_REQUEST_TIMEOUT_MS } from './client'
import { getHermesConfigRecord } from './config'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

describe('capability config startup reads', () => {
  it('keeps the startup deadline and the selected owner on every MCP config read', async () => {
    const api = vi.fn().mockResolvedValue({ mcp_servers: {} })
    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('foreground')
    setApiRequestProfile('chat')

    await getHermesConfigRecord()
    expect(api).toHaveBeenLastCalledWith({
      connectionId: 'foreground',
      profile: 'chat',
      path: '/api/config',
      timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
    })
    await getHermesConfigRecord({ connectionId: 'local', profile: 'unity' })
    expect(api).toHaveBeenLastCalledWith({
      connectionId: 'local',
      profile: 'unity',
      path: '/api/config',
      timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
    })
    await getHermesConfigRecord({ connectionId: 'remote', profile: 'default' })
    expect(api).toHaveBeenLastCalledWith({
      connectionId: 'remote',
      profile: 'default',
      path: '/api/config',
      timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
    })
  })
})
