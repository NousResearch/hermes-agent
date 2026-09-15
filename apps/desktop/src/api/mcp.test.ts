import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from './client'
import { setMcpServerEnabled } from './mcp'

describe('MCP enablement routing', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses the captured owner after the ambient connection and profile move on', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    setApiRequestConnection('connection-a')
    setApiRequestProfile('profile-a')
    const owner = { connectionId: 'connection-a', profile: 'profile-a' } as const

    setApiRequestConnection('connection-b')
    setApiRequestProfile('profile-b')
    await setMcpServerEnabled('linear', false, owner)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'connection-a',
        profile: 'profile-a',
        path: '/api/mcp/servers/linear/enabled',
        method: 'PUT',
        body: { enabled: false }
      })
    )
  })

  it('keeps an explicit local pin when the ambient registry primary is remote', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    setApiRequestConnection('remote-primary')
    setApiRequestProfile('default')
    await setMcpServerEnabled('filesystem', true, { connectionId: 'local', profile: 'default' })

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'local', profile: 'default', path: '/api/mcp/servers/filesystem/enabled' })
    )
  })

  it('does not invent a local pin for the legacy profile-remote compatibility path', async () => {
    const api = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    setApiRequestProfile('remote-profile')
    await setMcpServerEnabled('linear', false, { profile: 'remote-profile' })

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ profile: 'remote-profile', path: '/api/mcp/servers/linear/enabled' })
    )
    expect(api.mock.calls[0][0]).not.toHaveProperty('connectionId')
  })
})
