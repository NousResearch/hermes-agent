import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  capabilityScoped: vi.fn((scope?: { connectionId?: string; profile?: string }) =>
    scope?.connectionId ? { connectionId: scope.connectionId, priority: 'foreground' } : { profile: scope?.profile }
  ),
  hermesApi: vi.fn(),
  profileScoped: vi.fn()
}))

const { getComputerUseStatus } = await import('./toolsets')

afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.clearAllMocks()
})

describe('Computer Use target routing', () => {
  it('pins a Windows-host status probe to Desktop local instead of the active remote gateway', async () => {
    const api = vi.fn().mockResolvedValue({ platform: 'win32' })
    vi.stubGlobal('hermesDesktop', { api })

    await getComputerUseStatus('windows-host')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({
      connectionId: 'local',
      path: '/api/tools/computer-use/status',
      priority: 'foreground'
    }))
  })
})
