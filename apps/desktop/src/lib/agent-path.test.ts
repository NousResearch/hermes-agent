import { beforeEach, describe, expect, it, vi } from 'vitest'

const { hermesApi, isDesktopFsRemoteMode } = vi.hoisted(() => ({
  hermesApi: vi.fn(),
  isDesktopFsRemoteMode: vi.fn(() => false)
}))

vi.mock('@/api/client', () => ({ hermesApi }))
vi.mock('@/lib/desktop-fs', () => ({ desktopFsProfile: () => undefined, isDesktopFsRemoteMode }))

import { resolveAgentHostPath } from './agent-path'

describe('resolveAgentHostPath', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    isDesktopFsRemoteMode.mockReturnValue(false)
  })

  it('returns the host path the local backend maps a container path to', async () => {
    hermesApi.mockResolvedValue({ path: '/home/user/projects/report.md' })

    await expect(resolveAgentHostPath('/workspace/report.md')).resolves.toBe('/home/user/projects/report.md')
    expect(hermesApi).toHaveBeenCalledWith({
      path: '/api/fs/resolve-agent-path?path=%2Fworkspace%2Freport.md',
      profile: undefined
    })
  })

  it('returns null when the backend misses or answers with the same path', async () => {
    hermesApi.mockRejectedValueOnce(new Error('404'))
    hermesApi.mockResolvedValueOnce({ path: '/srv/report.md' })

    await expect(resolveAgentHostPath('/workspace/gone.md')).resolves.toBeNull()
    await expect(resolveAgentHostPath('/srv/report.md')).resolves.toBeNull()
  })

  it('never asks in remote mode or for non-POSIX paths', async () => {
    await expect(resolveAgentHostPath('C:\\report.md')).resolves.toBeNull()
    isDesktopFsRemoteMode.mockReturnValue(true)
    await expect(resolveAgentHostPath('/workspace/report.md')).resolves.toBeNull()
    expect(hermesApi).not.toHaveBeenCalled()
  })
})
