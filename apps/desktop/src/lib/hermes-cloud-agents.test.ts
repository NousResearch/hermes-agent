import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudAgent } from '@/global'

import { backup, connectCloudAgent } from './hermes-cloud-agents'

const AGENT: DesktopCloudAgent = {
  dashboardGatewayState: 'active',
  dashboardUrl: 'https://agent-1.cloud.example/',
  id: 'agent-1',
  name: 'Agent One',
  status: 'running'
}

afterEach(() => vi.unstubAllGlobals())

/** A connection registry the sign-in itself may write to, like the Webapp's saved-connection OAuth. */
function stubBridge({ signInSaves }: { signInSaves: boolean }) {
  const connections: { id: string; kind: string; url: string }[] = []

  const save = vi.fn(async (input: { url: string }) => {
    if (connections.some(c => c.url === input.url)) {
      throw new Error('A connection for this URL already exists')
    }

    const connection = { id: `saved-${connections.length + 1}`, kind: 'cloud', url: input.url }
    connections.push(connection)

    return { connection }
  })

  const agentSignIn = vi.fn(async (url: string) => {
    if (signInSaves) {
      connections.push({ id: 'signed-in', kind: 'cloud', url })
    }

    return { baseUrl: url, connected: true }
  })

  vi.stubGlobal('hermesDesktop', {
    api: vi.fn(),
    cloud: { agentSignIn },
    connections: { list: async () => ({ connections: [...connections] }), save }
  })

  return { connections, save }
}

describe('connectCloudAgent', () => {
  it('reuses the connection a first sign-in created instead of saving a duplicate', async () => {
    const { save } = stubBridge({ signInSaves: true })

    await expect(connectCloudAgent(AGENT)).resolves.toBe('signed-in')
    expect(save).not.toHaveBeenCalled()
  })

  it('prefers the connection id the sign-in reports', async () => {
    const { save } = stubBridge({ signInSaves: false })
    const desktop = window.hermesDesktop as unknown as { cloud: { agentSignIn: ReturnType<typeof vi.fn> } }
    desktop.cloud.agentSignIn.mockResolvedValueOnce({ baseUrl: '', connected: true, connectionId: 'from-sign-in' })

    await expect(connectCloudAgent(AGENT)).resolves.toBe('from-sign-in')
    expect(save).not.toHaveBeenCalled()
  })

  it('saves the agent once when the sign-in left no connection (native)', async () => {
    const { save } = stubBridge({ signInSaves: false })

    await expect(connectCloudAgent(AGENT, 'team')).resolves.toBe('saved-1')
    expect(save).toHaveBeenCalledWith({
      authMode: 'oauth',
      kind: 'cloud',
      label: 'Agent One',
      org: 'team',
      url: 'https://agent-1.cloud.example'
    })
  })
})

describe('backup', () => {
  it('saves the archive through the backup route of the backend that wrote it', async () => {
    const owner = { connectionId: 'ssh-1', profile: 'coder' }
    const archive = '/home/u/.hermes/profiles/coder/backups/hermes-backup.zip'

    const api = vi.fn(async (req: { path: string }) =>
      req.path === '/api/ops/backup' ? { archive, name: 'backup' } : { exit_code: 0, lines: [], running: false }
    )

    const saveGatewayFile = vi.fn(async () => ({ saved: true }))
    vi.stubGlobal('hermesDesktop', { api, saveGatewayFile })
    vi.useFakeTimers()

    try {
      const pending = backup(owner)
      await vi.runAllTimersAsync()
      await expect(pending).resolves.toEqual({ archive, saved: true })
    } finally {
      vi.useRealTimers()
    }

    expect(api.mock.calls.every(([req]) => (req as { connectionId?: string }).connectionId === 'ssh-1')).toBe(true)
    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'ssh-1',
      downloadPath: `/api/ops/backup/download?archive=${encodeURIComponent(archive)}`,
      path: archive,
      profile: 'coder',
      suggestedName: 'hermes-backup.zip'
    })
  })
})
