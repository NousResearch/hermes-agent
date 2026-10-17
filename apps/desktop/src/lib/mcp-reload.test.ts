import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  announceMcpReload,
  MCP_RELOAD_TIMEOUT_MS,
  mcpReloadSlashParams,
  parseMcpReloadReply,
  reloadMcp,
  reloadMcpFor
} from './mcp-reload'

const mocks = vi.hoisted(() => ({ gateway: null as unknown, notify: vi.fn() }))

vi.mock('@/store/gateway', () => ({ $gateway: { get: () => mocks.gateway } }))
vi.mock('@/store/notifications', () => ({ notify: mocks.notify }))

// The Capabilities save path: reload, then tell the user what did not take effect.
const save = async (result: unknown, profile?: string) => {
  const request = vi.fn().mockResolvedValue(result)
  announceMcpReload(await reloadMcp({ request } as never, 's1', profile))

  return request
}

describe('a saved MCP change reports what the chat got', () => {
  beforeEach(() => mocks.notify.mockReset())

  it('stays quiet when the chat runs the change', async () => {
    const request = await save({ session: 'applied', status: 'reloaded' })

    expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, session_id: 's1' }, MCP_RELOAD_TIMEOUT_MS)
    expect(mocks.notify).not.toHaveBeenCalled()
  })

  it('says the change is deferred while the chat is busy, and when it applies', async () => {
    await save({ session: 'busy', status: 'reloaded' })

    expect(mocks.notify).toHaveBeenCalledTimes(1)
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ message: expect.stringMatching(/busy.*(turn|reopen)/i) })
    )
  })

  it('names every server that did not connect beside a partial success', async () => {
    await save({ session: 'applied', status: 'reloaded', unavailable: ['broken', 'gone'] })

    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: expect.stringContaining('broken, gone') })
    )
  })

  it('still reads the failed list an older gateway answers', async () => {
    await save({ session: 'applied', status: 'reloaded', failed: ['legacy'] })

    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: expect.stringContaining('legacy') })
    )
  })

  it('reports both a deferral and an unavailable server', async () => {
    await save({ session: 'busy', status: 'reloaded', unavailable: ['broken'] })

    expect(mocks.notify).toHaveBeenCalledTimes(2)
  })

  it('lets a refused reload reach the caller', async () => {
    const request = vi.fn().mockRejectedValue(new Error('mcp_servers_unavailable'))

    await expect(reloadMcp({ request } as never, null)).rejects.toThrow('mcp_servers_unavailable')
    expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, session_id: undefined }, MCP_RELOAD_TIMEOUT_MS)
  })

  it('asks for the pane profile and retries once without it on a gateway that predates it', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error('invalid params for reload.mcp: profile: unexpected key'))
      .mockResolvedValueOnce({ session: 'applied', status: 'reloaded' })

    await expect(reloadMcp({ request } as never, 's1', 'work')).resolves.toEqual(
      expect.objectContaining({ deferred: false, session: 'applied' })
    )
    expect(request).toHaveBeenNthCalledWith(
      1,
      'reload.mcp',
      { confirm: true, profile: 'work', session_id: 's1' },
      MCP_RELOAD_TIMEOUT_MS
    )
    expect(request).toHaveBeenNthCalledWith(2, 'reload.mcp', { confirm: true, session_id: 's1' }, MCP_RELOAD_TIMEOUT_MS)
  })

  it('does not retry a failure that is not the profile-admission rejection', async () => {
    const request = vi.fn().mockRejectedValue(new Error('session busy'))

    await expect(reloadMcp({ request } as never, 's1', 'work')).rejects.toThrow('session busy')
    expect(request).toHaveBeenCalledTimes(1)
  })
})

describe('parseMcpReloadReply', () => {
  it('defaults every optional field an older gateway omits', () => {
    expect(parseMcpReloadReply(null)).toEqual({ message: null, session: null, status: null, unavailable: [] })
  })

  it('keeps only string names and known session states', () => {
    expect(
      parseMcpReloadReply({
        message: 'ok',
        session: 'queued',
        status: 'reloaded',
        unavailable: ['broken', 2, null]
      })
    ).toEqual({ message: 'ok', session: null, status: 'reloaded', unavailable: ['broken'] })
  })
})

describe('mcpReloadSlashParams', () => {
  const cases: [string, Record<string, unknown>][] = [
    ['', { session_id: 's-1' }],
    ['now', { confirm: true, session_id: 's-1' }],
    ['approve', { confirm: true, session_id: 's-1' }],
    ['once', { confirm: true, session_id: 's-1' }],
    ['yes', { confirm: true, session_id: 's-1' }],
    ['always', { always: true, confirm: true, session_id: 's-1' }],
    ['bogus', { session_id: 's-1' }]
  ]

  it.each(cases)('maps %s to its reload parameters', (arg, expected) => {
    expect(mcpReloadSlashParams(arg, 's-1')).toEqual(expected)
  })

  it('omits a session a draft does not have', () => {
    expect(mcpReloadSlashParams('always', null)).toEqual({ always: true, confirm: true })
  })
})

describe('a flow that claims a server is live in the chat', () => {
  const claim = (result: unknown) => {
    mocks.gateway = { request: vi.fn().mockResolvedValue(result) }

    return reloadMcpFor('fixt', 's1')
  }

  beforeEach(() => mocks.notify.mockReset())

  it('resolves once the chat runs the server', async () => {
    await expect(claim({ session: 'applied', session_servers: ['fixt'], status: 'reloaded' })).resolves.toBeUndefined()
  })

  it('refuses when the chat does not run that server (no_mcp, an explicit pick, an allowlist)', async () => {
    await expect(claim({ session: 'unchanged', session_servers: [], status: 'reloaded' })).rejects.toThrow(/fixt/)
  })

  it('keeps the earlier answer from a gateway that does not report the selection', async () => {
    await expect(claim({ session: 'applied', status: 'reloaded' })).resolves.toBeUndefined()
  })
})
