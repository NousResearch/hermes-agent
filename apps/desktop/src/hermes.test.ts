import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages, listAllProfileSessions, listSessions, searchSessions } from './hermes'

const emptySessionsResponse = {
  limit: 0,
  offset: 0,
  sessions: [],
  total: 0
}

describe('Hermes REST session helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue(emptySessionsResponse)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses a longer timeout for the single-profile session list', async () => {
    await listSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent',
        timeoutMs: 60_000
      })
    )
  })

  it('uses a longer timeout for the all-profile session list', async () => {
    await listAllProfileSessions(50, 1)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/profiles/sessions?limit=50&offset=0&min_messages=1&archived=exclude&order=recent&profile=all',
        timeoutMs: 60_000
      })
    )
  })

  it('tags cross-profile message reads for Electron routing and backend lookup', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu',
      profile: 'xiaoxuxu'
    })
  })

  it('searches all profiles by default with the cross-profile timeout', async () => {
    api.mockResolvedValue({ results: [] })

    await searchSessions('docker deploy')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/search?q=docker%20deploy&profile=all',
      timeoutMs: 60_000
    })
  })

  it('scopes search to a concrete profile without request-level routing', async () => {
    api.mockResolvedValue({ results: [] })

    await searchSessions('docker', 'coder')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/search?q=docker&profile=coder',
      timeoutMs: 60_000
    })
  })
})
