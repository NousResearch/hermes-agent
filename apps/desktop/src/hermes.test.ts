import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, listAllProfileSessions, listSessions, setSessionArchived } from './hermes'

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

  // The owning profile must reach the SERVER as ?profile=, not just Electron's
  // backend router: the serving process can be scoped to a different profile
  // than the desktop believes (sticky active_profile honored on a legacy
  // launch), so without it the delete 404s against the wrong state.db (#44117).
  it('passes the owning profile to the server when deleting a session', async () => {
    await deleteSession('sess-1', 'default')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        method: 'DELETE',
        path: '/api/sessions/sess-1?profile=default',
        profile: 'default'
      })
    )
  })

  it('omits the profile param when deleting without an owning profile', async () => {
    await deleteSession('sess-1')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        method: 'DELETE',
        path: '/api/sessions/sess-1'
      })
    )
    expect(api.mock.calls[0][0]).not.toHaveProperty('profile')
  })

  it('passes the owning profile in the body when archiving a session', async () => {
    await setSessionArchived('sess-1', true, 'default')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: { archived: true, profile: 'default' },
        method: 'PATCH',
        path: '/api/sessions/sess-1',
        profile: 'default'
      })
    )
  })
})
