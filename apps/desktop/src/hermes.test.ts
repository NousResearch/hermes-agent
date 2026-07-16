import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getCronJobs,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getHermesConfig,
  getHermesConfigDefaults,
  getProfiles,
  getSessionMessages,
  getStatus,
  listAllProfileSessions,
  listSessions,
  setApiRequestProfile
} from './hermes'
import { refreshActiveProfile } from './store/profile'

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
    setApiRequestProfile(null)
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

  it('uses a longer timeout for profile listing during desktop startup', async () => {
    api.mockResolvedValue({ profiles: [] })

    await getProfiles()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/profiles',
        timeoutMs: 60_000
      })
    )
  })

  it('uses a longer timeout for active profile refresh during desktop startup', async () => {
    api.mockResolvedValueOnce({ current: 'default' }).mockResolvedValueOnce({ profiles: [] })

    await refreshActiveProfile()

    expect(api).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: '/api/profiles/active',
        timeoutMs: 60_000
      })
    )
    expect(api).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: '/api/profiles',
        timeoutMs: 60_000
      })
    )
  })

  it('gives the whole startup data burst the long timeout, not just profiles', async () => {
    setApiRequestProfile(null)
    api.mockResolvedValue({})

    const bootCalls: [() => Promise<unknown>, string][] = [
      [getHermesConfig, '/api/config'],
      [getHermesConfigDefaults, '/api/config/defaults'],
      [getGlobalModelInfo, '/api/model/info'],
      [() => getGlobalModelOptions(), '/api/model/options?explicit_only=1'],
      [getCronJobs, '/api/cron/jobs']
    ]

    for (const [call, path] of bootCalls) {
      api.mockClear()
      await call()
      expect(api).toHaveBeenCalledWith(expect.objectContaining({ path, timeoutMs: 60_000 }))
    }
  })

  it('keeps the liveness poll on the short default so a dead backend fails fast', async () => {
    api.mockResolvedValue({})
    api.mockClear()

    await getStatus()

    // /api/status must NOT carry the long startup timeout — it is the runtime
    // liveness probe and has to fail quickly when the backend drops.
    const call = api.mock.calls[0]?.[0] as { path: string; timeoutMs?: number }
    expect(call.path).toBe('/api/status')
    expect(call.timeoutMs).toBeUndefined()
  })

  it('tags cross-profile message reads for Electron routing and backend lookup', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu',
      profile: 'xiaoxuxu'
    })
  })

  it('defaults model options to configured providers only', async () => {
    await getGlobalModelOptions()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/model/options?explicit_only=1'
      })
    )
  })

  it('can opt into unconfigured providers for onboarding flows', async () => {
    await getGlobalModelOptions({ includeUnconfigured: true, refresh: true, explicitOnly: false })

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/model/options?refresh=1&include_unconfigured=1'
      })
    )
  })
})

describe('Hermes REST cron helpers honour the active profile', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue([])
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    // _apiProfile is module-level state in hermes.ts; reset so it never leaks
    // into the session helpers describe above (or another suite entirely).
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('scopes the cron job listing to the active profile', async () => {
    setApiRequestProfile('worker_alpha')

    await getCronJobs()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/cron/jobs?profile=worker_alpha',
        timeoutMs: 60_000
      })
    )
  })

  it('omits the profile query when no profile is set (single-profile / legacy)', async () => {
    // _apiProfile defaults to null; the unscoped path is preserved so the
    // legacy single-profile install behaves identically to before the fix.
    await getCronJobs()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/cron/jobs',
        timeoutMs: 60_000
      })
    )
  })

  it('treats empty-string profiles as null (still unscoped)', async () => {
    // setApiRequestProfile coerces '' to null via `profile || null`.
    setApiRequestProfile('')

    await getCronJobs()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/cron/jobs',
        timeoutMs: 60_000
      })
    )
  })

  it('url-encodes profile names with reserved characters', async () => {
    setApiRequestProfile('team a/b')

    await getCronJobs()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/cron/jobs?profile=team%20a%2Fb',
        timeoutMs: 60_000
      })
    )
  })
})
