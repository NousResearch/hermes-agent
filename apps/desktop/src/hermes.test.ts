import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getCronJobs,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getHermesConfig,
  getHermesConfigDefaults,
  getHermesConfigRecord,
  getHermesConfigSchema,
  getLogs,
  getProfiles,
  getSessionMessages,
  getStatus,
  listAllProfileSessions,
  listSessions
} from './hermes'

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

  it('uses a longer timeout for the profile list', async () => {
    api.mockResolvedValue({ profiles: [] })

    await getProfiles()

    expect(api).toHaveBeenCalledWith({
      path: '/api/profiles',
      timeoutMs: 60_000
    })
  })

  it('uses a longer timeout for boot aggregate calls', async () => {
    await getCronJobs()
    await getGlobalModelInfo()
    await getGlobalModelOptions()
    await getHermesConfig()
    await getHermesConfigRecord()
    await getHermesConfigDefaults()
    await getHermesConfigSchema()
    await getLogs({ file: 'gui', lines: 12 })
    await getStatus()

    expect(api).toHaveBeenCalledWith({
      path: '/api/cron/jobs',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/model/info',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/model/options',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/config',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/config/defaults',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/config/schema',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/logs?file=gui&lines=12',
      timeoutMs: 60_000
    })
    expect(api).toHaveBeenCalledWith({
      path: '/api/status',
      timeoutMs: 60_000
    })
  })

  it('tags cross-profile message reads for Electron routing and backend lookup', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu',
      profile: 'xiaoxuxu'
    })
  })
})
