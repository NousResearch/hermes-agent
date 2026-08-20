import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS,
  AUDIO_TRANSCRIBE_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_TRANSCRIBE_MIN_REQUEST_TIMEOUT_MS,
  audioSpeakRequestTimeoutMs,
  audioTranscribeRequestTimeoutMs,
  deleteProfile,
  deleteSession,
  getAllSessionMessages,
  getCronJobs,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getHermesConfig,
  getHermesConfigDefaults,
  getLatestSessionMessages,
  getOlderSessionMessages,
  getProfiles,
  getSession,
  getSessionMessages,
  getStatus,
  LATEST_SESSION_MESSAGES_LIMIT,
  listAllProfileSessions,
  listSessions,
  listSidebarSessions,
  pluginSocket,
  resetSidebarBatchCapability,
  setApiRequestConnection,
  setApiRequestProfile,
  speakText,
  transcribeAudio,
  triggerCronJob
} from './hermes'
import { refreshActiveProfile } from './store/profile'
import { $transcriptTailBySessionId, transcriptTailState } from './store/transcript-tail'

const emptySessionsResponse = {
  limit: 0,
  offset: 0,
  sessions: [],
  total: 0
}

describe('Hermes REST helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    resetSidebarBatchCapability()
    api = vi.fn().mockResolvedValue(emptySessionsResponse)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    setApiRequestConnection(null)
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

  it('batches the sidebar slices into a single request with per-slice limits + excludes', async () => {
    api.mockResolvedValue({ recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } })

    await listSidebarSessions({
      recentsProfile: 'work',
      recentsLimit: 30,
      recentsExclude: ['cron', 'tool'],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: ['cron', 'desktop']
    })

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path:
          '/api/profiles/sessions/sidebar?recents_profile=work&recents_limit=30&cron_limit=50' +
          '&messaging_limit=100&recents_exclude=cron%2Ctool&messaging_exclude=cron%2Cdesktop',
        timeoutMs: 60_000
      })
    )
  })

  it('routes session, profile, and model reads through the active registry source', async () => {
    api.mockImplementation(async ({ path }: { path: string }) =>
      path.startsWith('/api/profiles/sessions/sidebar')
        ? { recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } }
        : emptySessionsResponse
    )
    setApiRequestConnection('personal')

    await listSessions()
    await listAllProfileSessions()
    await listSidebarSessions({
      recentsProfile: 'default',
      recentsLimit: 20,
      recentsExclude: [],
      cronLimit: 20,
      messagingLimit: 20,
      messagingExclude: []
    })
    await getProfiles()
    await getGlobalModelInfo()

    for (const call of api.mock.calls) {
      expect(call[0]).toEqual(expect.objectContaining({ connectionId: 'personal' }))
    }
  })

  it('keeps an explicit This device source on REST requests', async () => {
    setApiRequestConnection('local')

    await getProfiles()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'local', path: '/api/profiles' }))
  })

  it('preserves ambient and explicit-local ownership for session and profile requests', async () => {
    setApiRequestConnection('remote-a')

    await getSession('ambient-session')
    await getSessionMessages('ambient-session')
    await deleteSession('ambient-session')

    for (const call of api.mock.calls) {
      expect(call[0]).toEqual(expect.objectContaining({ connectionId: 'remote-a' }))
    }

    api.mockClear()
    const localScope = { connectionId: 'local', profile: 'worker' }

    await getSession('local-session', localScope)
    await getSessionMessages('local-session', localScope)
    await deleteSession('local-session', localScope)
    await deleteProfile('worker', localScope)

    for (const call of api.mock.calls) {
      expect(call[0]).toEqual(expect.objectContaining({ connectionId: 'local', profile: 'worker' }))
    }
  })

  it('defaults missing sidebar slices to empty session arrays', async () => {
    api.mockResolvedValue({})

    const result = await listSidebarSessions({
      recentsProfile: 'all',
      recentsLimit: 20,
      recentsExclude: [],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: []
    })

    expect(result.recents.sessions).toEqual([])
    expect(result.cron.sessions).toEqual([])
    expect(result.messaging.sessions).toEqual([])
  })

  it('falls back to the per-slice endpoint when the batched route 404s on an older backend', async () => {
    const row = (id: string) => ({ id, title: id, profile: 'default' })

    api.mockImplementation(({ path }: { path: string }) => {
      if (path.startsWith('/api/profiles/sessions/sidebar')) {
        // The exact skew failure: Electron surfaces the backend catch-all.
        return Promise.reject(
          new Error(
            'Error invoking remote method \'hermes:api\': Error: 404: {"detail":"No such API endpoint: /api/profiles/sessions/sidebar"}'
          )
        )
      }

      if (path.includes('source=cron')) {
        return Promise.resolve({ ...emptySessionsResponse, sessions: [row('cron-1')], total: 1 })
      }

      if (path.includes('exclude_sources=cron%2Cdesktop')) {
        return Promise.resolve({ ...emptySessionsResponse, sessions: [row('msg-1')], total: 1 })
      }

      return Promise.resolve({
        ...emptySessionsResponse,
        sessions: [row('recent-1')],
        total: 7,
        profile_totals: { default: 7 }
      })
    })

    const result = await listSidebarSessions({
      recentsProfile: 'work',
      recentsLimit: 30,
      recentsExclude: ['cron', 'tool'],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: ['cron', 'desktop']
    })

    // Slices reassembled from the legacy per-slice route with the same
    // scoping: every section follows the caller's profile.
    expect(result.recents.sessions.map(s => s.id)).toEqual(['recent-1'])
    // One row back against a 30-row window: the profile is fully loaded, so
    // the legacy path must not claim there's another page.
    expect(result.recents.profiles_truncated).toEqual({ default: false })
    expect(result.cron.sessions.map(s => s.id)).toEqual(['cron-1'])
    expect(result.messaging.sessions.map(s => s.id)).toEqual(['msg-1'])

    const paths = api.mock.calls.map(call => (call[0] as { path: string }).path)
    expect(paths.filter(p => p.startsWith('/api/profiles/sessions/sidebar'))).toHaveLength(1)
    expect(paths.filter(p => p.startsWith('/api/profiles/sessions?'))).toHaveLength(3)
    expect(
      paths.filter(path => path.startsWith('/api/profiles/sessions?') && path.includes('profile=work'))
    ).toHaveLength(3)
    expect(paths.some(path => path.includes('profile=all'))).toBe(false)
    expect(paths).toContainEqual(expect.stringContaining('source=cron'))
    expect(paths).toContainEqual(expect.stringContaining('exclude_sources=cron%2Ctool'))
  })

  it('remembers endpoint-missing and skips re-probing the batched route on later refreshes', async () => {
    api.mockImplementation(({ path }: { path: string }) =>
      path.startsWith('/api/profiles/sessions/sidebar')
        ? Promise.reject(new Error('404: {"detail":"No such API endpoint: /api/profiles/sessions/sidebar"}'))
        : Promise.resolve(emptySessionsResponse)
    )

    const req = {
      recentsProfile: 'all' as const,
      recentsLimit: 20,
      recentsExclude: [],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: []
    }

    await listSidebarSessions(req)
    await listSidebarSessions(req)

    const batchedProbes = api.mock.calls.filter(call =>
      (call[0] as { path: string }).path.startsWith('/api/profiles/sessions/sidebar')
    )

    // First refresh probes once and learns; the second goes straight to the
    // per-slice route (3 calls each refresh, no repeated dead probe).
    expect(batchedProbes).toHaveLength(1)
    expect(api.mock.calls.length).toBe(1 + 3 + 3)
  })

  it('re-probes the batched route after a gateway switch resets the capability flag', async () => {
    api.mockImplementation(({ path }: { path: string }) =>
      path.startsWith('/api/profiles/sessions/sidebar')
        ? Promise.reject(new Error('404: {"detail":"No such API endpoint: /api/profiles/sessions/sidebar"}'))
        : Promise.resolve(emptySessionsResponse)
    )

    const req = {
      recentsProfile: 'all' as const,
      recentsLimit: 20,
      recentsExclude: [],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: []
    }

    await listSidebarSessions(req)
    // Soft gateway switch: the next backend may support the batched route.
    resetSidebarBatchCapability()
    api.mockResolvedValue({ recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } })

    const result = await listSidebarSessions(req)

    const batchedProbes = api.mock.calls.filter(call =>
      (call[0] as { path: string }).path.startsWith('/api/profiles/sessions/sidebar')
    )

    expect(batchedProbes).toHaveLength(2)
    expect(result.recents.sessions).toEqual([])
  })

  it('does NOT fall back on transient failures — only endpoint-missing shapes trigger legacy mode', async () => {
    api.mockRejectedValue(new Error('Request timed out after 60000ms'))

    await expect(
      listSidebarSessions({
        recentsProfile: 'all',
        recentsLimit: 20,
        recentsExclude: [],
        cronLimit: 50,
        messagingLimit: 100,
        messagingExclude: []
      })
    ).rejects.toThrow('timed out')

    // One call: the batched probe. No legacy fan-out, no sticky degradation.
    expect(api).toHaveBeenCalledTimes(1)

    // And the next refresh still uses the batched route.
    api.mockResolvedValue({ recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } })
    await listSidebarSessions({
      recentsProfile: 'all',
      recentsLimit: 20,
      recentsExclude: [],
      cronLimit: 50,
      messagingLimit: 100,
      messagingExclude: []
    })

    expect((api.mock.calls[1][0] as { path: string }).path).toMatch(/^\/api\/profiles\/sessions\/sidebar\?/)
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

  it('waits for synchronous cron triggers as a long-running operation', async () => {
    api.mockResolvedValue({ id: 'job-1' })

    await triggerCronJob('job-1')

    const request = api.mock.calls[0]?.[0]
    expect(request).toEqual(
      expect.objectContaining({
        path: '/api/cron/jobs/job-1/trigger',
        method: 'POST'
      })
    )
    expect(request.timeoutMs).toBeGreaterThanOrEqual(60 * 60 * 1000)
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

  it('pins session metadata and transcripts to an explicit connection scope', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })
    const scope = { connectionId: 'source-a', profile: 'backend-default' }

    await getSession('session-1', scope)
    await getSessionMessages('session-1', scope)

    expect(api).toHaveBeenNthCalledWith(1, {
      connectionId: 'source-a',
      path: '/api/sessions/session-1?profile=backend-default',
      profile: 'backend-default'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      connectionId: 'source-a',
      path: '/api/sessions/session-1/messages?profile=backend-default',
      profile: 'backend-default'
    })
  })

  it('scopes profile deletion and rejects default before Electron dispatch', async () => {
    api.mockResolvedValue({ ok: true, path: '/profiles/worker' })

    await deleteProfile('backend-worker', { connectionId: 'source-a', profile: 'backend-worker' })
    await expect(
      deleteProfile('worker', { connectionId: 'source-a', profile: 'default' })
    ).rejects.toThrow(/default profile cannot be deleted/i)

    expect(api).toHaveBeenCalledOnce()
    expect(api).toHaveBeenCalledWith({
      connectionId: 'source-a',
      method: 'DELETE',
      path: '/api/profiles/backend-worker',
      profile: 'backend-worker'
    })
  })

  it('hydrates the latest transcript with a small tail page (120, latest, compacted rows included)', async () => {
    api.mockResolvedValue({
      messages: [],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 0 },
      session_id: 'session-1'
    })

    await getLatestSessionMessages('session-1', 'xiaoxuxu')

    expect(LATEST_SESSION_MESSAGES_LIMIT).toBe(120)
    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu&limit=120&order=latest&include_compacted=true',
      profile: 'xiaoxuxu'
    })
  })

  it('records tail truncation state under the requested and resolved session ids', async () => {
    $transcriptTailBySessionId.set({})
    api.mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => ({ content: `m${index}`, id: index, role: 'user' })),
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 },
      session_id: 'resolved-1'
    })

    await getLatestSessionMessages('prefix-1')

    expect(transcriptTailState('prefix-1')).toMatchObject({ nextOffset: 120, possiblyTruncated: true })
    expect(transcriptTailState('resolved-1')).toMatchObject({ nextOffset: 120, possiblyTruncated: true })
  })

  it('requests older pages backwards from the newest message', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getOlderSessionMessages('session-1', 'xiaoxuxu', 240)

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu&limit=120&offset=240&order=latest&include_compacted=true',
      profile: 'xiaoxuxu'
    })
  })

  it('passes bounded transcript pagination through to the backend', async () => {
    api.mockResolvedValue({ messages: [], session_id: 'session-1' })

    await getSessionMessages('session-1', 'xiaoxuxu', {
      limit: 500,
      offset: 1000,
      order: 'latest'
    })

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu&limit=500&offset=1000&order=latest',
      profile: 'xiaoxuxu'
    })
  })

  it('loads complete transcripts through bounded oldest-first pages', async () => {
    api
      .mockResolvedValueOnce({
        messages: [{ id: 1 }, { id: 2 }],
        session_id: 'session-1',
        pagination: { limit: 2, offset: 0, order: 'oldest', returned: 2 }
      })
      .mockResolvedValueOnce({
        messages: [{ id: 3 }],
        session_id: 'session-1',
        pagination: { limit: 2, offset: 2, order: 'oldest', returned: 1 }
      })

    const result = await getAllSessionMessages('session-1', 'xiaoxuxu')

    expect(result.messages).toEqual([{ id: 1 }, { id: 2 }, { id: 3 }])
    expect(api).toHaveBeenNthCalledWith(1, {
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu&limit=500&offset=0&order=oldest&include_compacted=true',
      profile: 'xiaoxuxu'
    })
    expect(api).toHaveBeenNthCalledWith(2, {
      path: '/api/sessions/session-1/messages?profile=xiaoxuxu&limit=500&offset=2&order=oldest&include_compacted=true',
      profile: 'xiaoxuxu'
    })
  })

  it('stops complete transcript loads before Desktop memory becomes unbounded', async () => {
    api.mockResolvedValueOnce({
      messages: [{ id: 1, content: 'large transcript page' }],
      session_id: 'session-1',
      pagination: { limit: 1, offset: 0, order: 'oldest', returned: 1 }
    })

    await expect(getAllSessionMessages('session-1', null, { maxJsonChars: 1 })).rejects.toThrow(
      'Desktop safe-load limit'
    )
  })

  it('bounds blocking TTS synthesis timeouts by text length', () => {
    expect(audioSpeakRequestTimeoutMs('short message')).toBe(AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(8_000))).toBe(280_000)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(100_000))).toBe(AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS)
  })

  it('routes blocking TTS synthesis through the active profile backend', async () => {
    setApiRequestProfile('rhaegal')
    api.mockResolvedValueOnce({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    await expect(speakText('Read this aloud')).resolves.toEqual({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    expect(api).toHaveBeenCalledWith({
      body: { text: 'Read this aloud' },
      method: 'POST',
      path: '/api/audio/speak',
      profile: 'rhaegal',
      timeoutMs: AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS
    })
  })

  it('bounds blocking transcription timeouts by payload length', () => {
    expect(audioTranscribeRequestTimeoutMs('data:audio/webm;base64,AA==')).toBe(AUDIO_TRANSCRIBE_MIN_REQUEST_TIMEOUT_MS)
    expect(audioTranscribeRequestTimeoutMs('x'.repeat(3_000_000))).toBe(300_000)
    expect(audioTranscribeRequestTimeoutMs('x'.repeat(9_000_000))).toBe(AUDIO_TRANSCRIBE_MAX_REQUEST_TIMEOUT_MS)
  })

  it('uses an extended timeout for blocking transcription', async () => {
    api.mockResolvedValueOnce({
      ok: true,
      provider: 'openai',
      text: 'transcribed text'
    })

    await expect(transcribeAudio('data:audio/webm;base64,AA==', 'audio/webm')).resolves.toEqual({
      ok: true,
      provider: 'openai',
      text: 'transcribed text'
    })

    expect(api).toHaveBeenCalledWith({
      body: { data_url: 'data:audio/webm;base64,AA==', mime_type: 'audio/webm' },
      method: 'POST',
      path: '/api/audio/transcribe',
      timeoutMs: AUDIO_TRANSCRIBE_MIN_REQUEST_TIMEOUT_MS
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

describe('pluginSocket', () => {
  let getConnection: ReturnType<typeof vi.fn>

  beforeEach(() => {
    getConnection = vi.fn().mockResolvedValue(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: vi.fn(), getConnection }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
    setApiRequestProfile(null)
  })

  it('scopes the connection to the active profile, like pluginRest', async () => {
    setApiRequestProfile('work')

    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalled())
    expect(getConnection).toHaveBeenCalledWith('work')

    dispose()
  })

  it('passes null when no profile is scoped (single-profile / primary)', async () => {
    const dispose = pluginSocket('kanban', '/events', () => {})

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalled())
    expect(getConnection).toHaveBeenCalledWith(null)

    dispose()
  })
})
