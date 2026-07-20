import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS,
  AUDIO_TRANSCRIBE_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_TRANSCRIBE_MIN_REQUEST_TIMEOUT_MS,
  audioSpeakRequestTimeoutMs,
  audioTranscribeRequestTimeoutMs,
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
  listSidebarSessions,
  resetSidebarBatchCapability,
  speakText,
  transcribeAudio
} from './hermes'
import { refreshActiveProfile } from './store/profile'

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
    // scoping: recents on the caller's profile, cron + messaging cross-profile.
    expect(result.recents.sessions.map(s => s.id)).toEqual(['recent-1'])
    expect(result.recents.total).toBe(7)
    expect(result.recents.profile_totals).toEqual({ default: 7 })
    expect(result.cron.sessions.map(s => s.id)).toEqual(['cron-1'])
    expect(result.messaging.sessions.map(s => s.id)).toEqual(['msg-1'])

    const paths = api.mock.calls.map(call => (call[0] as { path: string }).path)
    expect(paths.filter(p => p.startsWith('/api/profiles/sessions/sidebar'))).toHaveLength(1)
    expect(paths.filter(p => p.startsWith('/api/profiles/sessions?'))).toHaveLength(3)
    expect(paths).toContainEqual(expect.stringContaining('profile=work'))
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

  it('bounds blocking TTS synthesis timeouts by text length', () => {
    expect(audioSpeakRequestTimeoutMs('short message')).toBe(AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(8_000))).toBe(280_000)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(100_000))).toBe(AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS)
  })

  it('uses an extended timeout for blocking TTS synthesis', async () => {
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
