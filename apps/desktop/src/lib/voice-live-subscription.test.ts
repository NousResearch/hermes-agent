// @vitest-environment node
import { afterEach, describe, expect, it, vi } from 'vitest'

import { hermesApi } from '@/hermes'
import { fetchVoiceLiveStatus, resolveVoiceConversationStart, VoiceLiveSession } from '@/lib/voice-live'

const { ownerScoped } = vi.hoisted(() => ({
  ownerScoped: vi.fn((owner?: unknown) => ({ owner }))
}))

vi.mock('@/api/client', () => ({
  ownerScoped,
  profileScoped: () => ({ profile: 'test' })
}))
vi.mock('@/hermes', () => ({ hermesApi: vi.fn() }))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.resetAllMocks()
  vi.restoreAllMocks()
})

function media() {
  const sent: Array<Record<string, unknown>> = []

  const events = Object.assign(new EventTarget(), {
    readyState: 'open',
    close: vi.fn(),
    send: (raw: string) => sent.push(JSON.parse(raw))
  })

  const track = { enabled: true, stop: vi.fn() }
  vi.stubGlobal('navigator', {
    mediaDevices: {
      getUserMedia: async () => ({ getAudioTracks: () => [track], getTracks: () => [track] })
    }
  })
  vi.stubGlobal(
    'Audio',
    class {
      autoplay = false
      srcObject = null
      pause() {}
    }
  )
  vi.stubGlobal(
    'RTCPeerConnection',
    class extends EventTarget {
      iceGatheringState = 'complete'
      localDescription = { sdp: 'v=0 offer', type: 'offer' }
      addTrack() {}
      close() {}
      createDataChannel() {
        return events
      }
      async createOffer() {
        return this.localDescription
      }
      async setLocalDescription() {}
      async setRemoteDescription() {}
    }
  )

  return {
    events,
    sent,
    track,
    receive: (event: unknown) => events.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
  }
}

describe('native Live explicit billing', () => {
  it.each(['api', 'subscription'] as const)(
    'keeps %s voice events and Hermes replies on the selected protocol',
    async auth => {
      const wire = media()
      const onDelegation = vi.fn()
      const onTranscript = vi.fn()
      const onClosed = vi.fn()
      vi.mocked(hermesApi).mockResolvedValueOnce({
        ok: true,
        auth,
        session: { id: 'rtc_fixture' },
        transport: { type: 'webrtc', sdp: 'v=0 answer' }
      })

      const session = new VoiceLiveSession(
        { onDelegation, onTranscript, onClosed, onError: vi.fn() },
        null,
        auth
      )

      await session.start([])
      expect(vi.mocked(hermesApi).mock.calls[0][0].body).toMatchObject({ expected_auth: auth })
      wire.receive({ type: 'session.started', session: { id: 'rtc_fixture' } })
      expect(session.connected).toBe(true)

      const delegation =
        auth === 'subscription'
          ? {
              type: 'delegation.created',
              item: {
                id: 'provider-delegation',
                type: 'delegation',
                target: 'client',
                content: [{ type: 'input_text', text: 'Read the project status.' }]
              }
            }
          : {
              type: 'session.delegation.created',
              delegation: { id: 'provider-delegation', type: 'delegation', target: 'client' }
            }

      wire.receive(
        auth === 'subscription'
          ? { type: 'input_transcript.added', item: { text: 'Read the project status.' } }
          : { type: 'session.input_transcript.delta', delta: 'Read the project status.', start_ms: 1, end_ms: 5 }
      )
      wire.receive(delegation)
      wire.receive(delegation)
      expect(onDelegation).toHaveBeenCalledTimes(1)
      expect(onDelegation.mock.calls[0][0]).toBe('provider-delegation')
      expect(onDelegation.mock.calls[0][1].map((part: { text: string }) => part.text)).toEqual([
        'Read the project status.'
      ])
      expect(onTranscript).toHaveBeenCalledTimes(1)

      if (auth === 'subscription') {
        wire.receive({ type: 'output_transcript.added', item: { text: 'I will check.' } })
        expect(onTranscript).toHaveBeenLastCalledWith(
          expect.objectContaining({ speaker: 'assistant', text: 'I will check.' })
        )
        vi.spyOn(performance, 'now').mockReturnValue(performance.now() + 6 * 60_000)
        wire.receive({ type: 'input_transcript.added', item: { text: 'Use only this fresh request.' } })
        expect(session.contextWindow().map(part => part.text)).toEqual(['Use only this fresh request.'])
        expect(session.contextWindow()[0].timestampSource).toBe('arrival')
      }

      const answer = 'Project ready. ' + '✅漢字'.repeat(180)
      session.speak('provider-delegation', answer)
      session.think('provider-delegation', 'Hermes is checking.')
      session.instruct('Speak briefly.')
      const commandCount = wire.sent.length
      session.setMuted(true)
      expect(wire.track.enabled).toBe(false)

      if (auth === 'subscription') {
        const spoken = wire.sent.filter(event => event.channel === 'speakable')
        const contents = spoken.map(event => (event.content as Array<{ text: string }>)[0].text)
        expect(contents.join('')).toBe(answer)
        expect(contents.every(text => new TextEncoder().encode(text).length <= 500)).toBe(true)
        expect(
          spoken.every(
            event => event.type === 'delegation.context.append' && event.delegation_item_id === 'provider-delegation'
          )
        ).toBe(true)
        expect(wire.sent.at(-1)).toMatchObject({ type: 'session.context.append', channel: 'commentary' })
        expect(wire.sent).toHaveLength(commandCount)
      } else {
        expect(wire.sent[0]).toMatchObject({ type: 'session.commentary.append', delegation_id: 'provider-delegation' })
        expect(wire.sent.at(-1)?.type).toBe('session.input_audio.mute')
      }

      wire.receive({ type: 'session.closed', reason: 'closed' })
      expect(onClosed).toHaveBeenCalledOnce()
      expect(wire.track.stop).toHaveBeenCalledOnce()
    }
  )

  it('expires subscription transcript context during silence before a delayed delegation', async () => {
    const wire = media()
    const onDelegation = vi.fn()
    const clock = vi.spyOn(performance, 'now').mockReturnValue(1_000)

    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      auth: 'subscription',
      session: { id: 'rtc_fixture' },
      transport: { type: 'webrtc', sdp: 'v=0 answer' }
    })

    const session = new VoiceLiveSession({ onDelegation, onClosed: vi.fn(), onError: vi.fn() })

    await session.start([])
    wire.receive({ type: 'input_transcript.added', item: { text: 'An old request.' } })
    expect(session.contextWindow().map(part => part.text)).toEqual(['An old request.'])
    clock.mockReturnValue(1_000 + 6 * 60_000)
    wire.receive({
      type: 'delegation.created',
      item: { id: 'delayed-delegation', type: 'delegation', target: 'client' }
    })
    expect(onDelegation).toHaveBeenCalledWith('delayed-delegation', [])
    wire.receive({ type: 'session.closed', reason: 'closed' })
  })

  it('uses delegation content when transcript events are absent', async () => {
    const wire = media()
    const onDelegation = vi.fn()
    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      auth: 'subscription',
      session: { id: 'rtc_fixture' },
      transport: { type: 'webrtc', sdp: 'v=0 answer' }
    })

    const session = new VoiceLiveSession(
      { onDelegation, onClosed: vi.fn(), onError: vi.fn() },
      null,
      'subscription'
    )

    await session.start([])
    wire.receive({
      type: 'delegation.created',
      item: {
        id: 'content-delegation',
        type: 'delegation',
        target: 'client',
        content: [{ type: 'input_text', text: 'Use the provider request.' }]
      }
    })

    expect(onDelegation).toHaveBeenCalledWith(
      'content-delegation',
      expect.arrayContaining([expect.objectContaining({ speaker: 'user', text: 'Use the provider request.' })])
    )
  })

  it('waits for authoritative billing and never selects a fallback for subscription, invalid or unknown status', async () => {
    let resolveStatus!: (value: unknown) => void
    vi.mocked(hermesApi).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveStatus = resolve
        })
    )
    const owner = { connectionId: 'remote', profile: 'bot' }
    const pending = resolveVoiceConversationStart(owner)
    let settled = false
    void pending.then(
      () => {
        settled = true
      },
      () => {
        settled = true
      }
    )
    await Promise.resolve()
    expect(settled).toBe(false)
    resolveStatus({
      ok: true,
      mode: 'gpt-live',
      auth: 'subscription',
      available: true,
      reason: null,
      model: 'gpt-live-1-codex',
      voice: 'cove'
    })
    await expect(pending).resolves.toEqual({ mode: 'gpt-live', auth: 'subscription', fallbackReason: null })
    expect(ownerScoped).toHaveBeenCalledWith(owner)

    const unavailable = {
      ok: true,
      mode: 'gpt-live',
      auth: 'subscription',
      available: false,
      reason: 'Codex sign-in needed',
      model: 'gpt-live-1-codex',
      voice: 'cove'
    }

    vi.mocked(hermesApi).mockResolvedValueOnce(unavailable)
    expect((await fetchVoiceLiveStatus())?.auth).toBe('subscription')

    for (const result of [unavailable, { ...unavailable, auth: 'invalid' }]) {
      vi.mocked(hermesApi).mockResolvedValueOnce(result)
      await expect(resolveVoiceConversationStart(owner)).rejects.toThrow()
    }

    vi.mocked(hermesApi).mockRejectedValueOnce(new Error('404: {"detail":"Not Found"}'))
    await expect(resolveVoiceConversationStart(owner)).resolves.toEqual({
      mode: 'chained',
      auth: null,
      fallbackReason: null
    })
    vi.mocked(hermesApi).mockResolvedValueOnce({ ...unavailable, auth: 'api' })
    await expect(resolveVoiceConversationStart(owner)).resolves.toEqual({
      mode: 'chained',
      auth: 'api',
      fallbackReason: unavailable.reason
    })
    vi.mocked(hermesApi).mockResolvedValueOnce({ ...unavailable, mode: 'chained' })
    await expect(resolveVoiceConversationStart(owner)).resolves.toEqual({
      mode: 'chained',
      auth: 'subscription',
      fallbackReason: null
    })
  })

  it.each([
    '404: {"detail":"Not Found"}',
    '404: {"detail":"No such API endpoint: /api/audio/voice-live/status"}',
    "Error invoking remote method 'hermes:api': Error: 404: {\"detail\":\"No such API endpoint: /api/audio/voice-live/status\"}",
    'Expected JSON from http://127.0.0.1:8642/api/audio/voice-live/status?profile=bot but got HTML (status 200). The endpoint is likely missing on the Hermes backend.',
    "Error invoking remote method 'hermes:api': Error: Expected JSON from http://127.0.0.1:8642/api/audio/voice-live/status but got HTML (status 200). The endpoint is likely missing on the Hermes backend."
  ])('falls back for a legacy backend missing the voice-live status route: %s', async message => {
    const owner = { connectionId: 'remote', profile: 'bot' }

    vi.mocked(hermesApi).mockRejectedValueOnce(new Error(message))
    await expect(resolveVoiceConversationStart(owner)).resolves.toEqual({
      mode: 'chained',
      auth: null,
      fallbackReason: null
    })
  })

  it.each([
    `404: {"detail":"Profile 'bot' does not exist"}`,
    '404: {"detail":"Session not found"}',
    '404: {"detail":"No such API endpoint: /api/audio/transcribe"}',
    'Expected JSON from http://127.0.0.1:8642/api/audio/transcribe but got HTML (status 200). The endpoint is likely missing on the Hermes backend.',
    'Expected JSON from http://127.0.0.1:8642/api/audio/transcribe?next=/api/audio/voice-live/status but got HTML (status 200). The endpoint is likely missing on the Hermes backend.',
    '503: {"detail":"No such API endpoint: /api/audio/voice-live/status"}'
  ])('fails closed for profile, resource, unrelated-route, and non-404 status errors: %s', async message => {
    vi.mocked(hermesApi).mockRejectedValueOnce(new Error(message))

    await expect(resolveVoiceConversationStart({ connectionId: 'remote', profile: 'bot' })).rejects.toThrow(message)
  })

  it.each([
    ['missing mode', { ok: true, auth: 'subscription', available: true, model: 'gpt-live-1-codex', voice: 'cove' }],
    [
      'unknown mode',
      { ok: true, mode: 'unknown', auth: 'subscription', available: true, model: 'gpt-live-1-codex', voice: 'cove' }
    ],
    ['missing auth', { ok: true, mode: 'gpt-live', available: true, model: 'gpt-live-1-codex', voice: 'cove' }],
    [
      'unknown auth',
      { ok: true, mode: 'gpt-live', auth: 'unknown', available: true, model: 'gpt-live-1-codex', voice: 'cove' }
    ],
    [
      'missing available',
      { ok: true, mode: 'gpt-live', auth: 'subscription', model: 'gpt-live-1-codex', voice: 'cove' }
    ],
    [
      'non-boolean available',
      { ok: true, mode: 'gpt-live', auth: 'subscription', available: 'yes', model: 'gpt-live-1-codex', voice: 'cove' }
    ],
    ['missing model', { ok: true, mode: 'gpt-live', auth: 'subscription', available: true, voice: 'cove' }],
    [
      'invalid model',
      { ok: true, mode: 'gpt-live', auth: 'subscription', available: true, model: '', voice: 'cove' }
    ],
    [
      'missing voice',
      { ok: true, mode: 'gpt-live', auth: 'subscription', available: true, model: 'gpt-live-1-codex' }
    ],
    [
      'invalid voice',
      { ok: true, mode: 'gpt-live', auth: 'subscription', available: true, model: 'gpt-live-1-codex', voice: ' ' }
    ],
    [
      'missing reason',
      { ok: true, mode: 'gpt-live', auth: 'subscription', available: true, model: 'gpt-live-1-codex', voice: 'cove' }
    ],
    [
      'invalid numeric reason',
      {
        ok: true,
        mode: 'gpt-live',
        auth: 'subscription',
        available: true,
        reason: 1,
        model: 'gpt-live-1-codex',
        voice: 'cove'
      }
    ],
    [
      'invalid object reason',
      {
        ok: true,
        mode: 'gpt-live',
        auth: 'subscription',
        available: true,
        reason: {},
        model: 'gpt-live-1-codex',
        voice: 'cove'
      }
    ],
    [
      'invalid array reason',
      {
        ok: true,
        mode: 'gpt-live',
        auth: 'subscription',
        available: true,
        reason: [],
        model: 'gpt-live-1-codex',
        voice: 'cove'
      }
    ],
    [
      'invalid boolean reason',
      {
        ok: true,
        mode: 'gpt-live',
        auth: 'subscription',
        available: true,
        reason: true,
        model: 'gpt-live-1-codex',
        voice: 'cove'
      }
    ],
    ['reviewer example', { ok: true, mode: 'unknown', auth: 'subscription', available: true }],
    ['unsuccessful response', { ok: false }]
  ])('fails closed for a malformed status response: %s', async (_label, response) => {
    vi.mocked(hermesApi).mockResolvedValueOnce(response)

    await expect(resolveVoiceConversationStart()).rejects.toThrow(/status/i)
  })

  it('accepts an auth-less legacy status only as API mode when every legacy field is valid', async () => {
    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      mode: 'gpt-live',
      available: true,
      reason: null,
      model: 'gpt-live-1',
      voice: 'marin'
    })

    await expect(resolveVoiceConversationStart()).resolves.toEqual({
      mode: 'gpt-live',
      auth: 'api',
      fallbackReason: null
    })
  })

  it('accepts an auth-less session response only for legacy API mode', async () => {
    media()
    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      session: { id: 'rtc_fixture' },
      transport: { type: 'webrtc', sdp: 'v=0 answer' }
    })

    const apiSession = new VoiceLiveSession(
      { onDelegation: vi.fn(), onClosed: vi.fn(), onError: vi.fn() },
      null,
      'api'
    )

    await expect(apiSession.start([])).resolves.toBeUndefined()

    media()
    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      session: { id: 'rtc_fixture' },
      transport: { type: 'webrtc', sdp: 'v=0 answer' }
    })

    const subscriptionSession = new VoiceLiveSession(
      { onDelegation: vi.fn(), onClosed: vi.fn(), onError: vi.fn() },
      null,
      'subscription'
    )

    await expect(subscriptionSession.start([])).rejects.toThrow(/auth.*changed/i)
  })

  it('rejects when the session auth differs from the preflight auth', async () => {
    const wire = media()
    vi.mocked(hermesApi).mockResolvedValueOnce({
      ok: true,
      auth: 'api',
      session: { id: 'rtc_fixture' },
      transport: { type: 'webrtc', sdp: 'v=0 answer' }
    })

    const session = new VoiceLiveSession(
      { onDelegation: vi.fn(), onClosed: vi.fn(), onError: vi.fn() },
      null,
      'subscription'
    )

    await expect(session.start([])).rejects.toThrow(/auth.*changed/i)
    wire.receive({ type: 'session.closed', reason: 'closed' })
    expect(wire.track.stop).toHaveBeenCalledOnce()
  })
})
