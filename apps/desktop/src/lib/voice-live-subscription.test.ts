// @vitest-environment node
import { afterEach, describe, expect, it, vi } from 'vitest'

import { hermesApi } from '@/hermes'
import { fetchVoiceLiveStatus, resolveVoiceConversationStart, VoiceLiveSession } from '@/lib/voice-live'

vi.mock('@/api/client', () => ({ profileScoped: () => ({ profile: 'test' }) }))
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
      const session = new VoiceLiveSession({ onDelegation, onTranscript, onClosed, onError: vi.fn() })
      await session.start([])
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
                content: [{ type: 'input_text', text: 'provider paraphrase is not the user transcript' }]
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

  it('waits for authoritative billing and never selects a fallback for subscription, invalid or unknown status', async () => {
    let resolveStatus!: (value: unknown) => void
    vi.mocked(hermesApi).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveStatus = resolve
        })
    )
    const pending = resolveVoiceConversationStart()
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
    resolveStatus({ ok: true, mode: 'gpt-live', auth: 'subscription', available: true })
    await expect(pending).resolves.toEqual({ mode: 'gpt-live', fallbackReason: null })

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
    for (const result of [unavailable, { ...unavailable, auth: 'invalid' }, null]) {
      vi.mocked(hermesApi).mockResolvedValueOnce(result)
      await expect(resolveVoiceConversationStart()).rejects.toThrow()
    }
    vi.mocked(hermesApi).mockResolvedValueOnce({ ...unavailable, auth: 'api' })
    await expect(resolveVoiceConversationStart()).resolves.toEqual({
      mode: 'chained',
      fallbackReason: unavailable.reason
    })
    vi.mocked(hermesApi).mockResolvedValueOnce({ ...unavailable, mode: 'chained' })
    await expect(resolveVoiceConversationStart()).resolves.toEqual({ mode: 'chained', fallbackReason: null })
  })
})
