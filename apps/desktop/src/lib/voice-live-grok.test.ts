// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayRoute, setPrimaryGateway } from '@/store/gateway'

import { GrokVoiceSession } from './voice-live-grok'

interface Recorded {
  method: string
  params: Record<string, unknown>
}

/** Records RPC requests; stands in for the JSON-RPC client the app speaks on. */
function fakeGateway() {
  const requests: Recorded[] = []
  let sink: ((event: { payload?: unknown; type: string }) => void) | null = null

  const gateway = {
    onAny: (handler: (event: { payload?: unknown; type: string }) => void) => {
      sink = handler

      return () => {
        sink = null
      }
    },
    request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      requests.push({ method, params })

      if (method === 'voice.grok.start') {
        return { started: true, live: { model: 'grok-voice-latest', voice: 'eve' } }
      }

      if (method === 'voice.grok.stop') {
        sink?.({ type: 'voice.grok.state', payload: { session_id: params.session_id, state: 'close_requested' } })
      }

      return {}
    })
  }

  return {
    emit: (type: string, payload?: unknown) => sink?.({ payload, type }),
    gateway,
    requests
  }
}

type DelegationHandler = (delegationId: string, prompt: string, context: string) => void

function makeHandlers(onDelegation: DelegationHandler) {
  return {
    onClosed: () => undefined,
    onDelegation,
    onState: () => undefined
  }
}

class FakeAudioContext {
  currentTime = 0
  sampleRate = 48_000
  state = 'running'

  createBuffer(): unknown {
    return { duration: 0, getChannelData: () => new Float32Array(0) }
  }

  createBufferSource(): unknown {
    return { connect: () => undefined, start: () => undefined, stop: vi.fn() }
  }

  createGain(): unknown {
    return { connect: () => undefined, gain: { value: 0 } }
  }

  createMediaStreamSource(): unknown {
    return { connect: () => undefined }
  }

  createScriptProcessor(): unknown {
    return { connect: () => undefined, disconnect: () => undefined }
  }

  destination = {}

  resume(): Promise<void> {
    return Promise.resolve()
  }

  close(): Promise<void> {
    return Promise.resolve()
  }
}

describe('GrokVoiceSession (renderer-submits design)', () => {
  beforeEach(() => {
    vi.stubGlobal('AudioContext', FakeAudioContext)
    Object.defineProperty(window, 'AudioContext', { configurable: true, value: FakeAudioContext })
    Object.defineProperty(window.navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: vi.fn() }] }) as unknown as MediaStream) }
    })
  })

  afterEach(() => {
    setPrimaryGateway(null)
    $activeGatewayRoute.set('default')
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('exposes prompt+context from a delegation event (the renderer is the single submitter)', async () => {
    const delegations: Array<{ context: string; delegationId: string; prompt: string }> = []
    const { emit, gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never)

    const voice = new GrokVoiceSession(makeHandlers((delegationId, prompt, context) => {
      delegations.push({ context, delegationId, prompt })
    }))

    voice.useSessionId('sid-1')
    await voice.start()

    emit('voice.grok.delegation', {
      context: 'User: what time is it',
      delegation_id: 'grok-abc',
      prompt: 'what time is it',
      session_id: 'sid-1'
    })

    // The event carries BOTH the prompt (turn text the renderer submits) and
    // the context (voice_context, model input only) — the backend never calls
    // prompt.submit on its behalf.
    expect(delegations).toEqual([
      { context: 'User: what time is it', delegationId: 'grok-abc', prompt: 'what time is it' }
    ])
    expect(requests.filter(r => r.method === 'prompt.submit')).toEqual([])
  })

  it('re-keys the bridge when the real chat session id appears after start', async () => {
    const { gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never)

    const voice = new GrokVoiceSession(makeHandlers(() => undefined))
    voice.useSessionId('grok-live-1789482317652-2qi1hr')
    await voice.start()

    // The delegation's submit minted the real chat session id — adopt it.
    voice.useSessionId('1d96782d')

    const rekey = requests.find(r => r.method === 'voice.grok.rekey')
    expect(rekey?.params.from_session_id).toBe('grok-live-1789482317652-2qi1hr')
    expect(rekey?.params.to_session_id).toBe('1d96782d')
  })

  it('speaks the settled reply through voice.grok.speak, keyed by the real id', () => {
    const { gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never)

    const voice = new GrokVoiceSession(makeHandlers(() => undefined))
    voice.useSessionId('1d96782d')
    voice.speak('It is noon.')

    const speak = requests.find(r => r.method === 'voice.grok.speak')
    expect(speak?.params.session_id).toBe('1d96782d')
    expect(speak?.params.text).toBe('It is noon.')
    // Empty text is a no-op, not an extra RPC.
    voice.speak('   ')
    expect(requests.filter(r => r.method === 'voice.grok.speak')).toHaveLength(1)
  })

  it('accepts old-id events while a fresh-draft rekey is in flight', async () => {
    const { emit, gateway } = fakeGateway()
    setPrimaryGateway(gateway as never)
    const onTranscript = vi.fn()
    const voice = new GrokVoiceSession({ ...makeHandlers(() => undefined), onTranscript })
    voice.useSessionId('synthetic-1')
    await voice.start()
    voice.useSessionId('real-1')

    emit('voice.grok.transcript', { session_id: 'synthetic-1', speaker: 'user', text: 'still routed' })
    expect(onTranscript).toHaveBeenCalledWith({ speaker: 'user', text: 'still routed' })
  })

  it('finishes, stops the backend bridge, and releases local audio on a fatal error', async () => {
    const { emit, gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never)
    const onClosed = vi.fn()
    const voice = new GrokVoiceSession({ ...makeHandlers(() => undefined), onClosed })
    await voice.start()

    emit('voice.grok.state', { session_id: voice.sessionId, state: 'error', reason: 'websocket failed' })
    expect(onClosed).toHaveBeenCalledWith('websocket failed')
    expect(requests.filter(request => request.method === 'voice.grok.stop')).toHaveLength(1)
  })

  it('stops an accepted backend start when microphone permission fails', async () => {
    const { gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never)
    Object.defineProperty(window.navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new Error('Permission denied')) }
    })
    const voice = new GrokVoiceSession(makeHandlers(() => undefined))

    await expect(voice.start()).rejects.toThrow('Permission denied')
    voice.close()

    await vi.waitFor(() => {
      expect(requests.filter(request => request.method === 'voice.grok.stop')).toHaveLength(1)
    })
  })

  it('rejects a backend not-started response before opening the microphone', async () => {
    const { gateway } = fakeGateway()
    gateway.request = vi.fn(async () => ({ started: false, reason: 'no_xai_credential' })) as never
    setPrimaryGateway(gateway as never)
    const voice = new GrokVoiceSession(makeHandlers(() => undefined))

    await expect(voice.start()).rejects.toThrow('no_xai_credential')
    expect(window.navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled()
  })

  it('pins the starting profile for the whole session', async () => {
    const { gateway, requests } = fakeGateway()
    setPrimaryGateway(gateway as never, 'secondary')
    $activeGatewayRoute.set('secondary')
    const voice = new GrokVoiceSession(makeHandlers(() => undefined))
    await voice.start()

    setPrimaryGateway(gateway as never, 'default')
    await voice.close()

    const voiceRequests = requests.filter(request => request.method.startsWith('voice.grok.'))
    expect(voiceRequests.length).toBeGreaterThan(0)

    for (const request of voiceRequests) {
      expect(request.params.profile).toBe('secondary')
    }
  })
})
