// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setPrimaryGateway } from '@/store/gateway'

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
    return { getChannelData: () => new Float32Array(0) }
  }

  createBufferSource(): unknown {
    return { connect: () => undefined, start: () => undefined }
  }

  createGain(): unknown {
    return { connect: () => undefined, gain: { value: 0 } }
  }

  createMediaStreamSource(): unknown {
    return { connect: () => undefined }
  }

  createScriptProcessor(): unknown {
    return { connect: () => undefined }
  }

  destination = {}

  resume(): Promise<void> {
    return Promise.resolve()
  }
}

describe('GrokVoiceSession (renderer-submits design)', () => {
  beforeEach(() => {
    vi.stubGlobal('AudioContext', FakeAudioContext)
    Object.defineProperty(window, 'AudioContext', { configurable: true, value: FakeAudioContext })
    Object.defineProperty(window.navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => ({}) as MediaStream) }
    })
  })

  afterEach(() => {
    setPrimaryGateway(null)
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
})
