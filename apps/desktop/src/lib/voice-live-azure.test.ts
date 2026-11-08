import { afterEach, beforeEach, describe, expect, it, type Mock, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import type { LiveTranscriptFragment } from './voice-live'
import { AZURE_LIVE_SOCKET_PATH, AZURE_PCM_SAMPLE_RATE, VoiceLiveAzureSession } from './voice-live-azure'

// The Azure transport dials the ticket-gated relay socket beside `/api/ws` on the
// session OWNER's (connection, profile) — the same routing chat uses — then speaks
// the relay protocol: session.start → synthetic session.started, PCM16 24 kHz
// input_audio_buffer.append out, response.audio.delta / transcript events /
// delegate_to_hermes function calls in.

class FakeWebSocket {
  static OPEN = 1
  static instances: FakeWebSocket[] = []

  readyState = 0
  sent: string[] = []
  closed = false
  onmessage: null | ((event: { data: unknown }) => void) = null
  onclose: null | (() => void) = null
  private readonly listeners = new Map<string, Set<() => void>>()

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }

  addEventListener(type: string, listener: () => void): void {
    const set = this.listeners.get(type) ?? new Set<() => void>()
    set.add(listener)
    this.listeners.set(type, set)
  }

  removeEventListener(type: string, listener: () => void): void {
    this.listeners.get(type)?.delete(listener)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    if (this.closed) {
      return
    }

    this.closed = true
    this.readyState = 3
    this.fire('close')
    this.onclose?.()
  }

  fire(type: string): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener()
    }
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN
    this.fire('open')
  }

  receive(event: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(event) })
  }

  sentJson(): Array<Record<string, unknown>> {
    return this.sent.map(frame => JSON.parse(frame) as Record<string, unknown>)
  }
}

interface FakeProcessor {
  onaudioprocess: null | ((event: { inputBuffer: { getChannelData: () => Float32Array } }) => void)
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
}

interface FakeBufferSource {
  buffer: null | { duration: number }
  onended: null | (() => void)
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []

  readonly destination = {}
  state = 'running'
  currentTime = 0
  processors: FakeProcessor[] = []
  sources: FakeBufferSource[] = []
  buffers: Array<{ channels: number; length: number; sampleRate: number }> = []

  constructor(readonly options?: { sampleRate?: number }) {
    this.sampleRate = options?.sampleRate ?? 48_000
    FakeAudioContext.instances.push(this)
  }

  sampleRate: number
  createMediaStreamSource() {
    return { connect: vi.fn(), disconnect: vi.fn() }
  }

  createScriptProcessor(): FakeProcessor {
    const processor: FakeProcessor = { connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null }
    this.processors.push(processor)

    return processor
  }

  createBuffer(channels: number, length: number, sampleRate: number) {
    this.buffers.push({ channels, length, sampleRate })

    return { duration: length / sampleRate, getChannelData: () => new Float32Array(length) }
  }

  createBufferSource(): FakeBufferSource {
    const source: FakeBufferSource = {
      buffer: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
      onended: null,
      start: vi.fn(),
      stop: vi.fn()
    }

    this.sources.push(source)

    return source
  }

  close() {
    return Promise.resolve()
  }
}

const localWsUrl = 'ws://127.0.0.1:5151/api/ws?token=local'
const remoteWsUrl = 'wss://gateway.example/api/ws?ticket=fresh'

let getConnection: ReturnType<typeof vi.fn>
let getConnectionFor: ReturnType<typeof vi.fn>
let getGatewayWsUrl: ReturnType<typeof vi.fn>
let getGatewayWsUrlFor: ReturnType<typeof vi.fn>
let track: { enabled: boolean; stop: ReturnType<typeof vi.fn> }

let handlers: {
  onClosed: Mock<(reason: string, usageSeconds: null | number) => void>
  onDelegation: Mock<(delegationId: string, context: LiveTranscriptFragment[]) => void>
  onError: Mock<(message: string, fatal: boolean) => void>
  onSpeakingChange: Mock<(speaking: boolean) => void>
  onTranscript: Mock<(fragment: LiveTranscriptFragment) => void>
}

function installBridge() {
  getConnection = vi.fn(async () => ({ authMode: 'token', baseUrl: 'http://127.0.0.1:5151', wsUrl: localWsUrl }))
  getConnectionFor = vi.fn(async () => ({ authMode: 'token', baseUrl: 'https://gateway.example', wsUrl: remoteWsUrl }))
  getGatewayWsUrl = vi.fn(async () => ({ ok: true, wsUrl: localWsUrl }))
  getGatewayWsUrlFor = vi.fn(async () => ({ ok: true, wsUrl: remoteWsUrl }))

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
  })
}

function installMedia() {
  track = { enabled: true, stop: vi.fn() }
  const stream = { getAudioTracks: () => [track], getTracks: () => [track] }

  Object.defineProperty(window.navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => stream) }
  })
  vi.stubGlobal('WebSocket', FakeWebSocket)
  vi.stubGlobal('AudioContext', FakeAudioContext)
}

function makeHandlers(): typeof handlers {
  return {
    onClosed: vi.fn<(reason: string, usageSeconds: null | number) => void>(),
    onDelegation: vi.fn<(delegationId: string, context: LiveTranscriptFragment[]) => void>(),
    onError: vi.fn<(message: string, fatal: boolean) => void>(),
    onSpeakingChange: vi.fn<(speaking: boolean) => void>(),
    onTranscript: vi.fn<(fragment: LiveTranscriptFragment) => void>()
  }
}

const history = [{ content: [{ text: 'earlier turn', type: 'input_text' as const }], role: 'user' as const, type: 'message' as const }]

/** Start a session against the fake socket; returns once `session.started` landed. */
async function startSession(session: VoiceLiveAzureSession): Promise<FakeWebSocket> {
  const pending = session.start(history)
  // The socket is constructed after the async URL resolution, not synchronously.
  await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
  const ws = FakeWebSocket.instances[0]

  ws.open()
  await vi.waitFor(() => expect(ws.sentJson().map(frame => frame.type)).toContain('session.start'))
  ws.receive({ session: { id: 'sess-azure-1' }, type: 'session.started' })
  await pending

  return ws
}

/** Wait for start() to construct its socket and return it (for failure-path tests). */
async function startedDial(session: VoiceLiveAzureSession): Promise<{ pending: Promise<void>; ws: FakeWebSocket }> {
  const pending = session.start(history)
  await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))

  return { pending, ws: FakeWebSocket.instances[0] }
}

beforeEach(() => {
  installBridge()
  installMedia()
  handlers = makeHandlers()
  FakeWebSocket.instances = []
  FakeAudioContext.instances = []
})

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('VoiceLiveAzureSession — relay routing', () => {
  it('dials the owner (connection, profile) relay socket, never the active scope or an SDP POST', async () => {
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const session = new VoiceLiveAzureSession(handlers, { connectionId: 'gw-bots', profile: 'bot-adam' })
    const ws = await startSession(session)

    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'gw-bots', profile: 'bot-adam' })
    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'gw-bots', profile: 'bot-adam' })
    expect(getConnection).not.toHaveBeenCalled()
    expect(getGatewayWsUrl).not.toHaveBeenCalled()

    const url = new URL(ws.url)

    expect(url.origin).toBe('wss://gateway.example')
    expect(url.pathname).toBe(AZURE_LIVE_SOCKET_PATH)
    expect(url.searchParams.get('profile')).toBe('bot-adam')
    // No WebRTC, no session POST: the only wire is the relay socket.
    expect(ws.sentJson()[0]).toMatchObject({ history, type: 'session.start' })
    expect(session.sessionId).toBe('sess-azure-1')
    expect(session.connected).toBe(true)
  })

  it('keeps the active scope verbatim for an ownerless chat', async () => {
    setApiRequestProfile('coder')

    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    expect(getConnection).toHaveBeenCalledWith('coder')
    expect(getConnectionFor).not.toHaveBeenCalled()

    const url = new URL(ws.url)

    expect(url.origin).toBe('ws://127.0.0.1:5151')
    expect(url.pathname).toBe(AZURE_LIVE_SOCKET_PATH)
    expect(url.searchParams.get('profile')).toBe('coder')
  })

  it('throws the recorder-mappable DOMException when mediaDevices is missing', async () => {
    Reflect.deleteProperty(window.navigator, 'mediaDevices')

    const session = new VoiceLiveAzureSession(handlers, null)

    await expect(session.start([])).rejects.toBeInstanceOf(DOMException)
    expect(FakeWebSocket.instances).toEqual([])
  })

  it('rejects start when the socket never opens, without a close notification', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const { pending, ws } = await startedDial(session)

    ws.close() // error/close before open
    await expect(pending).rejects.toThrow('Could not open the Azure live stream connection')
    expect(handlers.onClosed).not.toHaveBeenCalled()
  })

  it('rejects start when the relay answers error before session.started', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const { pending, ws } = await startedDial(session)

    ws.open()
    await vi.waitFor(() => expect(ws.sent.length).toBe(1))
    ws.receive({ error: { message: 'bad deployment' }, type: 'error' })
    await expect(pending).rejects.toThrow('bad deployment')
    expect(handlers.onError).toHaveBeenCalledWith('bad deployment', false)
  })

  it('surfaces the relay route’s top-level startup error without waiting for a timeout', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const { pending, ws } = await startedDial(session)

    ws.open()
    await vi.waitFor(() => expect(ws.sent.length).toBe(1))
    ws.receive({ message: 'missing Azure API key', type: 'error' })
    await expect(pending).rejects.toThrow('missing Azure API key')
    expect(handlers.onError).toHaveBeenCalledWith('missing Azure API key', false)
  })
})

describe('VoiceLiveAzureSession — PCM capture and playback', () => {
  it('streams mic audio as base64 PCM16 mono 24 kHz input_audio_buffer.append, resampled from the device rate', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    const capture = FakeAudioContext.instances[0]
    expect(capture.sampleRate).toBe(48_000)
    const processor = capture.processors[0]

    // 48 kHz → 24 kHz: every second input sample survives.
    processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array([0.5, 0, -0.5, 0]) } })

    const appends = ws.sentJson().filter(frame => frame.type === 'input_audio_buffer.append')
    expect(appends).toHaveLength(1)

    const pcm = new Int16Array([Math.round(0.5 * 32_768), Math.round(-0.5 * 32_768)])
    const expected = btoa(String.fromCharCode(...new Uint8Array(pcm.buffer)))

    expect(appends[0].audio).toBe(expected)
  })

  it('plays response.audio.delta through Web Audio at the pinned rate and tracks speaking', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    const pcm = new Int16Array([1_000, -1_000, 2_000])
    ws.receive({ delta: btoa(String.fromCharCode(...new Uint8Array(pcm.buffer))), type: 'response.audio.delta' })

    const playback = FakeAudioContext.instances.at(-1) as FakeAudioContext

    expect(playback.buffers).toEqual([{ channels: 1, length: 3, sampleRate: AZURE_PCM_SAMPLE_RATE }])
    expect(playback.sources).toHaveLength(1)
    expect(playback.sources[0].start).toHaveBeenCalledTimes(1)
    expect(handlers.onSpeakingChange).toHaveBeenLastCalledWith(true)

    playback.sources[0].onended?.()
    expect(handlers.onSpeakingChange).toHaveBeenLastCalledWith(false)
  })

  it('maps the Azure transcript events onto the shared fragment surface', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    ws.receive({ item: { transcript: 'what is the weather ' }, type: 'conversation.item.input_audio_transcription.completed' })
    ws.receive({ item: { transcript: 'in Paris?' }, type: 'conversation.item.input_audio_transcription.completed' })
    ws.receive({ delta: 'Let me check. ', type: 'response.audio_transcript.delta' })

    expect(handlers.onTranscript.mock.calls.map(([fragment]) => fragment)).toEqual([
      { endMs: expect.any(Number), speaker: 'user', startMs: expect.any(Number), text: 'what is the weather ' },
      { endMs: expect.any(Number), speaker: 'user', startMs: expect.any(Number), text: 'in Paris?' },
      { endMs: expect.any(Number), speaker: 'assistant', startMs: expect.any(Number), text: 'Let me check. ' }
    ])
  })
})

describe('VoiceLiveAzureSession — delegation and command surface', () => {
  it('maps response.function_call_arguments.done (delegate_to_hermes) to onDelegation with the transcript window', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    ws.receive({ item: { transcript: 'run the tests' }, type: 'conversation.item.input_audio_transcription.completed' })
    ws.receive({ arguments: '{}', call_id: 'call_42', name: 'delegate_to_hermes', type: 'response.function_call_arguments.done' })

    expect(handlers.onDelegation).toHaveBeenCalledTimes(1)
    const [delegationId, context] = handlers.onDelegation.mock.calls[0]

    expect(delegationId).toBe('call_42')
    expect(session.activeDelegationId).toBe('call_42')
    expect(context.map(fragment => `${fragment.speaker}:${fragment.text}`)).toEqual(['user:run the tests'])

    // Other function calls are not delegations.
    ws.receive({ arguments: '{}', call_id: 'call_43', name: 'something_else', type: 'response.function_call_arguments.done' })
    expect(handlers.onDelegation).toHaveBeenCalledTimes(1)
  })

  it('sends commentary appends, the commentary.complete seal, think and instruct as the existing messages', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    session.speak('call_42', 'One sentence. Another sentence.')
    session.completeCommentary('call_42')
    session.think('call_42', 'working on it')
    session.instruct('Speak like a pirate.')

    expect(ws.sentJson().slice(1)).toEqual([
      { content: 'One sentence. Another sentence.', delegation_id: 'call_42', event_id: expect.any(String), type: 'session.commentary.append' },
      { delegation_id: 'call_42', event_id: expect.any(String), type: 'session.commentary.complete' },
      { content: 'working on it', delegation_id: 'call_42', event_id: expect.any(String), type: 'session.thinking.append' },
      { content: 'Speak like a pirate.', delegation_id: null, event_id: expect.any(String), type: 'session.instructions.append' }
    ])
  })

  it('mutes the mic track and stops streaming while muted', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)
    const processor = FakeAudioContext.instances[0].processors[0]

    session.setMuted(true)
    expect(track.enabled).toBe(false)

    processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array([0.5, 0, -0.5, 0]) } })
    expect(ws.sentJson().filter(frame => frame.type === 'input_audio_buffer.append')).toHaveLength(0)

    session.setMuted(false)
    expect(track.enabled).toBe(true)

    processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array([0.5, 0, -0.5, 0]) } })
    expect(ws.sentJson().filter(frame => frame.type === 'input_audio_buffer.append')).toHaveLength(1)
  })
})

describe('VoiceLiveAzureSession — close', () => {
  it('closes gracefully: session.close out, session.closed seals with close_requested, exactly one onClosed', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    session.close()
    expect(ws.sentJson().at(-1)).toEqual({ type: 'session.close' })

    ws.receive({ type: 'session.closed' })

    expect(handlers.onClosed).toHaveBeenCalledTimes(1)
    expect(handlers.onClosed).toHaveBeenCalledWith('close_requested', null)
    expect(ws.closed).toBe(true)
    expect(track.stop).toHaveBeenCalled()

    session.close()
    ws.receive({ type: 'session.closed' })
    expect(handlers.onClosed).toHaveBeenCalledTimes(1)
  })

  it('reports a dropped socket as connection_lost', async () => {
    const session = new VoiceLiveAzureSession(handlers, null)
    const ws = await startSession(session)

    ws.close()

    expect(handlers.onClosed).toHaveBeenCalledWith('connection_lost', null)
  })
})
