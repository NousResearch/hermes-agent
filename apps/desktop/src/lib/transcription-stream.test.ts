import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { openTranscriptionStream, resolveTranscriptionStreamUrl } from './transcription-stream'

// The streaming STT client is provider-neutral: it only consumes the gateway's
// `/api/audio/transcribe-stream` socket when the profile advertises
// `stt_streaming`, and returns null (leaving the file-based path untouched)
// on every older/non-streaming/unreachable case.

const mocks = vi.hoisted(() => ({
  hermesApi: vi.fn(),
  resolveSiblingWsUrl: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => 'local',
  getApiRequestProfile: () => 'default',
  hermesApi: mocks.hermesApi
}))

vi.mock('@/api/client', () => ({ profileScoped: () => ({}) }))

vi.mock('@/lib/sibling-ws-url', () => ({ resolveSiblingWsUrl: mocks.resolveSiblingWsUrl }))

class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances: FakeWebSocket[] = []
  bufferedAmount = 0
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  readyState = 0
  sent: unknown[] = []
  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }
  send = vi.fn((data: unknown) => {
    this.sent.push(data)
  })
  close = vi.fn(() => {
    this.readyState = FakeWebSocket.CLOSED
  })
  emit(message: unknown) {
    this.onmessage?.({ data: JSON.stringify(message) })
  }
}

const GATEWAY_WS = 'ws://127.0.0.1:5151/api/ws?token=local'

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  mocks.hermesApi.mockResolvedValue({ ok: true, stt_streaming: true })
  mocks.resolveSiblingWsUrl.mockResolvedValue(GATEWAY_WS)
})

afterEach(() => {
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

describe('resolveTranscriptionStreamUrl', () => {
  it('resolves the active (connection, profile) route onto the streaming path', async () => {
    const url = await resolveTranscriptionStreamUrl()

    expect(mocks.resolveSiblingWsUrl).toHaveBeenCalledWith(
      { connectionId: 'local', profile: 'default' },
      '/api/audio/transcribe-stream'
    )
    expect(url).toContain('profile=default')
  })

  it('preserves a backend-namespace profile already minted into the ws URL', async () => {
    mocks.resolveSiblingWsUrl.mockResolvedValue('wss://gw.example/api/ws?ticket=t&profile=backend-ns')

    const url = await resolveTranscriptionStreamUrl()

    expect(url).toContain('profile=backend-ns')
    expect(url).not.toContain('profile=default')
  })

  it('returns null when no route can be resolved', async () => {
    mocks.resolveSiblingWsUrl.mockRejectedValue(new Error('no connection bridge'))

    await expect(resolveTranscriptionStreamUrl()).resolves.toBeNull()
  })
})

describe('openTranscriptionStream', () => {
  it('opens a stream when the gateway advertises streaming', async () => {
    const stream = await openTranscriptionStream(vi.fn())

    expect(stream).not.toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(FakeWebSocket.instances[0].url).toBe(`${GATEWAY_WS}&profile=default`)
    expect(mocks.resolveSiblingWsUrl).toHaveBeenCalledWith(
      { connectionId: 'local', profile: 'default' },
      '/api/audio/transcribe-stream'
    )
    expect(mocks.hermesApi).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/audio/voice-config' }))
  })

  it('returns null when the profile does not advertise stt_streaming (file fallback)', async () => {
    mocks.hermesApi.mockResolvedValue({ ok: true, stt_streaming: false })

    await expect(openTranscriptionStream(vi.fn())).resolves.toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it('returns null when the capability endpoint is unavailable (older backend)', async () => {
    mocks.hermesApi.mockRejectedValue(new Error('HTTP 404'))

    await expect(openTranscriptionStream(vi.fn())).resolves.toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it('returns null when streaming is advertised but no stream route resolves', async () => {
    mocks.resolveSiblingWsUrl.mockRejectedValue(new Error('no route'))

    await expect(openTranscriptionStream(vi.fn())).resolves.toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it('plumbs partial transcripts and resolves the final transcript on finish', async () => {
    const onPartial = vi.fn()
    const stream = await openTranscriptionStream(onPartial)
    const socket = FakeWebSocket.instances[0]

    socket.emit({ type: 'ready' })
    socket.emit({ type: 'partial', text: 'hello wor' })
    expect(onPartial).toHaveBeenCalledWith('hello wor')

    const final = stream!.finish()
    await vi.waitFor(() => expect(socket.sent).toContain('finish'))

    socket.emit({ type: 'final', text: 'hello world' })
    await expect(final).resolves.toBe('hello world')
  })

  it('rejects the final transcript when the socket reports an error', async () => {
    const stream = await openTranscriptionStream(vi.fn())
    const socket = FakeWebSocket.instances[0]

    socket.emit({ type: 'error', error: 'provider exploded' })

    await expect(stream!.finish()).rejects.toThrow('provider exploded')
  })
})
