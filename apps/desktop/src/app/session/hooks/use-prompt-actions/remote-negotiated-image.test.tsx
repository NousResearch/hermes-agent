import { cleanup } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { hermesApi, HermesGateway } from '@/api/client'
import { $composerAttachments } from '@/store/composer'
import { $connectionWires, recordConnectionWire } from '@/store/connection-wire'
import { requestGatewayForAgent } from '@/store/gateway'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'

interface FixtureSocket {
  on(event: 'message', handler: (bytes: Buffer) => void): void
  send(data: string): void
}

vi.mock('@/api/client', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()), hermesApi: vi.fn()
}))
vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: RUNTIME_SESSION_ID })),
  getProfiles: vi.fn(async () => ({ profiles: [] })), getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000, setApiRequestProfile: vi.fn(), transcribeAudio: vi.fn()
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()), requestGatewayForAgent: vi.fn()
}))

afterEach(() => {
  cleanup()
  $connection.set(null)
  $composerAttachments.set([])
  $connectionWires.set({})
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.clearAllMocks()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

// Real composer upload/submit and real protocol client. Only the remote service,
// native file read and connection-pool selection are fixtures.
test.each(['public-canonical', 'public-legacy', 'native-canonical'] as const)('%s image submission preserves delivery semantics', async transport => {
  const canonical = transport !== 'public-legacy'
  const native = transport === 'native-canonical'
  const wsPackage = 'ws'
  const { WebSocketServer, WebSocket } = await import(wsPackage)
  vi.stubGlobal('WebSocket', WebSocket)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const frames: Array<{ method: string; params: Record<string, unknown> }> = []
  server.on('connection', (socket: FixtureSocket) => {
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      frames.push(frame)
      let result: unknown = {}

      if (frame.method === 'session.resume') {
        result = { session_id: RUNTIME_SESSION_ID, stored_session_id: RUNTIME_SESSION_ID,
          revision: 1, execution_generation: 1, authority_epoch: 1,
          running: false, messages: [], prompts: [], info: { profile_name: 'work' } }
      }

      if (frame.method === 'image.attach_bytes') {
        result = { attached: true, path: '/gateway/cache/images/fixture.png', mime: 'image/png' }
      }

      if (frame.method === 'prompt.submit') {
        result = canonical
          ? { admission_id: `accepted-${frames.length}`, ref: { session_id: RUNTIME_SESSION_ID },
              status: 'queued', sequence: 1, authority_epoch: 1, execution_generation: 1 }
          : { admission_id: frame.params.submission_id, submission_id: frame.params.submission_id,
              session_id: RUNTIME_SESSION_ID, status: 'queued' }
      }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: canonical ? { session_authority: true } : {}
    } }))
  })
  const client = new HermesGateway()
  const ready = new Promise<void>(resolve => client.on('gateway.ready', () => resolve()))
  const { port } = server.address() as { port: number }
  // The public cases have no native_dial marker or locally patched resolver.
  const wsUrl = `ws://127.0.0.1:${port}/api/ws?ticket=fixture${native ? '&native_dial=1' : ''}`
  const previous = window.hermesDesktop
  window.hermesDesktop = { ...previous, readFileDataUrl: vi.fn(async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC5sAAAAASUVORK5CYII=') }

  try {
    await client.connect(wsUrl)
    await ready
    // What the registry's gateway.ready hook publishes for this socket's owner connection
    // (covered with the real registry in gateway-background-reconnect.test.ts).
    recordConnectionWire('remote', client.sessionWire)
    $connection.set({ mode: native ? 'local' : 'remote', connectionId: 'remote', wsUrl } as never)
    setSessionOwnerHint(RUNTIME_SESSION_ID, { connectionId: 'remote', profile: 'work', mode: native ? 'local' : 'remote' })
    vi.mocked(requestGatewayForAgent).mockImplementation((_connectionId, profile, method, params, timeout, signal) =>
      client.request(method, { ...params, profile }, timeout, signal))
    vi.mocked(hermesApi).mockResolvedValue({ path: '/gateway/cache/images/fixture.png', mime_type: 'image/png' })

    // Positive control: the real client negotiated and adapts an ordinary send.
    const receipt = await client.request('prompt.submit', {
      session_id: RUNTIME_SESSION_ID, profile: 'work', submission_id: 'text-control', text: 'control'
    })

    expect(receipt).toMatchObject({ session_id: RUNTIME_SESSION_ID, submission_id: 'text-control', status: 'queued' })
    expect(frames.some(frame => frame.method === 'session.resume')).toBe(canonical)

    $composerAttachments.set([{ id: 'image:fixture', kind: 'image', label: 'fixture.png', path: '/local/fixture.png' }])
    let handle!: HarnessHandle
    await actRender(<Harness onReady={value => { handle = value }} rawAdmissionReceipts refreshSessions={async () => {}} requestGateway={vi.fn()} />)
    expect(await handle.submitText('Describe the attached image')).toBe(true)
    expect($composerAttachments.get()).toEqual([])
    const imageSubmit = frames.filter(frame => frame.method === 'prompt.submit').at(-1)!
    expect(imageSubmit.params.text).toContain('Describe the attached image')

    if (canonical) {
      expect(imageSubmit.params.attachments).toEqual([{ path: '/gateway/cache/images/fixture.png', mime: 'image/png' }])
      expect(hermesApi).toHaveBeenCalledWith(expect.objectContaining({
        path: '/api/chat/image-upload', connectionId: 'remote', profile: 'work'
      }))
      expect(frames.some(frame => frame.method === 'image.attach_bytes')).toBe(false)
    } else {
      expect(frames.some(frame => frame.method === 'image.attach_bytes')).toBe(true)
      expect(imageSubmit.params).not.toHaveProperty('attachments')
      expect(hermesApi).not.toHaveBeenCalled()
    }

  } finally {
    window.hermesDesktop = previous
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
