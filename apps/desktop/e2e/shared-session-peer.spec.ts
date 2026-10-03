import { expect, test } from './test'
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

const SEED = 'E2E_SHARED_SESSION_LOCAL_SEED'
const PEER_INPUT = 'E2E_SHARED_SESSION_PEER_START'

// A real second transport client, without access to renderer stores.
class Peer {
  readonly events: Array<{ type: string; session_id?: string; payload?: Record<string, unknown> }> = []
  private nextId = 0
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()

  constructor(readonly socket: WebSocket) {
    socket.addEventListener('message', event => {
      const frame = JSON.parse(String(event.data))
      if (frame.method === 'event') this.events.push(frame.params)
      const request = this.pending.get(frame.id)
      if (!request) return
      this.pending.delete(frame.id)
      if (frame.error) request.reject(new Error(JSON.stringify(frame.error)))
      else request.resolve(frame.result)
    })
  }

  request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const id = ++this.nextId
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`Peer request timed out: ${method}`))
      }, 30_000)
      this.pending.set(id, {
        resolve: value => { clearTimeout(timer); resolve(value as T) },
        reject: error => { clearTimeout(timer); reject(error) },
      })
      this.socket.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
    })
  }
}

test('shows a peer starting input before completion and keeps one bubble after resume', async () => {
  test.setTimeout(180_000)
  let fixture: MockBackendFixture | undefined
  let peer: Peer | undefined
  try {
    fixture = await setupMockBackend({ mockServer: { holdFirstStreamForPrompt: PEER_INPUT } })
    await waitForAppReady(fixture, 120_000)
    const { page, mock } = fixture
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.fill(SEED)
    await page.keyboard.press('Enter')
    const viewport = page.locator('[data-slot="aui_thread-viewport"]')
    await expect(viewport).toContainText(MOCK_REPLY, { timeout: 60_000 })

    const url = await page.evaluate(async () => {
      const bridge = window as unknown as { hermesDesktop: { getGatewayWsUrl: () => Promise<string | { ok: boolean; wsUrl?: string }> } }
      const result = await bridge.hermesDesktop.getGatewayWsUrl()
      if (typeof result === 'string') return result
      if (!result.ok || !result.wsUrl) throw new Error('No sandbox gateway URL')
      return result.wsUrl
    })
    peer = new Peer(new WebSocket(url))
    await expect.poll(() => peer!.events.some(event => event.type === 'gateway.ready')).toBe(true)
    const listed = await peer.request<{ sessions: Array<{ id: string; preview?: string }> }>('session.list')
    const stored = listed.sessions.find(session => session.preview?.includes(SEED))
    expect(stored).toBeDefined()
    const resumed = await peer.request<{ session_id: string }>('session.resume', { session_id: stored!.id, omit_messages: true })
    const reply = peer.request<{ status: string }>('prompt.submit', {
      session_id: resumed.session_id, text: PEER_INPUT, submission_ref: 'peer-e2e-start',
    })
    await mock.waitForHeldStream()
    await expect(viewport.locator('[data-role="user"]').filter({ hasText: PEER_INPUT })).toHaveCount(1)
    expect(peer.events.filter(event => event.type === 'message.complete' && event.session_id === resumed.session_id)).toHaveLength(0)
    expect((await reply).status).toBe('streaming')

    // A second subscription/snapshot must not redeliver the starting bubble.
    await peer.request('session.resume', { session_id: stored!.id, omit_messages: true })
    await expect(viewport.locator('[data-role="user"]').filter({ hasText: PEER_INPUT })).toHaveCount(1)
    mock.releaseHeldStream()
    await expect.poll(() => peer!.events.some(event => event.type === 'message.complete' && event.session_id === resumed.session_id), { timeout: 60_000 }).toBe(true)
    await expect(viewport.locator('[data-role="user"]').filter({ hasText: PEER_INPUT })).toHaveCount(1)
  } finally {
    fixture?.mock.releaseHeldStream()
    peer?.socket.close()
    await fixture?.cleanup()
  }
})
