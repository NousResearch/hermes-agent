import { afterEach, expect, test, vi } from 'vitest'

import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { HermesGateway } from '@/hermes'
import { closeSecondaryGateways, requestGatewayForAgent, retainGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessions } from '@/store/session'

import { discardLostPrompt } from './discard-lost-prompt'

const originalBridge = window.hermesDesktop
const originalWebSocket = globalThis.WebSocket

afterEach(() => {
  closeSecondaryGateways()
  $sessions.set([])
  $activeGatewayProfile.set('default')
  window.hermesDesktop = originalBridge
  globalThis.WebSocket = originalWebSocket
})

async function fixture() {
  const wsPackage = 'ws'
  const { WebSocket, WebSocketServer } = await import(wsPackage)
  globalThis.WebSocket = WebSocket
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const frames: { route: string; method: string; params: Record<string, unknown> }[] = []
  server.on('connection', (socket: any, request: { url: string }) => {
    const route = new URL(request.url, 'http://localhost').pathname.slice(1)
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      frames.push({ route, method: frame.method, params: frame.params })

      const result = frame.method === 'session.resume'
        ? { session_id: 'stored', stored_session_id: 'stored', execution_generation: 9, revision: 1,
            pending: [{ admission_id: 'lost', status: 'unknown', execution_generation: route === 'owner' ? 4 : 7, text: 'LOST' }] }
        : { admission_id: frame.params.admission_id, ref: { session_id: frame.params.session_id }, status: 'terminal', outcome: 'interrupted' }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
  })
  const { port } = server.address() as { port: number }
  const url = (route: string) => `ws://127.0.0.1:${port}/${route}?native_dial=fixture&ticket=one-use`
  window.hermesDesktop = {
    connections: { list: async () => ({}) },
    getConnection: async () => ({ mode: 'local' }),
    getConnectionFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({ connectionId, profile, mode: 'remote' }),
    getGatewayWsUrlFor: async ({ connectionId }: { connectionId: string }) => url(connectionId)
  } as never
  $sessions.set([{ id: 'stored', connection_id: 'owner', profile: 'background' }] as never)
  $activeGatewayProfile.set('foreground')
  const ambient = new HermesGateway()
  await ambient.connect(url('ambient'))
  await ambient.request('session.resume', { session_id: 'stored' })
  const ambientRequest = vi.fn(ambient.request.bind(ambient)) as unknown as GatewayRequest

  return {
    frames,
    ambientRequest,
    async close() {
      closeSecondaryGateways()
      ambient.close()

      for (const socket of server.clients) { socket.terminate() }
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }
}

test('Discard uses the queue owner socket and its lost generation, including an unbound runtime', async () => {
  const f = await fixture()
  const release = await retainGatewayForAgent('owner', 'background')

  try {
    await requestGatewayForAgent('owner', 'background', 'session.resume', { session_id: 'stored' })
    await discardLostPrompt(null, 'stored', 'lost', f.ambientRequest)
    expect(f.frames.filter(frame => frame.method === 'prompt.resolve_unknown')).toEqual([
      { route: 'owner', method: 'prompt.resolve_unknown', params: { session_id: 'stored', admission_id: 'lost', execution_generation: 4 } }
    ])
    expect(f.ambientRequest).not.toHaveBeenCalled()
    // The acknowledged row is retired by the owning protocol, not by local queue deletion.
    await expect(discardLostPrompt('stored', 'stored', 'lost', f.ambientRequest)).rejects.toThrow('unknown')
    expect(f.frames.filter(frame => frame.method === 'prompt.resolve_unknown')).toHaveLength(1)
  } finally { release(); await f.close() }
})

test('Discard refuses unresolved or unavailable owners instead of falling back to the ambient socket', async () => {
  const f = await fixture()

  try {
    $sessions.set([])
    await expect(discardLostPrompt('stored', 'stored', 'lost', f.ambientRequest)).rejects.toThrow('owner')
    $sessions.set([{ id: 'stored', connection_id: 'owner', profile: 'background' }] as never)

    window.hermesDesktop!.getConnectionFor = async () => { throw new Error('owner unavailable') }
    await expect(discardLostPrompt('stored', 'stored', 'lost', f.ambientRequest)).rejects.toThrow('owner unavailable')
    expect(f.ambientRequest).not.toHaveBeenCalled()
    expect(f.frames.filter(frame => frame.method === 'prompt.resolve_unknown')).toEqual([])
  } finally { await f.close() }
})
