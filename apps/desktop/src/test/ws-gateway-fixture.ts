// A real `ws` server for HermesGateway dial tests (node environment only).
import { type ConnectionState, JsonRpcGatewayClient } from '@hermes/shared'

import type { HermesGateway } from '@/api/client'

export interface FixtureSocket {
  on(event: 'message', handler: (bytes: Buffer) => void): void
  on(event: 'close', handler: () => void): void
  send(data: string): void
  terminate(): void
}

export interface FixtureFrame {
  id: number | string
  method: string
  params: Record<string, any>
}

export function readyFrame(payload: Record<string, unknown> = { session_authority: true }): string {
  return JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload } })
}

export function reply(socket: FixtureSocket, frame: FixtureFrame, result: unknown = {}): void {
  socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
}

export async function listen() {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))

  return server
}

export type FixtureServer = Awaited<ReturnType<typeof listen>>

export function publicUrl(server: FixtureServer): string {
  return `ws://127.0.0.1:${(server.address() as { port: number }).port}/api/ws?ticket=public-fixture`
}

export async function shut(client: HermesGateway, server: FixtureServer): Promise<void> {
  client.close()

  for (const socket of server.clients) {
    socket.terminate()
  }
  await new Promise<void>(resolve => server.close(() => resolve()))
}

// The transport's own states. HermesGateway holds `open` from its subscribers until the dial's
// wire is known, so a test that needs the moment the socket opened listens below that.
export function onTransportState(client: HermesGateway, handler: (state: ConnectionState) => void): () => void {
  return JsonRpcGatewayClient.prototype.onState.call(client, handler)
}

export function transportOpened(client: HermesGateway): Promise<void> {
  return new Promise(resolve => {
    const off = onTransportState(client, state => {
      if (state === 'open') {
        queueMicrotask(() => off())
        resolve()
      }
    })
  })
}
