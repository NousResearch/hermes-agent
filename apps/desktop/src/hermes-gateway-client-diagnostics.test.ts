import { type GatewayDiagnosticContext, JsonRpcGatewayClient } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

class FakeSocket {
  readyState: number = WebSocket.CONNECTING
  throwOnSend = false
  private listeners = new Map<string, Set<(event: Event) => void>>()

  addEventListener(type: string, listener: (event: Event) => void): void {
    const handlers = this.listeners.get(type) ?? new Set<(event: Event) => void>()
    handlers.add(listener)
    this.listeners.set(type, handlers)
  }

  removeEventListener(type: string, listener: (event: Event) => void): void {
    this.listeners.get(type)?.delete(listener)
  }

  send(_data: string): void {
    if (this.throwOnSend) {
      throw new Error('send boom')
    }

    if (this.readyState !== WebSocket.OPEN) {
      throw new Error('socket is not open')
    }
  }

  close(code = 1000, reason = ''): void {
    this.readyState = WebSocket.CLOSED
    const event = new Event('close') as Event & { code: number; reason: string }
    event.code = code
    event.reason = reason
    this.dispatch('close', event)
  }

  open(): void {
    this.readyState = WebSocket.OPEN
    this.dispatch('open', new Event('open'))
  }

  fail(): void {
    this.readyState = WebSocket.CLOSED
    this.dispatch('error', new Event('error'))
  }

  private dispatch(type: string, event: Event): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event)
    }
  }
}

describe('JsonRpcGatewayClient diagnostics', () => {
  it('logs close code, reason, and pending method names without request params', async () => {
    const socket = new FakeSocket()
    const diagnosticLogger = vi.fn<(event: string, context: GatewayDiagnosticContext) => void>()

    const client = new JsonRpcGatewayClient({
      createRequestId: nextId => nextId,
      diagnosticLogger,
      socketFactory: () => socket as unknown as WebSocket
    })

    const connected = client.connect('ws://example.test/ws?ticket=secret-ticket')
    socket.open()
    await connected

    const pending = client.request('model.options', { session_id: 'secret-session-id' })
    socket.close(1006, 'abnormal closure')

    await expect(pending).rejects.toThrow('WebSocket closed')
    expect(diagnosticLogger).toHaveBeenCalledWith(
      'gateway.websocket.closed',
      expect.objectContaining({
        closeCode: 1006,
        closeReason: 'abnormal closure',
        event: 'gateway.websocket.closed',
        pendingCount: 1,
        pendingRequests: [expect.objectContaining({ id: 1, method: 'model.options' })],
        state: 'open'
      })
    )
    expect(JSON.stringify(diagnosticLogger.mock.calls[0]?.[1])).not.toContain('secret-session-id')
    expect(JSON.stringify(diagnosticLogger.mock.calls[0]?.[1])).not.toContain('secret-ticket')
  })

  it('keeps request cleanup best-effort when the diagnostic logger throws', async () => {
    const socket = new FakeSocket()

    const client = new JsonRpcGatewayClient({
      createRequestId: nextId => nextId,
      diagnosticLogger: () => {
        throw new Error('logger boom')
      },
      socketFactory: () => socket as unknown as WebSocket
    })

    const connected = client.connect('ws://example.test/ws')
    socket.open()
    await connected

    const pending = client.request('model.options')
    socket.close(1006, 'abnormal closure')

    await expect(pending).rejects.toThrow('WebSocket closed')
    expect(client.connectionState).toBe('closed')
  })

  it('includes the failing request in send-error diagnostics before cleanup', async () => {
    const socket = new FakeSocket()
    const diagnosticLogger = vi.fn<(event: string, context: GatewayDiagnosticContext) => void>()

    const client = new JsonRpcGatewayClient({
      createRequestId: nextId => nextId,
      diagnosticLogger,
      socketFactory: () => socket as unknown as WebSocket
    })

    const connected = client.connect('ws://example.test/ws')
    socket.open()
    await connected
    socket.throwOnSend = true

    await expect(client.request('prompt.submit')).rejects.toThrow('send boom')
    expect(diagnosticLogger).toHaveBeenCalledWith(
      'gateway.request.send_error',
      expect.objectContaining({
        pendingCount: 1,
        pendingRequests: [expect.objectContaining({ id: 1, method: 'prompt.submit' })]
      })
    )
  })
})
