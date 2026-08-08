import { GatewayRequestError } from '@hermes/shared'
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $gateway, setPrimaryGateway } from '@/store/gateway'
import { $gatewayState } from '@/store/session'

import { useGatewayRequest } from './use-gateway-request'

const fakeGateway = { connectionState: 'open' } as unknown as HermesGateway

function stubDesktop() {
  window.hermesDesktop = {
    getConnection: vi.fn(async () => ({ authMode: 'token', wsUrl: 'ws://gw' })),
    touchBackend: vi.fn(async () => undefined)
  } as never
}

function renderRequestHook() {
  const request = vi.fn()
  const connect = vi.fn(async () => undefined)
  const gateway = { connectionState: 'closed', connect, request } as unknown as HermesGateway

  $gateway.set(gateway)
  setPrimaryGateway(gateway)

  const { result } = renderHook(() => useGatewayRequest())

  return { connect, gateway, request, result }
}

afterEach(() => {
  $gateway.set(null)
  $gatewayState.set('idle')
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  vi.restoreAllMocks()
})

describe('useGatewayRequest', () => {
  // The composer's `/` completions only exist when ChatBar receives a non-null
  // gateway PROP. `gatewayRef` is populated by a subscription effect, so it is
  // still null on the first render — a surface that read the ref while
  // rendering (session tiles / ⌘T tabs) shipped `gateway={null}` and silently
  // lost slash completions. The returned `gateway` value must be live
  // immediately so that never happens again.
  it('exposes the live gateway on the first render, before effects run', () => {
    $gateway.set(fakeGateway)

    const { result } = renderHook(() => useGatewayRequest())

    expect(result.current.gateway).toBe(fakeGateway)
  })

  it('tracks the gateway when the active socket changes', () => {
    const { result } = renderHook(() => useGatewayRequest())

    expect(result.current.gateway).toBeNull()

    act(() => $gateway.set(fakeGateway))

    expect(result.current.gateway).toBe(fakeGateway)
  })

  describe('reconnect + replay authority', () => {
    it('reconnects and replays a PRE-dispatch "not connected" failure — nothing was sent, so a replay cannot double-dispatch', async () => {
      stubDesktop()
      const { connect, request, result } = renderRequestHook()

      request
        .mockRejectedValueOnce(new GatewayRequestError('not_connected', 'Hermes gateway is not connected', false))
        .mockResolvedValueOnce({ ok: true })

      await waitFor(() => expect(result.current.gateway).toBeDefined())

      await expect(result.current.requestGateway('session.resume', { session_id: 's' })).resolves.toEqual({ ok: true })

      // The request went out exactly once after the socket was restored.
      expect(request).toHaveBeenCalledTimes(2)
      expect(connect).toHaveBeenCalledTimes(1)
    })

    it('reconnects but does NOT replay a POST-dispatch transport failure — the gateway may have accepted the request', async () => {
      stubDesktop()
      const { connect, request, result } = renderRequestHook()

      const closed = new GatewayRequestError('closed', 'Hermes gateway connection closed', true)
      request.mockRejectedValueOnce(closed)

      await waitFor(() => expect(result.current.gateway).toBeDefined())

      // prompt.submit must never be re-sent after a post-dispatch drop: the
      // first dispatch may have landed and a replay would run the turn twice.
      await expect(result.current.requestGateway('prompt.submit', { text: 'x' })).rejects.toBe(closed)
      expect(request).toHaveBeenCalledTimes(1)
      // The socket is still restored so the NEXT call works.
      expect(connect).toHaveBeenCalledTimes(1)
    })

    it('applies the same authority to untyped errors via message shape', async () => {
      stubDesktop()
      const { connect, request, result } = renderRequestHook()

      // Untyped (non-GatewayRequestError) transport error with the pre-flight
      // message → pre-dispatch → replayed after reconnect.
      request
        .mockRejectedValueOnce(new Error('Hermes gateway is not connected'))
        .mockResolvedValueOnce({ ok: true })

      await waitFor(() => expect(result.current.gateway).toBeDefined())
      await expect(result.current.requestGateway('session.status', {})).resolves.toEqual({ ok: true })
      expect(request).toHaveBeenCalledTimes(2)

      // Untyped 'connection closed' → may have been dispatched → reconnects
      // but does NOT replay.
      const closed = new Error('Hermes gateway connection closed')
      request.mockReset()
      request.mockRejectedValueOnce(closed)
      await expect(result.current.requestGateway('session.status', {})).rejects.toBe(closed)
      expect(request).toHaveBeenCalledTimes(1)
      expect(connect).toHaveBeenCalledTimes(2)
    })

    it('throws the original error when the reconnect fails (reauth or transport)', async () => {
      stubDesktop()
      const { connect, request, result } = renderRequestHook()

      const offline = new GatewayRequestError('not_connected', 'Hermes gateway is not connected', false)
      request.mockRejectedValueOnce(offline)
      connect.mockRejectedValueOnce(new Error('connect failed'))

      await waitFor(() => expect(result.current.gateway).toBeDefined())

      await expect(result.current.requestGateway('session.resume', {})).rejects.toBe(offline)
      expect(request).toHaveBeenCalledTimes(1)
    })
  })
})
