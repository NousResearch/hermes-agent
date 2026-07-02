import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { $gatewayState } from '@/store/session'

import { useGatewayRequest } from './use-gateway-request'

// The desktop app, dashboard SPA, and Ink stdio TUI all drive the same backend
// JSON-RPC server, which historically stamped every turn platform="tui". This
// hook now tags SESSION-ORIGINATING calls with source:'desktop' so the backend
// attributes turns to the desktop client (→ blackbox turns.platform →
// tokens.ace source chart). These tests pin that contract.

type RequestCall = { method: string; params: Record<string, unknown> }

function mountHook() {
  const calls: RequestCall[] = []
  const fakeGateway = {
    request: vi.fn(async (method: string, params: Record<string, unknown>) => {
      calls.push({ method, params })
      return { ok: true }
    })
  }

  let requestGateway!: ReturnType<typeof useGatewayRequest>['requestGateway']
  function Probe() {
    requestGateway = useGatewayRequest().requestGateway
    return null
  }

  // The hook reads the active gateway from the $gateway store.
  $gatewayState.set('open')
  $gateway.set(fakeGateway as never)
  render(<Probe />)

  return { calls, fakeGateway, requestGateway: () => requestGateway }
}

describe('useGatewayRequest — client source tagging', () => {
  beforeEach(() => {
    $gatewayState.set('open')
  })

  afterEach(() => {
    cleanup()
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  it("stamps source:'desktop' on session.create", async () => {
    const { calls, requestGateway } = mountHook()

    await act(async () => {
      await requestGateway()('session.create', { cols: 96 })
    })

    expect(calls).toHaveLength(1)
    expect(calls[0]).toMatchObject({
      method: 'session.create',
      params: { cols: 96, source: 'desktop' }
    })
  })

  it("stamps source:'desktop' on session.resume", async () => {
    const { calls, requestGateway } = mountHook()

    await act(async () => {
      await requestGateway()('session.resume', { session_id: 's1' })
    })

    expect(calls[0].params).toMatchObject({ session_id: 's1', source: 'desktop' })
  })

  it('does NOT add source to non-session-originating methods', async () => {
    const { calls, requestGateway } = mountHook()

    await act(async () => {
      await requestGateway()('session.steer', { session_id: 's1', text: 'hi' })
    })

    expect(calls[0].params).not.toHaveProperty('source')
  })

  it('never overrides an explicit caller-provided source', async () => {
    const { calls, requestGateway } = mountHook()

    await act(async () => {
      await requestGateway()('session.create', { source: 'tool', close_on_disconnect: true })
    })

    expect(calls[0].params).toMatchObject({ source: 'tool' })
  })
})
