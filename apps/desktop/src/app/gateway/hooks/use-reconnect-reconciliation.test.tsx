import { cleanup, render, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { HermesGateway } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { useReconnectReconciliation } from './use-reconnect-reconciliation'

// The B3 fix: after a socket drop + reconnect, per-session busy latches whose
// message.complete died with the old socket must be cleared from the fresh
// gateway's live-session snapshot — while genuinely running turns stay busy.

type Reconcile = (gateway: HermesGateway, profile: string) => Promise<void>

const sessionRow = (id: string, profile?: string): SessionInfo => ({ id, profile }) as unknown as SessionInfo

const busyState = (storedSessionId: string | null): ClientSessionState => ({
  ...createClientSessionState(storedSessionId),
  busy: true,
  awaitingResponse: true,
  streamId: 'assistant-stream-1',
  turnStartedAt: Date.now()
})

function makeHarness(initial: Record<string, ClientSessionState>) {
  const cache = new Map<string, ClientSessionState>(Object.entries(initial))
  let reconcile: Reconcile = async () => undefined

  function Harness() {
    const sessionStateByRuntimeIdRef = useRef(cache)

    reconcile = useReconnectReconciliation({
      sessionStateByRuntimeIdRef,
      updateSessionState: (sessionId, updater) => {
        const current = cache.get(sessionId) ?? createClientSessionState()
        const next = updater(current)
        cache.set(sessionId, next)

        return next
      }
    })

    return null
  }

  render(<Harness />)

  return { cache, reconcile: (gateway: HermesGateway, profile: string) => reconcile(gateway, profile) }
}

const gatewayWith = (rows: unknown[] | Error): HermesGateway =>
  ({
    request: vi.fn(async (method: string) => {
      expect(method).toBe('session.active_list')

      if (rows instanceof Error) {
        throw rows
      }

      return { sessions: rows }
    })
  }) as unknown as HermesGateway

describe('useReconnectReconciliation', () => {
  beforeEach(() => {
    $sessions.set([])
  })

  afterEach(() => {
    cleanup()
    $sessions.set([])
  })

  it('clears busy state for a session the gateway no longer knows (turn died with the socket)', async () => {
    const { cache, reconcile } = makeHarness({ 'rt-1': busyState('stored-1') })

    await reconcile(gatewayWith([]), 'default')

    await waitFor(() => {
      const state = cache.get('rt-1')!
      expect(state.busy).toBe(false)
      expect(state.awaitingResponse).toBe(false)
      expect(state.streamId).toBeNull()
      expect(state.turnStartedAt).toBeNull()
    })
  })

  it('clears busy state when the live session reports idle', async () => {
    const { cache, reconcile } = makeHarness({ 'rt-1': busyState('stored-1') })

    await reconcile(gatewayWith([{ id: 'rt-1', session_key: 'stored-1', status: 'idle' }]), 'default')

    expect(cache.get('rt-1')!.busy).toBe(false)
  })

  it('keeps a genuinely running turn busy (running:true stays busy)', async () => {
    const { cache, reconcile } = makeHarness({ 'rt-1': busyState('stored-1') })

    await reconcile(gatewayWith([{ id: 'rt-1', session_key: 'stored-1', status: 'working' }]), 'default')

    expect(cache.get('rt-1')!.busy).toBe(true)
    expect(cache.get('rt-1')!.awaitingResponse).toBe(true)
  })

  it('keeps a turn blocked on a user prompt (waiting) busy with its needsInput intact', async () => {
    const { cache, reconcile } = makeHarness({
      'rt-1': { ...busyState('stored-1'), needsInput: true }
    })

    await reconcile(gatewayWith([{ id: 'rt-1', status: 'waiting' }]), 'default')

    expect(cache.get('rt-1')!.busy).toBe(true)
    expect(cache.get('rt-1')!.needsInput).toBe(true)
  })

  it('matches live rows by stored session key when the runtime id rotated', async () => {
    const { cache, reconcile } = makeHarness({ 'rt-old': busyState('stored-1') })

    await reconcile(gatewayWith([{ id: 'rt-new', session_key: 'stored-1', status: 'working' }]), 'default')

    expect(cache.get('rt-old')!.busy).toBe(true)
  })

  it("leaves another profile's sessions to that profile's own reconnect", async () => {
    $sessions.set([sessionRow('stored-james', 'pm-james')])
    const { cache, reconcile } = makeHarness({ 'rt-james': busyState('stored-james') })

    await reconcile(gatewayWith([]), 'default')

    expect(cache.get('rt-james')!.busy).toBe(true)
  })

  it('reconciles a session whose profile matches the reconnected gateway', async () => {
    $sessions.set([sessionRow('stored-james', 'pm-james')])
    const { cache, reconcile } = makeHarness({ 'rt-james': busyState('stored-james') })

    await reconcile(gatewayWith([]), 'pm-james')

    expect(cache.get('rt-james')!.busy).toBe(false)
  })

  it('keeps every latch when the live-session probe fails (half-open socket)', async () => {
    const { cache, reconcile } = makeHarness({ 'rt-1': busyState('stored-1') })

    await reconcile(gatewayWith(new Error('request timed out: session.active_list')), 'default')

    expect(cache.get('rt-1')!.busy).toBe(true)
  })

  it('does not touch idle sessions and skips the probe entirely when nothing is busy', async () => {
    const idle = createClientSessionState('stored-1')
    const { cache, reconcile } = makeHarness({ 'rt-1': idle })
    const gateway = gatewayWith([])

    await reconcile(gateway, 'default')

    expect(cache.get('rt-1')).toBe(idle)
    expect((gateway as unknown as { request: ReturnType<typeof vi.fn> }).request).not.toHaveBeenCalled()
  })
})
