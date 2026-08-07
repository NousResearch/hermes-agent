import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let stateByRuntimeId = new Map<string, ClientSessionState>()

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(stateByRuntimeId)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => handleEvent!({ payload, session_id: SID, type }))
}

function systemTexts() {
  const state = stateByRuntimeId.get(SID)

  return (state?.messages ?? [])
    .filter(message => message.role === 'system')
    .flatMap(message => message.parts.map(part => ('text' in part ? part.text : '')))
}

// #66829: the gateway downgrades an attached image to a text description when
// the routed model has no vision (or native attachment fails). The Ink TUI puts
// that on its status line; the desktop had no equivalent, so the user saw a
// confident answer with no sign the model never looked at the image.
describe('useMessageStream image-routing downgrade', () => {
  beforeEach(() => {
    handleEvent = null
    stateByRuntimeId = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders the downgrade notice as a persistent system message', async () => {
    await mountStream()

    emit('status.update', {
      kind: 'image_routing',
      text: '⚠ Image sent as a text description — model has no vision support.'
    })

    expect(systemTexts()).toEqual([
      '⚠ Image sent as a text description — model has no vision support.'
    ])
  })

  it('keeps one notice per downgraded turn rather than collapsing them', async () => {
    await mountStream()

    emit('status.update', { kind: 'image_routing', text: '⚠ first turn downgraded.' })
    emit('status.update', { kind: 'image_routing', text: '⚠ second turn downgraded.' })

    expect(systemTexts()).toEqual(['⚠ first turn downgraded.', '⚠ second turn downgraded.'])
  })

  it('ignores an empty payload instead of adding a blank message', async () => {
    await mountStream()

    emit('status.update', { kind: 'image_routing', text: '   ' })
    emit('status.update', { kind: 'image_routing' })

    expect(systemTexts()).toEqual([])
  })

  // Guards the actual defect: the notice used to ship as kind "process", which
  // the desktop consumes only to re-sync background-process state, dropping the
  // text entirely. If it ever regresses to that kind, nothing renders.
  it('does not render process-kind status updates as messages', async () => {
    await mountStream()

    emit('status.update', { kind: 'process', text: '⚠ Image sent as a text description — nope.' })

    expect(systemTexts()).toEqual([])
  })
})
