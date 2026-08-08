import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { chatMessageText } from '@/lib/chat-messages'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let latestState: ClientSessionState | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
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
      latestState = next
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

const fireBackground = (payload: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: SID, type: 'background.complete' }))

describe('useMessageStream background.complete', () => {
  beforeEach(() => {
    handleEvent = null
    latestState = null
  })

  afterEach(() => {
    cleanup()
  })

  it('voegt een system-bericht toe met het background-resultaat', async () => {
    await mountStream()
    fireBackground({ text: 'het onderzoeksresultaat', task_id: 'bg_1234_ab' })

    const messages = latestState?.messages ?? []
    const last = messages[messages.length - 1]
    expect(last?.role).toBe('system')
    const text = chatMessageText(last)
    expect(text).toContain('✅ Background task complete')
    expect(text).toContain('bg_1234_ab')
    expect(text).toContain('het onderzoeksresultaat')
  })

  it('voegt niets toe als de tekst leeg is', async () => {
    await mountStream()
    fireBackground({ text: '   ' })
    expect(latestState?.messages ?? []).toHaveLength(0)
  })
})
