import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearAllPrompts, sessionApprovalRequest } from '@/store/prompts'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-approval'
let handleEvent: ((event: RpcEvent) => void) | null = null

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

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

describe('approval.request rule metadata', () => {
  beforeEach(() => {
    handleEvent = null
    clearAllPrompts()
  })

  afterEach(() => {
    cleanup()
    clearAllPrompts()
    vi.restoreAllMocks()
  })

  it('preserves rule metadata alongside approval capability constraints', async () => {
    render(<Harness />)
    await waitFor(() => expect(handleEvent).not.toBeNull())

    act(() =>
      handleEvent!({
        payload: {
          allow_permanent: false,
          allowlist_key: 'plugin_rule:ext-nav',
          choices: ['once', 'deny'],
          command: '<browser_navigate> (plugin approval rule)',
          description: 'external navigation',
          pattern_key: 'plugin_rule:ext-nav',
          pattern_keys: ['plugin_rule:ext-nav'],
          rule_key: 'ext-nav',
          smart_denied: true
        },
        session_id: SID,
        type: 'approval.request'
      } as RpcEvent)
    )

    expect(sessionApprovalRequest(SID).get()).toMatchObject({
      allowPermanent: false,
      allowlistKey: 'plugin_rule:ext-nav',
      choices: ['once', 'deny'],
      patternKey: 'plugin_rule:ext-nav',
      ruleKey: 'ext-nav',
      smartDenied: true
    })
  })
})
