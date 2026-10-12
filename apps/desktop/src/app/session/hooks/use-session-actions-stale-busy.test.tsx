import { render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { type SessionProfileRoute } from '@/store/session-request-router'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'
import { suppressTranscriptForView, transcriptRowContentKey } from './use-session-actions/transcript-provenance'
import type { TranscriptViewCutoff } from './use-session-actions/transcript-provenance'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => {
  const original = await importOriginal<Record<string, unknown>>()

  return {
    ...original,
    // Default-preserving spy: tests that route by the active source override it.
    activeGatewayConnectionId: vi.fn(original.activeGatewayConnectionId as () => null | string),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => () => undefined)
  }
})

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

const clientState = (storedSessionId: string | null): ClientSessionState => createClientSessionState(storedSessionId)

function ResumeHarness({
  onStateUpdate,
  onViewSync,
  onReady,
  getRouteToken = () => 'token',
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId = null,
  sessionStateByRuntimeIdRef
}: {
  getRouteToken?: () => string
  onStateUpdate?: (sessionId: string, state: ClientSessionState) => void
  onViewSync?: (sessionId: string, state: ClientSessionState) => void
  onReady: (
    resume: (storedSessionId: string, replaceRoute?: boolean, ownerRoute?: SessionProfileRoute) => Promise<unknown>
  ) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: string | null
  sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const runtimeMapRef = runtimeIdByStoredSessionIdRef ?? ref(new Map<string, string>())
  const stateMapRef = sessionStateByRuntimeIdRef ?? ref(new Map<string, ClientSessionState>())
  // Stand in for the real hook's gate: without it this stub would publish the
  // unproven warm cache before REST authority lands. Kept here (not per-test)
  // so every resume through this harness sees one shared suppression policy.
  const heldCutoffsByRuntimeIdRef = ref(new Map<string, TranscriptViewCutoff>())

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken,
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: runtimeMapRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef: ref<string | null>(selectedStoredSessionId),
    sessionStateByRuntimeIdRef: stateMapRef,
    holdSessionTranscriptView: runtimeId => {
      const rows = stateMapRef.current.get(runtimeId)?.messages ?? []

      const cutoff: TranscriptViewCutoff = {
        cutoffIds: new Set(rows.map(message => message.id)),
        cutoffKeys: new Set(rows.map(transcriptRowContentKey))
      }

      heldCutoffsByRuntimeIdRef.current.set(runtimeId, cutoff)

      return () => {
        heldCutoffsByRuntimeIdRef.current.delete(runtimeId)
      }
    },
    syncSessionStateToView: (sessionId, state) => {
      const cutoff = heldCutoffsByRuntimeIdRef.current.get(sessionId) ?? null

      onViewSync?.(sessionId, suppressTranscriptForView(state, cutoff))
    },
    updateSessionState: (sessionId, updater, storedSessionId) => {
      // Full default shape (not a bare {} cast) so seeded/derived fields like
      // turnStartedAt behave as in production state updates.
      const current = stateMapRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)
      const next = updater(current)

      stateMapRef.current.set(sessionId, next)
      onStateUpdate?.(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

// A busy claim whose terminal events were lost — the socket it streamed on was
// closed mid-turn by a connection switch or a reconnect — must settle once the
// backend reports the turn over; nothing else ever will.
describe('resume settles busy claims older than its snapshot', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('settles a busy claim whose terminal events were lost once session.activate reports the turn over', async () => {
    // The window left this conversation mid-turn: a connection switch closed
    // the socket it streamed on, the turn finished on the backend with nobody
    // listening, and the cache still holds the frozen mid-stream snapshot.
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const frozen: ClientSessionState = {
      ...clientState('stored-A'),
      busy: true,
      sawAssistantPayload: true,
      messages: [
        { id: 'cached-user', role: 'user', parts: [{ type: 'text', text: 'second question' }] },
        {
          id: 'assistant-stream-rt-A',
          role: 'assistant',
          pending: true,
          parts: [{ type: 'text', text: 'partial tool commentary' }]
        }
      ]
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', frozen]])
    }

    const requestGateway = vi.fn(async (method: string) =>
      method === 'session.activate'
        ? ({
            session_id: 'rt-A',
            session_key: 'stored-A',
            resumed: 'stored-A',
            message_count: 2,
            messages: [],
            messages_omitted: true,
            running: false,
            info: {}
          } as never)
        : ({} as never)
    )

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'second question', role: 'user', timestamp: 1 },
        { content: 'FINAL ANSWER', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-A'
    } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const settled = sessionStateByRuntimeIdRef.current.get('rt-A')
    expect(settled?.busy).toBe(false)
    expect(settled?.messages.map(chatMessageText)).toContain('FINAL ANSWER')
  })

  it('settles a stale busy claim when a cold resume hands back the same parked runtime, now idle', async () => {
    // The stored->runtime binding was invalidated (source switch / reconnect),
    // but the backend kept the runtime parked: session.resume returns the very
    // id whose frozen mid-turn snapshot the cache still holds.
    const frozen: ClientSessionState = {
      ...clientState('stored-A'),
      busy: true,
      sawAssistantPayload: true,
      messages: [{ id: 'cached-user', role: 'user', parts: [{ type: 'text', text: 'second question' }] }]
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', frozen]])
    }

    const requestGateway = vi.fn(async (method: string) =>
      method === 'session.resume'
        ? ({
            session_id: 'rt-A',
            resumed: 'stored-A',
            message_count: 2,
            messages: [],
            running: false,
            info: {}
          } as never)
        : ({} as never)
    )

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'second question', role: 'user', timestamp: 1 },
        { content: 'FINAL ANSWER', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-A'
    } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(requestGateway.mock.calls.map(([method]) => method)).toContain('session.resume')
    expect(sessionStateByRuntimeIdRef.current.get('rt-A')?.busy).toBe(false)
  })
})
