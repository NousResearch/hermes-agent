import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(),
  getSession: vi.fn()
}))

type Resume = ReturnType<typeof useSessionActions>['resumeSession']

function ResumeHarness({
  onReady,
  onStateUpdate,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  sessionStateByRuntimeIdRef
}: {
  onReady: (resume: Resume) => void
  onStateUpdate: (sessionId: string, state: ClientSessionState) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef,
    holdSessionTranscriptView: () => () => undefined,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current =
        sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)

      const next = updater(current)

      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      onStateUpdate(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

const clientState = (storedSessionId: string | null): ClientSessionState => createClientSessionState(storedSessionId)

afterEach(() => {
  cleanup()
  vi.mocked(getLatestSessionMessages).mockReset()
})

it('keeps an attached in-flight prompt anchored before a persisted background notice', async () => {
  const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
    current: new Map([['stored-A', 'rt-A']])
  }

  const state = clientState('stored-A')
  state.messages = [
    {
      id: 'runtime-user',
      role: 'user',
      parts: [{ type: 'text', text: 'earlier prompt' }]
    },
    {
      id: 'runtime-assistant',
      role: 'assistant',
      parts: [{ type: 'text', text: 'earlier answer' }]
    },
    {
      id: 'user-optimistic',
      role: 'user',
      parts: [{ type: 'text', text: 'current prompt' }],
      attachmentRefs: ['@image:/tmp/screenshot.png']
    },
    {
      id: 'assistant-stream-rt-A',
      role: 'assistant',
      parts: [{ type: 'text', text: 'partial answer' }],
      pending: true
    }
  ]

  const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
    current: new Map([['rt-A', state]])
  }

  const compressedRuntimeMessages = [
    { content: 'earlier prompt', role: 'user', timestamp: 1 },
    { content: 'earlier answer', role: 'assistant', timestamp: 2 }
  ]

  const persistedMessages = [
    { content: 'older prompt removed by compression', role: 'user', timestamp: -1 },
    { content: 'older answer removed by compression', role: 'assistant', timestamp: 0 },
    ...compressedRuntimeMessages,
    { content: 'current prompt\n@image:/tmp/screenshot.png', role: 'user', timestamp: 3 },
    { content: 'partial tool activity', role: 'assistant', timestamp: 4 },
    {
      content: '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]',
      role: 'user',
      timestamp: 5
    }
  ]

  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    messages: persistedMessages,
    session_id: 'stored-A'
  } as never)

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'session.activate') {
      return {
        session_id: 'rt-A',
        session_key: 'stored-A',
        resumed: 'stored-A',
        message_count: compressedRuntimeMessages.length,
        messages: [],
        messages_omitted: true,
        running: true,
        turn_started_at: 3,
        inflight: {
          user: 'current prompt',
          assistant: 'partial answer',
          streaming: true
        },
        info: {}
      } as never
    }

    return {} as never
  })

  let resumedState: ClientSessionState | undefined
  let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

  render(
    <ResumeHarness
      onReady={ready => (resume = ready)}
      onStateUpdate={(_sessionId, next) => (resumedState = next)}
      requestGateway={requestGateway}
      runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
      sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
    />
  )
  await waitFor(() => expect(resume).not.toBeNull())
  await resume!('stored-A', true)

  const currentPromptRows = (resumedState?.messages ?? []).filter(message =>
    JSON.stringify(message).includes('current prompt')
  )

  expect(currentPromptRows).toHaveLength(1)
  expect(currentPromptRows[0]).toMatchObject({ attachmentRefs: ['@image:/tmp/screenshot.png'] })
  expect(currentPromptRows[0].id).not.toContain('inflight')
  expect(JSON.stringify(resumedState?.messages)).toContain('partial answer')
})
