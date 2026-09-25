import { afterEach, expect, it, vi } from 'vitest'

import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, setActiveSessionId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import { rebindPaneToResumedRuntime, type ResumedRuntimeBindingDeps } from './submit'

const STORED = 'stored-b'
const RESUMED_RUNTIME = 'rt-resumed'

function row(id: string, role: ChatMessage['role'], text: string): ChatMessage {
  return { id, parts: [textPart(text)], role }
}

function sessionInfo(id: string, lineageRoot?: string): SessionInfo {
  return {
    _lineage_root_id: lineageRoot,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: id,
    tool_call_count: 0
  }
}

interface BoundPaneResult {
  activeSessionIdRef: { current: string | null }
  earlier: ChatMessage[]
  resumed?: ClientSessionState
  updateSessionState: ResumedRuntimeBindingDeps['updateSessionState']
}

function bindPane({
  paneStoredSessionId,
  sessions = [],
  targetStoredSessionId = STORED
}: {
  paneStoredSessionId: string
  sessions?: SessionInfo[]
  targetStoredSessionId?: string
}): BoundPaneResult {
  const earlier: ChatMessage[] = [
    row('u1', 'user', 'earlier prompt'),
    row('a1', 'assistant', 'earlier reply')
  ]

  const paneState: ClientSessionState = { ...createClientSessionState(paneStoredSessionId), messages: earlier }

  const states: Map<string, ClientSessionState> = new Map([
    [RESUMED_RUNTIME, createClientSessionState(targetStoredSessionId)]
  ])

  const updateSessionState: ResumedRuntimeBindingDeps['updateSessionState'] = (
    sessionId,
    updater,
    storedSessionId
  ) => {
    const current = states.get(sessionId) ?? createClientSessionState(storedSessionId)
    const updated = updater(current)

    states.set(sessionId, updated)

    return updated
  }

  const activeSessionIdRef: BoundPaneResult['activeSessionIdRef'] = { current: 'rt-stale' }

  setActiveSessionId('rt-stale')
  rebindPaneToResumedRuntime({
    activeSessionIdRef,
    paneState,
    resumedRuntimeId: RESUMED_RUNTIME,
    sessions,
    storedSessionId: targetStoredSessionId,
    updateSessionState: vi.fn(updateSessionState)
  })

  return { activeSessionIdRef, earlier, resumed: states.get(RESUMED_RUNTIME), updateSessionState }
}

afterEach((): void => {
  setActiveSessionId(null)
})

it('moves the pane and carries the selected conversation into an empty resumed runtime', (): void => {
  const { activeSessionIdRef, earlier, resumed } = bindPane({ paneStoredSessionId: STORED })

  expect(activeSessionIdRef.current).toBe(RESUMED_RUNTIME)
  expect($activeSessionId.get()).toBe(RESUMED_RUNTIME)
  expect(resumed?.messages).toEqual(earlier)
})

it('recognizes the same conversation after compression rotates its stored id', (): void => {
  const rotatedStoredId = `${STORED}-next`

  const { earlier, resumed } = bindPane({
    paneStoredSessionId: STORED,
    sessions: [sessionInfo(rotatedStoredId, STORED)],
    targetStoredSessionId: rotatedStoredId
  })

  expect(resumed?.messages).toEqual(earlier)
})

it('never carries another conversation into the resumed runtime', (): void => {
  const { resumed } = bindPane({ paneStoredSessionId: 'stored-other' })

  expect(resumed?.messages).toEqual([])
  expect($activeSessionId.get()).toBe(RESUMED_RUNTIME)
})
