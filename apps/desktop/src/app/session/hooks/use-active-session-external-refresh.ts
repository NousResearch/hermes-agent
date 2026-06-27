import { type MutableRefObject, useEffect, useRef } from 'react'

import type { ClientSessionState } from '@/app/types'
import type { SessionInfo } from '@/types/hermes'

interface ActiveSessionExternalRefreshOptions {
  activeSessionId: string | null
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  messagingSessions: SessionInfo[]
  selectedStoredSessionId: string | null
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  sessions: SessionInfo[]
}

interface ObservedSessionRow {
  key: string
  signature: string | null
}

function sessionMatchesStoredId(session: SessionInfo, storedSessionId: string): boolean {
  return session.id === storedSessionId || session._lineage_root_id === storedSessionId
}

function findSessionRowByStoredId(rows: SessionInfo[], storedSessionId: string): SessionInfo | null {
  return rows.find(session => sessionMatchesStoredId(session, storedSessionId)) ?? null
}

export function sessionRefreshSignature(session: SessionInfo): string {
  return JSON.stringify([
    session.id,
    session._lineage_root_id ?? null,
    session.last_active,
    session.message_count,
    session.preview ?? null,
    session.title ?? null,
    session.input_tokens,
    session.output_tokens,
    session.tool_call_count,
    session.ended_at,
    session.is_active,
    session.source ?? null,
    session.profile ?? null
  ])
}

export function sameSessionRefreshRows(a: SessionInfo[], b: SessionInfo[]): boolean {
  if (a.length !== b.length) {
    return false
  }

  return a.every((session, index) => sessionRefreshSignature(session) === sessionRefreshSignature(b[index]))
}

export function activeSessionRefreshRow(
  sessions: SessionInfo[],
  messagingSessions: SessionInfo[],
  storedSessionId: string
): SessionInfo | null {
  return (
    findSessionRowByStoredId(sessions, storedSessionId) ?? findSessionRowByStoredId(messagingSessions, storedSessionId)
  )
}

export function useActiveSessionExternalRefresh({
  activeSessionId,
  hydrateFromStoredSession,
  messagingSessions,
  selectedStoredSessionId,
  sessionStateByRuntimeIdRef,
  sessions
}: ActiveSessionExternalRefreshOptions): void {
  const observedSessionRowRef = useRef<ObservedSessionRow | null>(null)

  useEffect(() => {
    if (!activeSessionId || !selectedStoredSessionId) {
      observedSessionRowRef.current = null

      return
    }

    const key = `${activeSessionId}\u0000${selectedStoredSessionId}`
    const row = activeSessionRefreshRow(sessions, messagingSessions, selectedStoredSessionId)
    const signature = row ? sessionRefreshSignature(row) : null
    const observed = observedSessionRowRef.current

    if (!observed || observed.key !== key) {
      observedSessionRowRef.current = { key, signature }

      return
    }

    if (observed.signature === signature) {
      return
    }

    observedSessionRowRef.current = { key, signature }

    if (!row || !signature) {
      return
    }

    const state = sessionStateByRuntimeIdRef.current.get(activeSessionId)
    const stateStoredSessionId = state?.storedSessionId

    const stateMatchesSelectedSession =
      !stateStoredSessionId ||
      stateStoredSessionId === selectedStoredSessionId ||
      sessionMatchesStoredId(row, stateStoredSessionId)

    const hasUnloadedPersistedMessages = row.message_count > (state?.messages.length ?? 0)

    if (
      !state ||
      !stateMatchesSelectedSession ||
      !hasUnloadedPersistedMessages ||
      state.busy ||
      state.awaitingResponse
    ) {
      return
    }

    void hydrateFromStoredSession(1, selectedStoredSessionId, activeSessionId).catch(() => undefined)
  }, [
    activeSessionId,
    hydrateFromStoredSession,
    messagingSessions,
    selectedStoredSessionId,
    sessionStateByRuntimeIdRef,
    sessions
  ])
}
