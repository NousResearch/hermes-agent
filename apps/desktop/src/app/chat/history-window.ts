import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped, hermesApi, type ProfileScope } from '@/api/client'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessagesResponse } from '@/types/hermes'

export const HISTORY_WINDOW_LIMIT = 120

/** The around route is intentionally isolated from tail/backfill bookkeeping. */
export interface HistoryWindowResponse extends SessionMessagesResponse {
  pagination: NonNullable<SessionMessagesResponse['pagination']> & {
    has_older: boolean
    has_newer: boolean
  }
}

interface HistoryPage {
  messages: ChatMessage[]
  olderAvailable: boolean
  newerAvailable: boolean
}

export async function fetchHistoryWindow(
  storedId: string,
  rowId: number,
  scope: ProfileScope,
  signal: AbortSignal
): Promise<HistoryPage> {
  signal.throwIfAborted()
  const route = capabilityScoped(scope)
  const query = new URLSearchParams({ row_id: String(rowId), limit: String(HISTORY_WINDOW_LIMIT) })

  if (route.profile) {
    query.set('profile', route.profile)
  }

  // The Electron REST bridge cannot transfer AbortSignal over IPC. Cancellation
  // below releases the caller immediately and fences the eventual bounded read;
  // it does not pretend to cancel backend I/O or fall back to a full transcript.
  const response = await hermesApi<HistoryWindowResponse>({
    ...route,
    ...(typeof scope === 'object' && scope?.connectionId === 'local' ? { connectionId: 'local' } : {}),
    method: 'GET',
    path: `/api/sessions/${encodeURIComponent(storedId)}/messages/around?${query}`
  })

  signal.throwIfAborted()

  if (!Array.isArray(response.messages) || response.messages.length > HISTORY_WINDOW_LIMIT) {
    throw new Error('History response exceeds the bounded page size.')
  }

  return {
    messages: toChatMessages(response.messages),
    olderAvailable: response.pagination.has_older === true,
    newerAvailable: response.pagination.has_newer === true
  }
}

interface HistoryWindowOptions {
  /** Include runtime, durable id, owner connection/profile, and suppression. */
  scopeKey: string
  storedId: string | null
  scope: ProfileScope
  isCurrent: () => boolean
}

/** A single replaceable display page, never merged into the live message store. */
export function useHistoryWindow({ scopeKey, storedId, scope, isCurrent }: HistoryWindowOptions) {
  const lifetime = useMemo(() => ({ scopeKey }), [scopeKey])
  const latest = useRef({ lifetime, storedId, scope, isCurrent })
  latest.current = { lifetime, storedId, scope, isCurrent }
  const pending = useRef<AbortController | null>(null)
  const [selection, setSelection] = useState<{ lifetime: object; page: HistoryPage } | null>(null)
  const page = selection?.lifetime === lifetime ? selection.page : null

  const cancel = useCallback(() => {
    pending.current?.abort()
    pending.current = null
  }, [])

  useEffect(() => cancel, [cancel, lifetime])

  const returnToLatest = useCallback(() => {
    cancel()
    setSelection(null)
  }, [cancel])

  const revealRow = useCallback(
    async (rowId: number, signal: AbortSignal): Promise<string | null> => {
      cancel()

      if (signal.aborted || !Number.isSafeInteger(rowId) || rowId <= 0) {
        return null
      }
      const captured = latest.current

      if (!captured.storedId || !captured.isCurrent()) {
        return null
      }
      const controller = new AbortController()
      pending.current = controller
      const abort = () => controller.abort()
      signal.addEventListener('abort', abort, { once: true })
      let release!: () => void

      const aborted = new Promise<null>(resolve => {
        release = () => resolve(null)
        controller.signal.addEventListener('abort', release, { once: true })
      })

      try {
        const next = await Promise.race([
          fetchHistoryWindow(captured.storedId, rowId, captured.scope, controller.signal),
          aborted
        ])

        if (
          !next ||
          controller.signal.aborted ||
          latest.current.lifetime !== captured.lifetime ||
          !captured.isCurrent()
        ) {
          return null
        }

        const target = next.messages.find(message => message.rowId === rowId)

        if (!target) {
          return null
        }
        setSelection({ lifetime: captured.lifetime, page: next })

        return target.id
      } catch {
        // Missing/older backend, unreadable row, and failed reads preserve the
        // current page. The caller reports failure and can retry explicitly.
        return null
      } finally {
        signal.removeEventListener('abort', abort)
        controller.signal.removeEventListener('abort', release)

        if (pending.current === controller) {
          pending.current = null
        }
      }
    },
    [cancel]
  )

  return { page, revealRow, returnToLatest }
}
