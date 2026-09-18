import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped, hermesApi, type ProfileScope } from '@/api/client'
import { cachedTimelineIndex, previousPromptRowId, timelineIndexKey } from '@/components/assistant-ui/thread/timeline-index'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessagesResponse } from '@/types/hermes'

import { mergeOlderTranscriptPage } from './transcript-backfill'

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
  /** Display rows before this page's first row; how two pages prove they touch. */
  offset: number
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

  if (route.profile) {query.set('profile', route.profile)}

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
    newerAvailable: response.pagination.has_newer === true,
    offset: Math.max(0, Number(response.pagination.offset) || 0)
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
  const latest = useRef({ lifetime, page: null as HistoryPage | null, storedId, scope, isCurrent })
  const pending = useRef<AbortController | null>(null)
  const [selection, setSelection] = useState<{ lifetime: object; page: HistoryPage } | null>(null)
  const page = selection?.lifetime === lifetime ? selection.page : null

  latest.current = { lifetime, page, storedId, scope, isCurrent }

  const cancel = useCallback(() => {
    pending.current?.abort()
    pending.current = null
  }, [])

  useEffect(() => cancel, [cancel, lifetime])

  const returnToLatest = useCallback(() => {
    cancel()
    setSelection(null)
  }, [cancel])

  const revealRow = useCallback(async (rowId: number, signal: AbortSignal): Promise<string | null> => {
    cancel()

    if (signal.aborted || !Number.isSafeInteger(rowId) || rowId <= 0) {return null}
    const captured = latest.current

    if (!captured.storedId || !captured.isCurrent()) {return null}
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

      if (!next || controller.signal.aborted || latest.current.lifetime !== captured.lifetime || !captured.isCurrent()) {
        return null
      }

      const target = next.messages.find(message => message.rowId === rowId)

      if (!target) {return null}
      setSelection({ lifetime: captured.lifetime, page: next })

      return target.id
    } catch {
      // Missing/older backend, unreadable row, and failed reads preserve the
      // current page. The caller reports failure and can retry explicitly.
      return null
    } finally {
      signal.removeEventListener('abort', abort)
      controller.signal.removeEventListener('abort', release)

      if (pending.current === controller) {pending.current = null}
    }
  }, [cancel])

  /**
   * Prepend the page before this window's first prompt. The anchor's
   * predecessor comes from the same prompt index the rail draws, so the
   * transcript's entry point and the rail page one range. `beforePrepend` is
   * spent in the same commit as the prepend, exactly like a live-page grow.
   */
  const revealOlder = useCallback(async (beforePrepend?: () => void): Promise<boolean> => {
    const captured = latest.current
    const current = captured.page

    // No older rows before this page's first row, or nothing to anchor on yet.
    if (!current?.olderAvailable || !captured.storedId || !captured.isCurrent()) {
      return false
    }

    const anchor = current.messages.find(message => message.role === 'user' && message.rowId !== undefined)?.rowId

    cancel()
    const controller = new AbortController()
    pending.current = controller

    try {
      const rowId = await previousPromptRowId(captured.storedId, captured.scope, anchor)

      if (rowId === null || controller.signal.aborted) {
        // A complete index that lists no prompt before this window means the
        // backend's older rows can never be paged to: retire the offer so the
        // button and the top-edge auto-page stop promising a page that never
        // arrives. An incomplete index still leaves the page untouched for a retry.
        if (
          rowId === null &&
          anchor !== undefined &&
          !controller.signal.aborted &&
          latest.current.page === current &&
          cachedTimelineIndex(timelineIndexKey(captured.storedId, captured.scope))?.complete
        ) {
          setSelection({ lifetime: captured.lifetime, page: { ...current, olderAvailable: false } })
        }

        return false
      }

      const next = await fetchHistoryWindow(captured.storedId, rowId, captured.scope, controller.signal)
      // The around route reads forward from a prompt, so a turn longer than the
      // page limit leaves rows between that page's end and this anchor. Never
      // paint that as one continuous transcript: show the older page on its
      // own instead, the way a rail jump to that mark would.
      const contiguous = next.offset + next.messages.length >= current.offset
      const messages = contiguous ? mergeOlderTranscriptPage(current.messages, next.messages) : next.messages

      // A window replaced while this one was in flight owns the display page.
      if (
        controller.signal.aborted ||
        messages === current.messages ||
        latest.current.lifetime !== captured.lifetime ||
        latest.current.page !== current
      ) {
        return false
      }

      if (contiguous) {
        beforePrepend?.()
      }

      setSelection({
        lifetime: captured.lifetime,
        page: contiguous ? { ...next, messages, newerAvailable: current.newerAvailable } : next
      })

      return true
    } catch {
      // Missing/older backend, unreadable page, and an index that cannot name
      // the predecessor all leave the page untouched; the caller reports
      // failure and can retry explicitly.
      return false
    } finally {
      if (pending.current === controller) {pending.current = null}
    }
  }, [cancel])

  return { page, revealRow, returnToLatest, revealOlder }
}
