import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped, hermesApi, type ProfileScope, sessionReadOwnerPin } from '@/api/client'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionMessagesResponse } from '@/types/hermes'

export const HISTORY_WINDOW_LIMIT = 120
export const HISTORY_RETAINED_PAGES = 3

export interface HistoryWindowResponse extends SessionMessagesResponse {
  pagination: NonNullable<SessionMessagesResponse['pagination']> & {
    has_older: boolean
    has_newer: boolean
    first_cursor?: number | null
    last_cursor?: number | null
    leading_prompt_row_id?: number | null
  }
}

interface HistorySlice {
  rows: SessionMessage[]
  olderAvailable: boolean
  newerAvailable: boolean
  firstCursor?: number
  lastCursor?: number
  leadingRowId?: number
}

interface HistoryPage {
  messages: ChatMessage[]
  slices: HistorySlice[]
  leadingRowId?: number
  olderAvailable: boolean
  newerAvailable: boolean
}

type PageAddress = { row_id: number } | { before_cursor: number } | { after_cursor: number }

const cursor = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined

async function fetchHistorySlice(
  storedId: string,
  address: PageAddress,
  scope: ProfileScope,
  signal: AbortSignal
): Promise<HistorySlice> {
  signal.throwIfAborted()

  // Owner connection pin (#125372): an around-read for a session owned by
  // another registry connection must read THAT host, not the ambient one.
  const route = {
    ...capabilityScoped(scope),
    ...(typeof scope === 'object' && scope?.connectionId === 'local' ? { connectionId: 'local' } : {}),
    ...sessionReadOwnerPin(storedId, scope)
  }

  const query = new URLSearchParams({
    ...Object.fromEntries(Object.entries(address).map(([k, v]) => [k, String(v)])),
    limit: String(HISTORY_WINDOW_LIMIT)
  })

  if (route.profile) {
    query.set('profile', route.profile)
  }

  // Electron's REST bridge cannot transfer AbortSignal. The caller races
  // cancellation and fences the eventual response; never fetch a full transcript.
  const response = await hermesApi<HistoryWindowResponse>({
    ...route,
    method: 'GET',
    path: `/api/sessions/${encodeURIComponent(storedId)}/messages/around?${query}`
  })

  signal.throwIfAborted()

  if (!Array.isArray(response.messages) || response.messages.length > HISTORY_WINDOW_LIMIT) {
    throw new Error('History response exceeds the bounded page size.')
  }

  const firstCursor = cursor(response.pagination.first_cursor)
  const lastCursor = cursor(response.pagination.last_cursor)

  // Mixed-version backends may ignore unknown parameters. Never publish a
  // repeated/non-adjacent page as though forward/backward navigation succeeded.
  if (
    !('row_id' in address) &&
    response.messages.length &&
    (firstCursor === undefined ||
      lastCursor === undefined ||
      firstCursor > lastCursor ||
      ('after_cursor' in address && firstCursor <= address.after_cursor) ||
      ('before_cursor' in address && lastCursor >= address.before_cursor))
  ) {
    throw new Error('History cursor did not advance.')
  }

  return {
    rows: response.messages,
    olderAvailable: response.pagination.has_older === true,
    newerAvailable: response.pagination.has_newer === true,
    firstCursor,
    leadingRowId: cursor(response.pagination.leading_prompt_row_id),
    lastCursor
  }
}

function historyPage(slices: HistorySlice[]): HistoryPage {
  // Fold across page boundaries so a tool result on the next page completes
  // its call rather than becoming an orphan. Retain raw pages, not bubble
  // counts: 120 tool rows can hydrate into one assistant message.
  const messages = toChatMessages(slices.flatMap(slice => slice.rows)).map(message => {
    const tool = message.parts.find(part => part.type === 'tool-call')

    const identity =
      message.rowId !== undefined
        ? `row-${message.rowId}`
        : tool?.type === 'tool-call' && tool.toolCallId
          ? `tool-${encodeURIComponent(tool.toolCallId)}`
          : message.id

    // Hydration's timestamp+page-index IDs can alias a different page before
    // React commits it. History identity must follow the durable occurrence.
    return { ...message, id: `history-${identity}` }
  })

  return {
    messages,
    slices,
    leadingRowId: slices[0].leadingRowId,
    olderAvailable: slices[0].olderAvailable,
    newerAvailable: slices.at(-1)!.newerAvailable
  }
}

interface HistoryWindowOptions {
  /** Includes the runtime, durable id, owner and suppression state. */
  scopeKey: string
  storedId: string | null
  scope: ProfileScope
  isCurrent: () => boolean
}

export function useHistoryWindow({ scopeKey, storedId, scope, isCurrent }: HistoryWindowOptions) {
  const lifetime = useMemo(() => ({ scopeKey }), [scopeKey])
  const latest = useRef({ lifetime, page: null as HistoryPage | null, storedId, scope, isCurrent })
  const pending = useRef<AbortController | null>(null)
  const [selection, setSelection] = useState<{ lifetime: object; page: HistoryPage } | null>(null)
  const [failure, setFailure] = useState<{ lifetime: object; kind: 'unavailable' | 'failed' } | null>(null)
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
    setFailure(null)
  }, [cancel])

  const read = useCallback(
    async (address: PageAddress, signal?: AbortSignal, beforeChange?: () => void) => {
      cancel()

      if (signal?.aborted) {
        return null
      }

      const captured = latest.current

      if (!captured.storedId || !captured.isCurrent()) {
        return null
      }

      const controller = new AbortController()
      pending.current = controller
      const abort = () => controller.abort()
      signal?.addEventListener('abort', abort, { once: true })
      let release!: () => void

      const aborted = new Promise<null>(resolve => {
        release = () => resolve(null)
        controller.signal.addEventListener('abort', release, { once: true })
      })

      setFailure(null)

      try {
        const next = await Promise.race([
          fetchHistorySlice(captured.storedId, address, captured.scope, controller.signal),
          aborted
        ])

        if (
          !next ||
          controller.signal.aborted ||
          latest.current.lifetime !== captured.lifetime ||
          latest.current.page !== captured.page ||
          !captured.isCurrent()
        ) {
          return null
        }

        const jumping = 'row_id' in address
        const older = 'before_cursor' in address

        if (!jumping && !next.rows.length) {
          const current = captured.page!
          const slices = [...current.slices]
          const edge = older ? 0 : slices.length - 1
          slices[edge] = { ...slices[edge], ...(older ? { olderAvailable: false } : { newerAvailable: false }) }
          // Exhaustion changes reach, not rendered content or the reading anchor.
          setSelection({
            lifetime: captured.lifetime,
            page: { ...current, slices, ...(older ? { olderAvailable: false } : { newerAvailable: false }) }
          })

          return null
        }

        const slices = jumping ? [next] : older ? [next, ...captured.page!.slices] : [...captured.page!.slices, next]
        const retained = older ? slices.slice(0, HISTORY_RETAINED_PAGES) : slices.slice(-HISTORY_RETAINED_PAGES)
        const selected = historyPage(retained)

        if (
          jumping &&
          !selected.messages.some(message => message.role === 'user' && message.rowId === address.row_id)
        ) {
          return null
        }

        // Releasing the opposite edge must not advertise that edge as exhausted.
        if (slices.length > retained.length) {
          if (older) {
            selected.newerAvailable = true
          } else {
            selected.olderAvailable = true
          }
        }

        // Keep the availability on the retained boundary too, for the next read.
        retained[0] = { ...retained[0], olderAvailable: selected.olderAvailable }
        retained[retained.length - 1] = { ...retained.at(-1)!, newerAvailable: selected.newerAvailable }
        beforeChange?.()
        setSelection({ lifetime: captured.lifetime, page: selected })

        return selected
      } catch {
        if (!controller.signal.aborted && latest.current.lifetime === captured.lifetime && captured.isCurrent()) {
          setFailure({ lifetime: captured.lifetime, kind: 'failed' })
        }

        return null
      } finally {
        signal?.removeEventListener('abort', abort)
        controller.signal.removeEventListener('abort', release)

        if (pending.current === controller) {
          pending.current = null
        }
      }
    },
    [cancel]
  )

  const revealRow = useCallback(
    async (rowId: number, signal: AbortSignal): Promise<string | null> => {
      if (!Number.isSafeInteger(rowId) || rowId <= 0) {
        return null
      }

      const selected = await read({ row_id: rowId }, signal)

      return selected?.messages.find(message => message.rowId === rowId)?.id ?? null
    },
    [read]
  )

  const adjacent = useCallback(
    async (older: boolean, beforeChange?: () => void): Promise<boolean> => {
      const captured = latest.current
      const current = captured.page

      if (!current || !(older ? current.olderAvailable : current.newerAvailable) || !captured.isCurrent()) {
        return false
      }

      const boundary = older ? current.slices[0].firstCursor : current.slices.at(-1)!.lastCursor

      if (boundary === undefined) {
        setFailure({ lifetime: captured.lifetime, kind: 'unavailable' })

        return false
      }

      return (
        (await read(older ? { before_cursor: boundary } : { after_cursor: boundary }, undefined, beforeChange)) !== null
      )
    },
    [read]
  )

  const revealOlder = useCallback((beforeChange?: () => void) => adjacent(true, beforeChange), [adjacent])
  const revealNewer = useCallback((beforeChange?: () => void) => adjacent(false, beforeChange), [adjacent])

  return {
    page,
    revealRow,
    returnToLatest,
    revealOlder,
    revealNewer,
    error: failure?.lifetime === lifetime ? failure.kind : null
  }
}
