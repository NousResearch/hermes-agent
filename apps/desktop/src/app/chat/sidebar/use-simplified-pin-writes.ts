import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect, useMemo, useRef, useState } from 'react'

import { mutateSessionHttp } from '@/api/sessions'
import type { SessionInfo } from '@/hermes'
import { captureIaRecorder } from '@/store/ia-metrics'
import { $cronSessions, $messagingSessions, $sessions, sessionPinId } from '@/store/session'
import { $sessionPinWriteIntent } from '@/store/session-pin-sync'
import {
  $removedSessionIds,
  captureSessionTombstoneGenerations,
  sessionRemovalIntersected,
  type SessionTombstoneGenerationSnapshot
} from '@/store/session-removal'

import { sessionRowKey } from './simplified-session-identity'

// A successful remote pin must outlive the search and the sidebar mount that discovered it.
interface CachedPin {
  row: SessionInfo
  removal: SessionTombstoneGenerationSnapshot
}
export const $simplifiedPinnedRows = atom<ReadonlyMap<string, CachedPin>>(new Map())

type PinWrite = CachedPin & { value: boolean; expiresAt: number | null }

function pinRowWasRemoved(pin: CachedPin) {
  return (
    sessionRemovalIntersected(pin.removal, pin.row.id) || sessionRemovalIntersected(pin.removal, sessionPinId(pin.row))
  )
}

/** Keep acknowledged pins visible while older list requests finish. */
export function useSimplifiedPinWrites(owner: string | null, refresh: () => void | Promise<void>) {
  const retained = useStore($simplifiedPinnedRows)
  const removed = useStore($removedSessionIds)
  const sessions = useStore($sessions)
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const [writes, setWrites] = useState<ReadonlyMap<string, PinWrite>>(new Map())
  const inflight = useRef(new Set<string>())
  const superseded = useRef(new Set<string>())
  useEffect(
    () =>
      $sessionPinWriteIntent.listen(intent => {
        if (!intent) {
          return
        }

        const key = JSON.stringify([intent.connectionId ?? '', intent.profile, intent.id])

        if (inflight.current.has(key)) {
          superseded.current.add(key)
        }

        const retained = new Map($simplifiedPinnedRows.get())
        retained.delete(key)
        $simplifiedPinnedRows.set(retained)
        setWrites(current => {
          if (!current.has(key)) {
            return current
          }

          const next = new Map(current)
          next.delete(key)

          return next
        })
      }),
    []
  )

  const keyFor = (row: SessionInfo) => sessionRowKey(row, sessionPinId(row))

  useEffect(() => {
    const next = new Map(retained)

    for (const row of [...sessions, ...cron, ...messaging]) {
      const key = sessionRowKey(row, sessionPinId(row))

      if (!writes.has(key) && (row.pinned === false || row.hidden || row.archived)) {
        next.delete(key)
      }
    }

    if (next.size !== retained.size) {
      $simplifiedPinnedRows.set(next)
    }
  }, [sessions, cron, messaging, writes, retained])

  const rows = useMemo(
    () =>
      [...retained.values(), ...writes.values()]
        .filter(pin => !removed.has(pin.row.id) && !pinRowWasRemoved(pin))
        .map(pin => pin.row)
        .filter(row => row.connection_id || owner === null),
    [writes, retained, owner, removed]
  )

  useEffect(() => {
    const now = Date.now()
    const next = new Map(writes)
    let delay = Infinity

    for (const [key, write] of writes) {
      if (write.expiresAt === null) {
        continue
      }

      if (write.expiresAt <= now) {
        next.delete(key)
      } else {
        delay = Math.min(delay, write.expiresAt - now)
      }
    }

    if (next.size !== writes.size) {
      setWrites(next)

      return
    }

    if (Number.isFinite(delay)) {
      const timer = window.setTimeout(() => setWrites(current => new Map(current)), delay)

      return () => window.clearTimeout(timer)
    }
  }, [writes])

  async function setPinned(row: SessionInfo, value: boolean) {
    const key = keyFor(row)

    if (inflight.current.has(key)) {
      return
    }

    const stopIntent = $sessionPinWriteIntent.listen(intent => {
      if (intent && JSON.stringify([intent.connectionId ?? '', intent.profile, intent.id]) === key) {
        superseded.current.add(key)
      }
    })

    const record = captureIaRecorder()
    const previous = writes.get(key)
    const capturedRow = row
    const removal = captureSessionTombstoneGenerations()
    superseded.current.delete(key)
    inflight.current.add(key)
    setWrites(current => new Map(current).set(key, { row: capturedRow, removal, value, expiresAt: null }))

    try {
      await mutateSessionHttp(
        row.id,
        'PATCH',
        { pinned: value },
        {
          connectionId: row.connection_id || undefined,
          profile: row.profile || 'default'
        }
      )
      record(value ? 'ia.pin.pinned' : 'ia.pin.unpinned')
    } catch (error) {
      record('ia.pin.failed')
      setWrites(current => {
        if (superseded.current.has(key)) {
          return current
        }

        const next = new Map(current)

        if (previous && (previous.expiresAt === null || previous.expiresAt > Date.now())) {
          next.set(key, previous)
        } else {
          next.delete(key)
        }

        return next
      })
      throw error
    } finally {
      stopIntent()
      inflight.current.delete(key)
    }

    // Hold through late list pages, then let server state win (same 10s window as classic pins).
    if (!superseded.current.has(key)) {
      const retained = new Map($simplifiedPinnedRows.get())

      if (value) {
        retained.set(key, { row: { ...capturedRow, pinned: true }, removal })
      } else {
        retained.delete(key)
      }

      $simplifiedPinnedRows.set(retained)
      setWrites(current =>
        new Map(current).set(key, { row: capturedRow, removal, value, expiresAt: Date.now() + 10_000 })
      )
    }

    await refresh()
  }

  return {
    rows,
    setPinned,
    value: (row: SessionInfo) => writes.get(keyFor(row))?.value,
    pending: (row: SessionInfo) => writes.get(keyFor(row))?.expiresAt === null
  }
}
