import { useEffect, useMemo, useRef, useState } from 'react'

import { mutateSessionHttp } from '@/api/sessions'
import type { SessionInfo } from '@/hermes'
import { captureIaRecorder } from '@/store/ia-metrics'
import { sessionPinId } from '@/store/session'

import { sessionRowKey } from './simplified-session-identity'

type PinWrite = { row: SessionInfo; value: boolean; expiresAt: number | null }

/** Keep acknowledged pins visible while older list requests finish. */
export function useSimplifiedPinWrites(owner: string | null, refresh: () => void | Promise<void>) {
  const [writes, setWrites] = useState<ReadonlyMap<string, PinWrite>>(new Map())
  const inflight = useRef(new Set<string>())

  const keyFor = (row: SessionInfo) =>
    sessionRowKey({ ...row, connection_id: row.connection_id || owner || undefined }, sessionPinId(row))

  const rows = useMemo(
    () => [...writes.values()].map(write => write.row).filter(row => row.connection_id || owner === null),
    [writes, owner]
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

    const record = captureIaRecorder()
    const previous = writes.get(key)
    const capturedRow = { ...row, connection_id: row.connection_id || owner || undefined }
    inflight.current.add(key)
    setWrites(current => new Map(current).set(key, { row: capturedRow, value, expiresAt: null }))

    try {
      await mutateSessionHttp(
        row.id,
        'PATCH',
        { pinned: value },
        {
          connectionId: row.connection_id || owner || undefined,
          profile: row.profile || 'default'
        }
      )
      record(value ? 'ia.pin.pinned' : 'ia.pin.unpinned')
    } catch (error) {
      record('ia.pin.failed')
      setWrites(current => {
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
      inflight.current.delete(key)
    }

    // Hold through late list pages, then let server state win (same 10s window as classic pins).
    setWrites(current => new Map(current).set(key, { row: capturedRow, value, expiresAt: Date.now() + 10_000 }))
    await refresh()
  }

  return {
    rows,
    setPinned,
    value: (row: SessionInfo) => writes.get(keyFor(row))?.value,
    pending: (row: SessionInfo) => writes.get(keyFor(row))?.expiresAt === null
  }
}
