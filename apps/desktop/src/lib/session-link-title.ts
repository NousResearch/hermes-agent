/**
 * Resolves `@session:<profile>/<id>` reference values to the session's title.
 *
 * Same shape as the external-link title resolver (`external-link.tsx`): a
 * process-lifetime cache, in-flight dedupe, and subscribers so every chip for
 * the same session repaints off one lookup. The sidebar list answers most
 * lookups for free; only an unknown id costs a REST round-trip.
 */
import { useEffect, useMemo, useState } from 'react'

import { getSession } from '@/hermes'
import { isCompactionEnvelopePreview } from '@/lib/chat-runtime'
import { parseSessionRefValue, sessionRefCacheKey, sessionRefFallbackLabel } from '@/lib/session-refs'
import { $sessions, sessionMatchesStoredId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

const titleCache = new Map<string, string>()
const titleInflight = new Map<string, Promise<string>>()
const titleSubs = new Map<string, Set<(value: string) => void>>()

/** Deliberately not `sessionTitle()` from chat-runtime: its "Untitled session"
 *  fallback is a worse chip label than the short id, so an untitled row
 *  resolves to empty and the caller's fallback wins. */
function sessionRowTitle(row: SessionInfo): string {
  const explicit = row.title?.trim()

  if (explicit) {return explicit}
  const preview = row.preview?.trim()

  if (preview && !isCompactionEnvelopePreview(preview)) {return preview}

  return ''
}

function profileMatches(sessionProfile: null | string | undefined, target?: string): boolean {
  if (!target) {
    return true
  }

  return ((sessionProfile ?? '').trim() || 'default') === (target.trim() || 'default')
}

export function lookupLocalSessionTitle(value: string): string {
  const { profile, sessionId } = parseSessionRefValue(value)

  if (!sessionId) {
    return ''
  }

  const row = $sessions
    .get()
    .find(session => sessionMatchesStoredId(session, sessionId) && profileMatches(session.profile, profile))

  return row ? sessionRowTitle(row) : ''
}

/** REST lookup that can't throw: the bridge is absent outside Electron, and a
 *  session id that isn't on this backend 404s. Both mean "no title". */
function requestSessionRow(sessionId: string, profile?: string): Promise<null | SessionInfo> {
  try {
    return Promise.resolve(getSession(sessionId, profile ?? null)).catch(() => null)
  } catch {
    return Promise.resolve(null)
  }
}

export function fetchSessionLinkTitle(value: string): Promise<string> {
  const key = sessionRefCacheKey(value)

  if (!key) {
    return Promise.resolve('')
  }

  const cached = titleCache.get(key)

  if (cached !== undefined) {
    return Promise.resolve(cached)
  }

  const inflight = titleInflight.get(key)

  if (inflight) {
    return inflight
  }

  const local = lookupLocalSessionTitle(value)

  if (local) {
    titleCache.set(key, local)

    return Promise.resolve(local)
  }

  const { profile, sessionId } = parseSessionRefValue(value)

  const promise = requestSessionRow(sessionId, profile)
    .then(row => (row ? sessionRowTitle(row) : ''))
    .then(title => {
      titleCache.set(key, title)
      titleInflight.delete(key)
      titleSubs.get(key)?.forEach(notify => notify(title))

      return title
    })

  titleInflight.set(key, promise)

  return promise
}

export function useSessionLinkTitle(value: string, fallbackLabel?: string): string {
  const key = useMemo(() => sessionRefCacheKey(value), [value])
  const fallback = fallbackLabel?.trim() || sessionRefFallbackLabel(value)
  const [title, setTitle] = useState(() => (key ? titleCache.get(key) || lookupLocalSessionTitle(value) : ''))

  useEffect(() => {
    if (!key) {
      return
    }

    const known = titleCache.get(key) || lookupLocalSessionTitle(value)

    setTitle(known)

    if (known) {
      return
    }

    const subs = titleSubs.get(key) ?? new Set<(resolved: string) => void>()

    subs.add(setTitle)
    titleSubs.set(key, subs)
    void fetchSessionLinkTitle(value)

    return () => {
      subs.delete(setTitle)

      if (!subs.size) {
        titleSubs.delete(key)
      }
    }
  }, [key, value])

  return title || fallback
}

export function __resetSessionLinkTitleCache(): void {
  titleCache.clear()
  titleInflight.clear()
  titleSubs.clear()
}
