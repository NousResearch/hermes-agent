import { atom } from 'nanostores'

import { previewArtifactKey, previewName } from '@/lib/preview-targets'
import { readJson, writeJson } from '@/lib/storage'

import { ownerLookupSessionRows, resolveComposerSessionKey } from './session'
import type { SessionOwnerRoute } from './session-request-router'
import { knownOwnerForSession, runtimeSessionOwner, storedSessionIdForRuntimeId } from './session-states'

/**
 * Session-scoped feed of previewable artifacts (HTML files, localhost dev URLs)
 * a tool produced. Surfaced as compact links in the composer status stack —
 * NOT auto-opened and NOT a bulky inline card. Click opens the rail preview or
 * the browser; both are manual.
 *
 * Fed from the tool row itself (see tool-fallback.tsx) using the same detected
 * target the inline card used, so detection parity is exact.
 */
export interface PreviewArtifact {
  /** cwd captured at detection so a relative path still resolves on click. */
  cwd: string
  /** Canonical target identity, independent of its display label. */
  id: string
  label: string
  target: string
  /** Captured owner; foreground switches must not retarget a later dismiss. */
  dismissalScope?: string
}

const MAX_PER_SESSION = 4
const DISMISSED_PREVIEWS_KEY = 'hermes.desktop.previewDismissals.v1'
const MAX_DISMISSED_SESSIONS = 128
const MAX_DISMISSED_TARGETS = 64

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})

interface DismissedPreviewIds {
  [session: string]: string[]
}
const volatileDismissals = new Map<string, string[]>()
const scopeByRuntime = new Map<string, string>()

function dismissalKey(runtimeId: string, storedId: string): string {
  // The route's stored selection can advance before the old runtime unmounts.
  storedId = storedSessionIdForRuntimeId(runtimeId) ?? storedId
  // Historical rows in background tiles do not belong to the active connection.
  const owner = knownOwnerForSession(runtimeId) ?? runtimeSessionOwner(runtimeId) ?? knownOwnerForSession(storedId)
  const profile = typeof owner === 'string' ? owner : owner?.targetProfile || owner?.profile

  const rows = ownerLookupSessionRows().filter(
    row =>
      row.profile === profile &&
      (row.connection_id || 'local') === (typeof owner === 'object' && owner ? owner.connectionId : 'local')
  )

  const stableId = resolveComposerSessionKey(storedId, rows) ?? storedId

  // Bare profiles name the legacy profile pool, not the foreground connection.
  const scope =
    owner && typeof owner === 'object'
      ? JSON.stringify([owner.connectionId, owner.profile, owner.targetProfile || owner.profile, stableId])
      : typeof owner === 'string'
        ? JSON.stringify([null, owner, owner, stableId])
        : JSON.stringify(['runtime', runtimeId])

  const previous = scopeByRuntime.get(runtimeId)
  const before = previous ? storedOwner(previous) : null
  const after = storedOwner(scope)

  if (
    previous &&
    previous !== scope &&
    after &&
    (previous === JSON.stringify(['runtime', runtimeId]) ||
      (before && before[0] === null && before[1] === after[1] && before[2] === after[2] && before[3] === after[3]))
  ) {
    // Only refinement observed on THIS runtime joins keys. Another profile or
    // connection with a coincidentally equal stored id must not inherit it.
    rewriteDismissalScopes(key => (key === previous ? scope : key))
  }

  scopeByRuntime.set(runtimeId, scope)

  while (scopeByRuntime.size > MAX_DISMISSED_SESSIONS) {
    scopeByRuntime.delete(scopeByRuntime.keys().next().value!)
  }

  return scope
}

function readDismissedPreviewIds(): DismissedPreviewIds {
  const value = readJson<unknown>(DISMISSED_PREVIEWS_KEY)

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value)
      .slice(-MAX_DISMISSED_SESSIONS)
      .flatMap(([sid, ids]) => {
        if (!Array.isArray(ids)) {
          return []
        }

        if (sid.length > 4096 || !storedOwner(sid)) {
          return []
        }

        const strings = ids
          .filter((id): id is string => typeof id === 'string' && id.length > 0 && id.length <= 8192)
          .slice(-MAX_DISMISSED_TARGETS)

        return strings.length > 0 ? [[sid, strings]] : []
      })
  )
}

function rememberDismissedPreview(sid: string, id: string): void {
  const dismissed = readDismissedPreviewIds()
  const ids = [...new Set([...(dismissed[sid] ?? []), ...(volatileDismissals.get(sid) ?? [])])]

  if (ids.includes(id)) {
    return
  }

  const { [sid]: _previous, ...rest } = dismissed

  const next = Object.fromEntries(
    [...Object.entries(rest), [sid, [...ids, id].slice(-MAX_DISMISSED_TARGETS)]].slice(-MAX_DISMISSED_SESSIONS)
  )

  // Keep the close effective for this renderer even when storage is unavailable.
  volatileDismissals.set(sid, next[sid])

  while (volatileDismissals.size > MAX_DISMISSED_SESSIONS) {
    volatileDismissals.delete(volatileDismissals.keys().next().value!)
  }

  if (!sid.startsWith('["runtime",')) {
    writeJson(DISMISSED_PREVIEWS_KEY, next)

    if (readDismissedPreviewIds()[sid]?.includes(id)) {
      volatileDismissals.delete(sid)
    }
  }
}

function rewriteDismissalScopes(rewrite: (scope: string) => string | null): void {
  const current = { ...readDismissedPreviewIds(), ...Object.fromEntries(volatileDismissals) }
  const next: DismissedPreviewIds = Object.create(null)

  for (const [scope, ids] of Object.entries(current)) {
    const target = rewrite(scope)

    if (target) {
      next[target] = [...new Set([...(next[target] ?? []), ...ids])].slice(-MAX_DISMISSED_TARGETS)
    }
  }

  volatileDismissals.clear()

  for (const [scope, ids] of Object.entries(next).slice(-MAX_DISMISSED_SESSIONS)) {
    volatileDismissals.set(scope, ids)
  }

  writeJson(
    DISMISSED_PREVIEWS_KEY,
    Object.fromEntries(Object.entries(next).filter(([key]) => !key.startsWith('["runtime",')))
  )

  for (const [sid, items] of Object.entries($previewStatusBySession.get())) {
    writePreviews(
      sid,
      items.flatMap(item => {
        const scope = item.dismissalScope ? rewrite(item.dismissalScope) : undefined

        return scope === null ? [] : [{ ...item, dismissalScope: scope }]
      })
    )
  }
}

function storedOwner(scope: string): [string | null, string, string, string] | null {
  try {
    const value: unknown = JSON.parse(scope)

    return Array.isArray(value) &&
      value.length === 4 &&
      (value[0] === null || typeof value[0] === 'string') &&
      value.slice(1).every(item => typeof item === 'string')
      ? (value as [string | null, string, string, string])
      : null
  } catch {
    return null
  }
}

export function migratePreviewArtifactsForProfile(from: string, to: string): void {
  rewriteDismissalScopes(scope => {
    const owner = storedOwner(scope)

    if (!owner || (owner[0] && owner[0] !== 'local')) {
      return scope
    }

    return JSON.stringify([owner[0], owner[1] === from ? to : owner[1], owner[2] === from ? to : owner[2], owner[3]])
  })
}

export function dropPreviewArtifactsForProfile(profile: string, route?: Partial<SessionOwnerRoute>): void {
  rewriteDismissalScopes(scope => {
    const owner = storedOwner(scope)

    if (!owner) {
      return scope
    }

    const matches = route
      ? owner[1] === route.profile?.trim() &&
        (!route.connectionId || owner[0] === route.connectionId.trim()) &&
        (!route.targetProfile || owner[2] === route.targetProfile.trim())
      : (!owner[0] || owner[0] === 'local') && (owner[1] === profile || owner[2] === profile)

    return matches ? null : scope
  })
}

const writePreviews = (sid: string, items: PreviewArtifact[]) => {
  const current = $previewStatusBySession.get()

  if (items.length === 0) {
    if (!current[sid]) {
      return
    }

    const next = { ...current }
    delete next[sid]
    $previewStatusBySession.set(next)

    return
  }

  const labelled = items.map(item => {
    const name = previewName(item.target)
    const peers = items.filter(other => previewName(other.target) === name)
    let label = name

    if (peers.length > 1) {
      const parts = item.id.split('/')

      for (let depth = 2; depth <= parts.length; depth += 1) {
        label = parts.slice(-depth).join('/')

        if (peers.every(other => other.id === item.id || other.id.split('/').slice(-depth).join('/') !== label)) {
          break
        }
      }
    }

    return item.label === label ? item : { ...item, label }
  })

  $previewStatusBySession.set({ ...current, [sid]: labelled })
}

/**
 * Record a detected artifact, newest last, capped. Idempotent: a target already
 * in the list keeps its slot (the tool row re-registers on every render, so this
 * must not churn the atom or reorder rows).
 */
export function recordPreviewArtifact(
  sid: string,
  target: string,
  cwd: string,
  dismissalSid = sid,
  newProduction = false
) {
  const raw = target.trim()
  const id = previewArtifactKey(raw, cwd)

  if (!sid || !raw) {
    return
  }

  const scope = dismissalKey(sid, dismissalSid)
  const list = $previewStatusBySession.get()[sid] ?? []

  // Only the live completion handler may re-offer a newly produced file.
  // Historical mounts and reconnect replay never grant this intent.
  if (newProduction) {
    const dismissed = readDismissedPreviewIds()

    if (dismissed[scope]?.includes(id) || volatileDismissals.get(scope)?.includes(id)) {
      dismissed[scope] = (dismissed[scope] ?? []).filter(value => value !== id)
      volatileDismissals.set(
        scope,
        (volatileDismissals.get(scope) ?? []).filter(value => value !== id)
      )
      writeJson(DISMISSED_PREVIEWS_KEY, dismissed)
    }
  }

  if (list.some(item => item.id === id)) {
    return
  }

  if (readDismissedPreviewIds()[scope]?.includes(id) || volatileDismissals.get(scope)?.includes(id)) {
    return
  }

  writePreviews(
    sid,
    [...list, { cwd, id, label: previewName(raw), target: raw, dismissalScope: scope }].slice(-MAX_PER_SESSION)
  )
}

export function dismissPreviewArtifact(sid: string, id: string, dismissalSid = sid) {
  dismissalKey(sid, dismissalSid) // reconcile an owner refined since the row mounted
  const list = $previewStatusBySession.get()[sid]
  const scope = list?.find(item => item.id === id)?.dismissalScope ?? dismissalKey(sid, dismissalSid)

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }

  if (dismissalSid && id) {
    rememberDismissedPreview(scope, id)
  }
}

export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
