import { atom } from 'nanostores'

import { previewArtifactKey, previewName } from '@/lib/preview-targets'
import { readJson, writeJson } from '@/lib/storage'

import { normalizeProfileKey } from './profile'
import { ownerLookupSessionRows, resolveComposerSessionKey } from './session'
import { isSessionOwnerRoute, type SessionOwnerRoute } from './session-request-router'
import { knownOwnerForSession, storedSessionIdForRuntimeId } from './session-states'

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

/** Durable owner of a dismissal. `connectionId: null` is the legacy bare-profile
 *  pool (a profile name without a connection tag), not the foreground connection. */
interface DismissalScope {
  connectionId: string | null
  profile: string
  targetProfile: string
  sessionId: string
}

interface DismissedPreviewIds {
  [scope: string]: string[]
}

/** Dismissals that could not (yet) be persisted: runtime-only scopes and
 *  storage-write failures. Keeps a close effective for this renderer. */
const volatileDismissals = new Map<string, string[]>()
const scopeByRuntime = new Map<string, string>()

/** Not JSON on purpose: it can never collide with an encoded owner scope. */
const RUNTIME_SCOPE_PREFIX = 'runtime:'

function runtimeScopeKey(runtimeId: string): string {
  return RUNTIME_SCOPE_PREFIX + runtimeId
}

function isRuntimeScopeKey(key: string): boolean {
  return key.startsWith(RUNTIME_SCOPE_PREFIX)
}

function encodeScope(scope: DismissalScope): string {
  return JSON.stringify([scope.connectionId, scope.profile, scope.targetProfile, scope.sessionId])
}

function decodeScope(key: string): DismissalScope | null {
  try {
    const value: unknown = JSON.parse(key)

    if (
      !Array.isArray(value) ||
      value.length !== 4 ||
      (value[0] !== null && typeof value[0] !== 'string') ||
      !value.slice(1).every(item => typeof item === 'string')
    ) {
      return null
    }

    const [connectionId, profile, targetProfile, sessionId] = value as [string | null, string, string, string]

    return { connectionId, profile, sessionId, targetProfile }
  } catch {
    return null
  }
}

function capMap<V>(map: Map<string, V>): void {
  while (map.size > MAX_DISMISSED_SESSIONS) {
    map.delete(map.keys().next().value!)
  }
}

/** The scope key a dismissal for `runtimeId` belongs to right now. Reads the
 *  owner stores but never touches `scopeByRuntime`; see reconcileDismissalScope. */
function resolveDismissalScope(runtimeId: string, storedId: string): string {
  // The route's stored selection can advance before the old runtime unmounts.
  storedId = storedSessionIdForRuntimeId(runtimeId) ?? storedId
  // Historical rows in background tiles do not belong to the active connection.
  const owner = knownOwnerForSession(runtimeId) ?? knownOwnerForSession(storedId)

  if (!owner) {
    return runtimeScopeKey(runtimeId)
  }

  const route = isSessionOwnerRoute(owner) ? owner : null
  const profile = normalizeProfileKey(isSessionOwnerRoute(owner) ? owner.profile : owner)
  const targetProfile = normalizeProfileKey(route?.targetProfile || profile)
  // A bare profile names the legacy pool (`null`), which rows tag as local.
  const connectionId = route?.connectionId ?? null

  const rows = ownerLookupSessionRows().filter(
    row =>
      normalizeProfileKey(row.profile) === targetProfile && (row.connection_id || 'local') === (connectionId ?? 'local')
  )

  return encodeScope({
    connectionId,
    profile,
    sessionId: resolveComposerSessionKey(storedId, rows) ?? storedId,
    targetProfile
  })
}

/** A runtime's owner can be learnt after its rows mounted (unknown → bare
 *  profile → exact route). Dismissals recorded under the coarser key follow
 *  the refinement — only for THIS runtime; another profile or connection with
 *  a coincidentally equal stored id must not inherit them. */
function reconcileDismissalScope(runtimeId: string, storedId: string): string {
  const scope = resolveDismissalScope(runtimeId, storedId)
  const previous = scopeByRuntime.get(runtimeId)

  if (previous && previous !== scope) {
    const before = decodeScope(previous)
    const after = decodeScope(scope)

    const refined =
      after &&
      (previous === runtimeScopeKey(runtimeId) ||
        (before?.connectionId === null &&
          before.profile === after.profile &&
          before.targetProfile === after.targetProfile &&
          before.sessionId === after.sessionId))

    if (refined) {
      rewriteDismissalScopes(key => (key === previous ? scope : key))
    }
  }

  scopeByRuntime.set(runtimeId, scope)
  capMap(scopeByRuntime)

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
      .flatMap(([scope, ids]) => {
        if (!Array.isArray(ids) || scope.length > 4096 || !decodeScope(scope)) {
          return []
        }

        const strings = ids
          .filter((id): id is string => typeof id === 'string' && id.length > 0 && id.length <= 8192)
          .slice(-MAX_DISMISSED_TARGETS)

        return strings.length > 0 ? [[scope, strings]] : []
      })
  )
}

function isDismissed(scope: string, id: string): boolean {
  return Boolean(readDismissedPreviewIds()[scope]?.includes(id) || volatileDismissals.get(scope)?.includes(id))
}

/** Apply `update` to one scope's dismissed ids in both the persisted map and
 *  the renderer-local copy. Runtime-only scopes never reach storage; a failed
 *  storage write leaves the local copy in force for this renderer. */
function updateDismissed(scope: string, update: (ids: string[]) => string[]): void {
  const dismissed = readDismissedPreviewIds()
  const ids = update([...new Set([...(dismissed[scope] ?? []), ...(volatileDismissals.get(scope) ?? [])])])
  const { [scope]: _previous, ...rest } = dismissed

  const next = Object.fromEntries(
    [...Object.entries(rest), ...(ids.length > 0 ? [[scope, ids.slice(-MAX_DISMISSED_TARGETS)]] : [])].slice(
      -MAX_DISMISSED_SESSIONS
    )
  )

  if (ids.length > 0) {
    volatileDismissals.set(scope, next[scope])
    capMap(volatileDismissals)
  } else {
    volatileDismissals.delete(scope)
  }

  if (!isRuntimeScopeKey(scope)) {
    writeJson(DISMISSED_PREVIEWS_KEY, next)

    if (ids.length > 0 && readDismissedPreviewIds()[scope]?.includes(ids[ids.length - 1])) {
      volatileDismissals.delete(scope)
    }
  }
}

function rememberDismissedPreview(scope: string, id: string): void {
  if (!isDismissed(scope, id)) {
    updateDismissed(scope, ids => [...ids, id])
  }
}

function forgetDismissedPreview(scope: string, id: string): void {
  if (isDismissed(scope, id)) {
    updateDismissed(scope, ids => ids.filter(value => value !== id))
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

  writeJson(DISMISSED_PREVIEWS_KEY, Object.fromEntries(Object.entries(next).filter(([key]) => !isRuntimeScopeKey(key))))

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

export function migratePreviewArtifactsForProfile(from: string, to: string): void {
  rewriteDismissalScopes(key => {
    const scope = decodeScope(key)

    if (!scope || (scope.connectionId && scope.connectionId !== 'local')) {
      return key
    }

    return encodeScope({
      ...scope,
      profile: scope.profile === from ? to : scope.profile,
      targetProfile: scope.targetProfile === from ? to : scope.targetProfile
    })
  })
}

export function dropPreviewArtifactsForProfile(profile: string, route?: Partial<SessionOwnerRoute>): void {
  const routeProfile = normalizeProfileKey(route?.profile)
  const routeTarget = route?.targetProfile ? normalizeProfileKey(route.targetProfile) : ''
  const routeConnection = (route?.connectionId ?? '').trim()

  rewriteDismissalScopes(key => {
    const scope = decodeScope(key)

    if (!scope) {
      return key
    }

    const matches = route
      ? scope.profile === routeProfile &&
        (!routeConnection || scope.connectionId === routeConnection) &&
        (!routeTarget || scope.targetProfile === routeTarget)
      : (!scope.connectionId || scope.connectionId === 'local') &&
        (scope.profile === profile || scope.targetProfile === profile)

    return matches ? null : key
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

function appendPreviewArtifact(sid: string, raw: string, cwd: string, id: string, scope: string): void {
  if (isDismissed(scope, id)) {
    return
  }

  // Read after reconciling: the refinement rewrite updates listed items' scopes.
  const list = $previewStatusBySession.get()[sid] ?? []

  if (list.some(item => item.id === id)) {
    return
  }

  writePreviews(
    sid,
    [...list, { cwd, id, label: previewName(raw), target: raw, dismissalScope: scope }].slice(-MAX_PER_SESSION)
  )
}

/**
 * Record a detected artifact, newest last, capped. Idempotent: a target already
 * in the list keeps its slot (the tool row re-registers on every mount, so this
 * must not churn the atom or reorder rows). A dismissed target stays hidden —
 * historical mounts and reconnect replay never re-offer it.
 */
export function recordPreviewArtifact(sid: string, target: string, cwd: string, dismissalSid = sid) {
  const raw = target.trim()

  if (!sid || !raw) {
    return
  }

  const id = previewArtifactKey(raw, cwd)

  if ($previewStatusBySession.get()[sid]?.some(item => item.id === id)) {
    return
  }

  appendPreviewArtifact(sid, raw, cwd, id, reconcileDismissalScope(sid, dismissalSid))
}

/** A genuinely new successful production of a target the user dismissed
 *  earlier may offer it again. Only the live completion handler has that
 *  intent; see `recordPreviewArtifact` for every other feed. */
export function reofferPreviewArtifact(sid: string, target: string, cwd: string, dismissalSid = sid) {
  const raw = target.trim()

  if (!sid || !raw) {
    return
  }

  const id = previewArtifactKey(raw, cwd)
  const scope = reconcileDismissalScope(sid, dismissalSid)

  forgetDismissedPreview(scope, id)
  appendPreviewArtifact(sid, raw, cwd, id, scope)
}

export function dismissPreviewArtifact(sid: string, id: string, dismissalSid = sid) {
  const current = reconcileDismissalScope(sid, dismissalSid)
  const list = $previewStatusBySession.get()[sid]
  const scope = list?.find(item => item.id === id)?.dismissalScope ?? current

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }

  rememberDismissedPreview(scope, id)
}

export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
