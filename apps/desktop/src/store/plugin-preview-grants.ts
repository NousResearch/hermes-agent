import { atom } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'

/**
 * USER GRANTS FOR `ctx.preview.executeJavaScript` — which plugin may run
 * script in the preview Browser on which page origin.
 *
 * The preview pane can hold pages the user is signed in to, so script there
 * reads session data. A plugin that declared the capability still runs
 * nothing until the user allows it for that origin; the grant is per plugin
 * AND per origin, persisted across launches and revocable from Capabilities ▸
 * Plugins. "Don't allow" holds for this app run only, so a plugin retrying in
 * a loop cannot stack prompts, and the question comes back next launch.
 *
 * Global scope (like the plugin enable decisions): Desktop plugins are
 * app-level, not per profile.
 */

const pairKey = (pluginId: string, origin: string) => `${pluginId}\n${origin}`

/** pluginId -> origins the user allowed. */
export type PreviewScriptGrants = Record<string, string[]>

function sanitizeGrants(value: unknown): PreviewScriptGrants {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const grants: PreviewScriptGrants = {}

  for (const [pluginId, origins] of Object.entries(value)) {
    if (!Array.isArray(origins)) {
      continue
    }

    const kept = origins.filter((origin): origin is string => typeof origin === 'string' && isGrantableOrigin(origin))

    if (kept.length > 0) {
      grants[pluginId] = [...new Set(kept)]
    }
  }

  return grants
}

const GRANTS_KEY = 'hermes.desktop.pluginPreviewScriptGrants.v1'

// Every window that runs plugins shares this localStorage record, so it — not
// a window's atom — is the source of truth: each check and each change reads
// it fresh (a revoke in one window binds the others at once, and a grant in a
// stale window cannot write a revoked origin back). The atom mirrors it for UI.
function readStoredGrants(): PreviewScriptGrants {
  const raw = readKey(GRANTS_KEY)

  try {
    return raw ? sanitizeGrants(JSON.parse(raw)) : {}
  } catch {
    return {}
  }
}

export const $previewScriptGrants = atom<PreviewScriptGrants>(readStoredGrants())

/** Persist `next` and mirror what storage actually holds afterwards (writes
 *  are best-effort: a full or blocked store keeps the old record). Returns
 *  whether `next` was saved. */
function commitGrants(next: PreviewScriptGrants): boolean {
  writeKey(GRANTS_KEY, JSON.stringify(next))
  const saved = readStoredGrants()

  $previewScriptGrants.set(saved)

  return JSON.stringify(saved) === JSON.stringify(next)
}

/** Revokes the user made in this window that storage refused to save. They
 *  still bind here (the call asks again), so a failed write never leaves a
 *  revoked plugin running silently. */
const unsavedRevokes = new Set<string>()

if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === GRANTS_KEY || event.key === null) {
      $previewScriptGrants.set(readStoredGrants())
    }
  })
}

/** "Don't allow" answers for this app run, as `${pluginId}\n${origin}`. */
const $sessionDenials = atom<ReadonlySet<string>>(new Set())


/** A grant names a real web origin: `http(s)://host[:port]`, nothing else.
 *  Opaque origins (`null` for about:blank, data:, sandboxed frames) and
 *  `file://` cannot scope a grant to one site, so they are never offered. */
export function isGrantableOrigin(origin: string): boolean {
  try {
    const url = new URL(origin)

    return (url.protocol === 'http:' || url.protocol === 'https:') && url.origin === origin
  } catch {
    return false
  }
}

export function hasPreviewScriptGrant(pluginId: string, origin: string): boolean {
  return (
    !unsavedRevokes.has(pairKey(pluginId, origin)) && (readStoredGrants()[pluginId]?.includes(origin) ?? false)
  )
}

export function grantPreviewScript(pluginId: string, origin: string): void {
  const grants = readStoredGrants()
  const current = grants[pluginId] ?? []

  unsavedRevokes.delete(pairKey(pluginId, origin))
  commitGrants(current.includes(origin) ? grants : { ...grants, [pluginId]: [...current, origin] })
}

/** Revoke one origin, or every origin for the plugin when omitted. Returns
 *  false when storage refused the change (it still binds this window). */
export function revokePreviewScript(pluginId: string, origin?: string): boolean {
  const { [pluginId]: current = [], ...rest } = readStoredGrants()
  const revoked = origin === undefined ? current : current.filter(entry => entry === origin)
  const kept = current.filter(entry => !revoked.includes(entry))

  if (commitGrants(kept.length > 0 ? { ...rest, [pluginId]: kept } : rest)) {
    return true
  }

  revoked.forEach(entry => unsavedRevokes.add(pairKey(pluginId, entry)))

  return false
}

export interface PreviewScriptConsentRequest {
  id: number
  pluginId: string
  pluginName: string
  origin: string
}

/** Pending prompts, oldest first. The app-owned dialog renders the head. */
export const $previewScriptConsentQueue = atom<readonly PreviewScriptConsentRequest[]>([])

const pending = new Map<string, { promise: Promise<boolean>; resolve: (allowed: boolean) => void }>()
let nextRequestId = 1

/**
 * Resolve whether `pluginId` may run script on `origin`: an existing grant
 * answers true, a "Don't allow" from this run answers false, otherwise the
 * user is asked once (concurrent calls for the same pair share the prompt).
 */
export function ensurePreviewScriptConsent(pluginId: string, pluginName: string, origin: string): Promise<boolean> {
  if (hasPreviewScriptGrant(pluginId, origin)) {
    return Promise.resolve(true)
  }

  const key = pairKey(pluginId, origin)

  if ($sessionDenials.get().has(key)) {
    return Promise.resolve(false)
  }

  const inFlight = pending.get(key)

  if (inFlight) {
    return inFlight.promise
  }

  let resolve!: (allowed: boolean) => void
  const promise = new Promise<boolean>(done => (resolve = done))

  pending.set(key, { promise, resolve })
  $previewScriptConsentQueue.set([
    ...$previewScriptConsentQueue.get(),
    { id: nextRequestId++, origin, pluginId, pluginName }
  ])

  return promise
}

/**
 * A plugin left (uninstalled, its folder retired): drop every grant it holds
 * and refuse its open prompts, so a late "Allow" cannot re-grant a removed
 * plugin and a later plugin reusing the id starts from nothing.
 */
export function forgetPluginPreviewAccess(pluginId: string): void {
  revokePreviewScript(pluginId)

  const queue = $previewScriptConsentQueue.get()

  $previewScriptConsentQueue.set(queue.filter(entry => entry.pluginId !== pluginId))

  for (const request of queue) {
    if (request.pluginId === pluginId) {
      const key = pairKey(request.pluginId, request.origin)

      pending.get(key)?.resolve(false)
      pending.delete(key)
    }
  }
}

/** The dialog's answer for request `id`. Anything but an explicit allow —
 *  Esc, backdrop, the close button — arrives here as `false`. */
export function answerPreviewScriptConsent(id: number, allowed: boolean): void {
  const queue = $previewScriptConsentQueue.get()
  const request = queue.find(entry => entry.id === id)

  if (!request) {
    return
  }

  $previewScriptConsentQueue.set(queue.filter(entry => entry.id !== id))

  const key = pairKey(request.pluginId, request.origin)

  if (allowed) {
    grantPreviewScript(request.pluginId, request.origin)
  } else {
    $sessionDenials.set(new Set([...$sessionDenials.get(), key]))
  }

  pending.get(key)?.resolve(allowed)
  pending.delete(key)
}
