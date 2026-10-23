import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'

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

export const $previewScriptGrants = persistentAtom<PreviewScriptGrants>(
  'hermes.desktop.pluginPreviewScriptGrants.v1',
  {},
  Codecs.json(sanitizeGrants)
)

/** "Don't allow" answers for this app run, as `${pluginId}\n${origin}`. */
const $sessionDenials = atom<ReadonlySet<string>>(new Set())

const pairKey = (pluginId: string, origin: string) => `${pluginId}\n${origin}`

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
  return $previewScriptGrants.get()[pluginId]?.includes(origin) ?? false
}

export function grantPreviewScript(pluginId: string, origin: string): void {
  const grants = $previewScriptGrants.get()
  const current = grants[pluginId] ?? []

  if (!current.includes(origin)) {
    $previewScriptGrants.set({ ...grants, [pluginId]: [...current, origin] })
  }
}

/** Revoke one origin, or every origin for the plugin when omitted. */
export function revokePreviewScript(pluginId: string, origin?: string): void {
  const { [pluginId]: current, ...rest } = $previewScriptGrants.get()

  if (!current) {
    return
  }

  const kept = origin === undefined ? [] : current.filter(entry => entry !== origin)

  $previewScriptGrants.set(kept.length > 0 ? { ...rest, [pluginId]: kept } : rest)
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
