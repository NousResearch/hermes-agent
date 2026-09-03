import { atom } from 'nanostores'

import type { MediaKind } from '@/lib/media'

/**
 * Durable metadata for one media deliverable.
 *
 * This is the D1 event payload's renderer-facing projection (`{path, kind,
 * mime, size, session_id, origin}`) plus the bookkeeping fields the desktop
 * stamps on receipt. Every field is optional-tolerant: a metadata row renders
 * a card even when the server sends only `path`.
 */
export interface MediaDeliverableMeta {
  path: string
  kind?: MediaKind
  mime?: string
  size?: number
  origin?: string
  receivedAt: number
}

const MAX_REGISTRY_ENTRIES = 600

const KINDS: readonly string[] = ['audio', 'file', 'image', 'video']

const $records = atom(new Map<string, MediaDeliverableMeta>())

function publish(): void {
  $records.set(new Map($records.get()))
}

/**
 * Record (or refresh) metadata for one deliverable. Unknown `kind` values
 * downgrade to `file`; later rows for the same path win (the gateway re-emits
 * on each turn that re-tags the same file). Returns false for garbage rows —
 * bad metadata never throws, it just doesn't register.
 */
export function recordMediaDeliverable(payload: unknown, receivedAt: number = Date.now()): boolean {
  if (!payload || typeof payload !== 'object') {
    return false
  }

  const row = payload as Record<string, unknown>
  const path = typeof row.path === 'string' ? row.path.trim() : ''

  // Control characters in a path are always hostile (or corrupt) input. Char
  // codes instead of a regex literal — the no-control-regex lint rule forbids
  // the literal form even for defensive validation.
  const hasControlChars = [...path].some(ch => {
    const code = ch.charCodeAt(0)

    return code < 32 || (code >= 127 && code < 160)
  })

  if (!path || path.length > 4096 || hasControlChars) {
    return false
  }

  const rawKind = typeof row.kind === 'string' ? row.kind : ''
  const kind = (KINDS.includes(rawKind) ? rawKind : 'file') as MediaKind

  const record: MediaDeliverableMeta = {
    kind,
    path,
    receivedAt,
    ...(typeof row.mime === 'string' && row.mime ? { mime: row.mime } : {}),
    ...(typeof row.origin === 'string' && row.origin ? { origin: row.origin } : {}),
    ...(typeof row.size === 'number' && Number.isFinite(row.size) && row.size >= 0 ? { size: row.size } : {})
  }

  $records.get().set(path, record)

  // Bounded ring: the registry is a live-view cache, not a transcript. When
  // over budget, drop the OLDEST receipts first — recent turns are the ones
  // still rendering live cards.
  if ($records.get().size > MAX_REGISTRY_ENTRIES) {
    const entries = [...$records.get().entries()].sort((a, b) => a[1].receivedAt - b[1].receivedAt)

    for (const [stalePath] of entries.slice(0, $records.get().size - MAX_REGISTRY_ENTRIES)) {
      $records.get().delete(stalePath)
    }
  }

  publish()

  return true
}

/** Metadata for one path, if this desktop has seen a deliverable event for it. */
export function mediaCardMeta(path: string): MediaDeliverableMeta | null {
  const meta = $records.get().get(path)

  return meta ? { ...meta } : null
}

/**
 * Re-key a registry entry when the renderer normalizes a ref (e.g. stripping a
 * `file://` prefix): `aliasPath` (the alternate spelling) becomes readable via
 * the row registered under `rowPath`. The copied row keeps its ORIGINAL path
 * so prune allowlists keyed on raw gateway refs keep matching. No-op when the
 * alias already exists or no row backs it.
 */
export function aliasMediaCardMeta(aliasPath: string, rowPath: string): void {
  const map = $records.get()
  const row = map.get(rowPath)

  if (aliasPath !== rowPath && row && !map.has(aliasPath)) {
    map.set(aliasPath, { ...row })
    publish()
  }
}

/**
 * Drop entries whose path is not in `keepPaths` (aliases included). Called by
 * the chat-message hydration/reconciliation pass with the media refs of the
 * messages being re-rendered, so a long-lived desktop window cannot accumulate
 * rows for files whose transcripts are long gone. With no argument the
 * registry is fully reset (used when a session closes and between tests).
 */
export function pruneMediaDeliverables(keepPaths?: readonly string[]): void {
  if (!keepPaths) {
    $records.get().clear()
    publish()

    return
  }

  const keep = new Set(keepPaths)
  const map = $records.get()

  const dropped = [...map.keys()].filter(path => {
    if (keep.has(path)) {
      return false
    }

    // Alias spellings (file://… of a kept raw ref) survive with their target.
    const aliased = path.startsWith('file:') ? map.get(path) : undefined

    return !(aliased && keep.has(aliased.path))
  })

  if (dropped.length === 0) {
    return
  }

  for (const path of dropped) {
    map.delete(path)
  }

  publish()
}

/** Test seam: wipe the registry between tests. */
export function resetMediaDeliverables(): void {
  pruneMediaDeliverables()
}

/**
 * Seed the registry from the history media projection (D5).
 *
 * A reopened transcript has no `media.deliverable` events in memory — the
 * in-memory ring died with the old window — so the renderer falls back to
 * capture-time href sizes and blank cards when metadata is missing. The
 * server now derives refs from stored history (`include_media=true`), and
 * this funnels each row into the SAME registry the live events write, before
 * `toChatMessages` renders parts (`renderMediaTags` → `mediaCardMeta` reads
 * it at render time). Existing-file rows land with `receivedAt: 0` — older
 * than any live receipt, so the bounded prune keeps live rows first.
 *
 * Garbage rows (missing/hostile path) are skipped silently — record does
 * the validation, and history rendering must never throw. Returns the number
 * of rows registered.
 */
export function seedMediaDeliverablesFromHistory(refs: unknown): number {
  if (!Array.isArray(refs)) {
    return 0
  }

  let seeded = 0

  for (const ref of refs) {
    if (recordMediaDeliverable(ref, 0)) {
      seeded += 1
    }
  }

  return seeded
}
