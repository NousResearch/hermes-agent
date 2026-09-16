import { atom } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'
import { previewName } from '@/lib/preview-targets'

/**
 * Session-scoped feed of previewable artifacts (HTML files, localhost dev URLs)
 * a tool produced. Surfaced as compact links in the composer status stack —
 * NOT auto-opened and NOT a bulky inline card. Click opens the rail preview or
 * the browser; both are manual.
 *
 * Fed from the tool row itself (see tool-fallback.tsx) using the same detected
 * target the inline card used, so detection parity is exact.
 *
 * DISMISSAL IS DURABLE. The tool row registers its artifact from a mount
 * effect, and rows remount constantly — transcript virtualization while
 * scrolling, switching sessions and back, a reconnect that re-renders history.
 * Without a memory of what the user closed, every remount resurrected a
 * days-old chip the user had already dismissed. Dismissed targets are
 * remembered per session and persisted, so a closed artifact stays closed;
 * the transcript card for that turn remains the way to open it again.
 */
export interface PreviewArtifact {
  /** cwd captured at detection so a relative path still resolves on click. */
  cwd: string
  /** Dedupe key + display id (the raw target). */
  id: string
  label: string
  target: string
}

const MAX_PER_SESSION = 4

/** Bound on remembered dismissals across all sessions; oldest evicted first. */
const MAX_DISMISSED = 500

const DISMISSED_STORAGE_KEY = 'hermes.desktop.previewStatusDismissed.v1'

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})

/** `sid\u0000target` keys the user has dismissed, oldest first. */
export const $dismissedPreviewArtifacts = persistentAtom<string[]>(DISMISSED_STORAGE_KEY, [], {
  decode: raw => {
    const parsed: unknown = JSON.parse(raw)

    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []
  },
  encode: value => JSON.stringify(value)
})

const dismissalKey = (sid: string, id: string) => `${sid}\u0000${id}`

export function isPreviewArtifactDismissed(sid: string, id: string): boolean {
  return $dismissedPreviewArtifacts.get().includes(dismissalKey(sid, id))
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

  $previewStatusBySession.set({ ...current, [sid]: items })
}

/**
 * Record a detected artifact, newest last, capped. Idempotent: a target already
 * in the list keeps its slot (the tool row re-registers on every render, so this
 * must not churn the atom or reorder rows). A target the user dismissed in this
 * session is ignored — re-registration is a render artifact, not new work.
 */
export function recordPreviewArtifact(sid: string, target: string, cwd: string) {
  const raw = target.trim()

  if (!sid || !raw || isPreviewArtifactDismissed(sid, raw)) {
    return
  }

  const list = $previewStatusBySession.get()[sid] ?? []

  if (list.some(item => item.id === raw)) {
    return
  }

  writePreviews(sid, [...list, { cwd, id: raw, label: previewName(raw), target: raw }].slice(-MAX_PER_SESSION))
}

/** Remove the row now AND remember the choice, so the tool row that fed it
 *  cannot bring it back on its next mount. */
export function dismissPreviewArtifact(sid: string, id: string) {
  const list = $previewStatusBySession.get()[sid]

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }

  const key = dismissalKey(sid, id)
  const remembered = $dismissedPreviewArtifacts.get().filter(item => item !== key)

  $dismissedPreviewArtifacts.set([...remembered, key].slice(-MAX_DISMISSED))
}

/** Forget a dismissal so the artifact may surface again. */
export function undismissPreviewArtifact(sid: string, id: string) {
  const key = dismissalKey(sid, id)
  const current = $dismissedPreviewArtifacts.get()

  if (current.includes(key)) {
    $dismissedPreviewArtifacts.set(current.filter(item => item !== key))
  }
}

/** Drop the session's rows. Used when a timeline is discarded (rewind/branch,
 *  tile teardown); dismissals are left alone so a surviving row that remounts
 *  still honors the user's earlier close. */
export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
