import { atom } from 'nanostores'

import { activeConnectionScopeSuffix } from '@/lib/connection-scoped'
import { previewName } from '@/lib/preview-targets'
import { readJson, writeJson } from '@/lib/storage'

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
  /** Dedupe key + display id (the raw target). */
  id: string
  label: string
  target: string
}

const MAX_PER_SESSION = 4
const DISMISSED_PREVIEWS_KEY = 'hermes.desktop.previewDismissals.v1'

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})

type DismissedPreviewIds = Record<string, string[]>

function dismissedPreviewsKey(): string {
  return `${DISMISSED_PREVIEWS_KEY}${activeConnectionScopeSuffix()}`
}

function readDismissedPreviewIds(): DismissedPreviewIds {
  const value = readJson<unknown>(dismissedPreviewsKey())

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value).flatMap(([sid, ids]) => {
      if (!Array.isArray(ids)) {
        return []
      }

      const strings = ids.filter((id): id is string => typeof id === 'string' && id.length > 0)
      return strings.length > 0 ? [[sid, strings]] : []
    })
  )
}

function rememberDismissedPreview(sid: string, id: string): void {
  const dismissed = readDismissedPreviewIds()
  const ids = dismissed[sid] ?? []

  if (ids.includes(id)) {
    return
  }

  writeJson(dismissedPreviewsKey(), { ...dismissed, [sid]: [...ids, id] })
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
 * must not churn the atom or reorder rows).
 */
export function recordPreviewArtifact(sid: string, target: string, cwd: string, dismissalSid = sid) {
  const raw = target.trim()

  if (!sid || !raw) {
    return
  }

  const list = $previewStatusBySession.get()[sid] ?? []

  if (list.some(item => item.id === raw)) {
    return
  }

  if (readDismissedPreviewIds()[dismissalSid]?.includes(raw)) {
    return
  }

  writePreviews(sid, [...list, { cwd, id: raw, label: previewName(raw), target: raw }].slice(-MAX_PER_SESSION))
}

export function dismissPreviewArtifact(sid: string, id: string, dismissalSid = sid) {
  const list = $previewStatusBySession.get()[sid]

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }

  if (dismissalSid && id) {
    rememberDismissedPreview(dismissalSid, id)
  }
}

export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
