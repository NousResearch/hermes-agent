import { atom } from 'nanostores'

import { previewName } from '@/lib/preview-targets'

/**
 * Session-scoped feed of previewable artifacts (HTML files, localhost dev URLs)
 * a tool produced. Surfaced as compact links in the composer status stack —
 * NOT auto-opened and NOT a bulky inline card. Click opens the rail preview or
 * the browser; both are manual.
 *
 * Fed from the authoritative thread collection using the same lightweight
 * target extractor as tool rendering, so paginated rows remain discoverable
 * without mounting their expensive visual subtree.
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
const clearGenerationBySession = new Map<string, number>()

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})

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
 * Record detected artifacts in one atom update, newest last and capped.
 * Idempotent: a target already in the list keeps its slot (transcript
 * reconciliation may see it repeatedly, so this must not churn or reorder).
 */
export function recordPreviewArtifacts(sid: string, targets: readonly string[], cwd: string) {
  if (!sid) {
    return
  }

  let next = $previewStatusBySession.get()[sid] ?? []
  let changed = false

  for (const target of targets) {
    const raw = target.trim()

    if (!raw || next.some(item => item.id === raw)) {
      continue
    }

    changed = true
    next = [...next, { cwd, id: raw, label: previewName(raw), target: raw }].slice(-MAX_PER_SESSION)
  }

  if (!changed) {
    return
  }

  writePreviews(sid, next)
}

export function recordPreviewArtifact(sid: string, target: string, cwd: string) {
  recordPreviewArtifacts(sid, [target], cwd)
}

export function dismissPreviewArtifact(sid: string, id: string) {
  const list = $previewStatusBySession.get()[sid]

  if (list) {
    writePreviews(
      sid,
      list.filter(item => item.id !== id)
    )
  }
}

export function clearPreviewArtifacts(sid: string) {
  clearGenerationBySession.set(sid, (clearGenerationBySession.get(sid) ?? 0) + 1)
  writePreviews(sid, [])
}

export function getPreviewClearGeneration(sid: string): number {
  return clearGenerationBySession.get(sid) ?? 0
}
