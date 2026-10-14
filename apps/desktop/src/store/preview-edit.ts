import { atom } from 'nanostores'

// Unsaved spot-editor drafts, keyed by `target.url`. The editor in
// `preview-file.tsx` is the sole writer. A file preview's body unmounts
// whenever its session leaves the screen (only live pages are retained) or
// the zone evicts an inactive tab, so the draft cannot live in the component:
// here it survives the unmount, a remount restores it, and closing its tab can
// see it and ask first (inspired by Claude Cowork's Files pane, which prompts
// before leaving a session with unsaved edits). Memory-only.
export interface PreviewDraft {
  /** The disk text the edit started from (the save-time conflict check). */
  baseline: string
  draft: string
  /** The connection/profile the edit belongs to; saving elsewhere is refused. */
  scope: string
}

const drafts = new Map<string, PreviewDraft>()

// URLs holding a draft, for reactive readers (tab chrome, close confirmation).
export const $dirtyPreviewUrls = atom<Record<string, true>>({})

export function setPreviewDraft(url: string, draft: null | PreviewDraft): void {
  if (!url) {
    return
  }

  const had = drafts.has(url)

  if (draft) {
    drafts.set(url, draft)
  } else {
    drafts.delete(url)
  }

  if (had === Boolean(draft)) {
    return
  }

  const next = { ...$dirtyPreviewUrls.get() }

  if (draft) {
    next[url] = true
  } else {
    delete next[url]
  }

  $dirtyPreviewUrls.set(next)
}

export function previewDraft(url: string): PreviewDraft | undefined {
  return drafts.get(url)
}

export function hasPreviewDraft(url: string): boolean {
  return drafts.has(url)
}

/** Tabs (ids) whose close is waiting on the discard confirmation, in the
 *  order they were asked; the dialog answers the first. */
export const $pendingPreviewDiscards = atom<readonly string[]>([])

export function queuePreviewDiscard(tabIds: readonly string[]): void {
  const queued = $pendingPreviewDiscards.get()
  const added = tabIds.filter(id => !queued.includes(id))

  if (added.length > 0) {
    $pendingPreviewDiscards.set([...queued, ...added])
  }
}

export function dequeuePreviewDiscard(tabId: string): void {
  $pendingPreviewDiscards.set($pendingPreviewDiscards.get().filter(id => id !== tabId))
}

/** Forget every draft whose url is not in `keep` (its tabs all closed). */
export function prunePreviewDrafts(keep: ReadonlySet<string>): void {
  for (const url of [...drafts.keys()]) {
    if (!keep.has(url)) {
      setPreviewDraft(url, null)
    }
  }
}
