import { atom, computed, onStop, type ReadableAtom } from 'nanostores'

/** Latest live progress of a running tool call (`tool.progress`, from MCP `notifications/progress`). */
export interface ToolProgress {
  message: null | string
  percent: null | number
}

// Keyed by `<session>\u0000<tool call id>`: providers reuse call ids (`call_0`) across chats, so the
// call id alone is not an owner. Ephemeral by design: progress only exists while a call runs, so it is
// never persisted, the row drops it on tool.complete and the session's turn end sweeps any leftover.
export const $toolProgress = atom<Record<string, ToolProgress>>({})

const progressKey = (sessionId: string, toolCallId: string) => `${sessionId}\u0000${toolCallId}`

const progressCache = new Map<string, ReadableAtom<ToolProgress | undefined>>()

export function $toolProgressFor(sessionId: string, toolCallId: string): ReadableAtom<ToolProgress | undefined> {
  const key = progressKey(sessionId, toolCallId)
  let cached = progressCache.get(key)

  if (!cached) {
    // MCP rows render per batched call as `<call id>:<index>`; progress is reported for the call.
    const callKey = progressKey(sessionId, toolCallId.replace(/:\d+$/, ''))
    const store = computed($toolProgress, rows => rows[key] ?? rows[callKey])
    // Evict once no row listens, so finished or never-progressing rows do not pin a store forever.
    onStop(store, () => {
      if (progressCache.get(key) === store) {
        progressCache.delete(key)
      }
    })
    progressCache.set(key, store)
    cached = store
  }

  return cached
}

export function setToolProgress(sessionId: string, toolCallId: string, progress: ToolProgress): void {
  if (sessionId && toolCallId) {
    $toolProgress.set({ ...$toolProgress.get(), [progressKey(sessionId, toolCallId)]: progress })
  }
}

export function clearToolProgress(sessionId: string, toolCallId: string): void {
  const rows = $toolProgress.get()
  const key = progressKey(sessionId, toolCallId)

  if (key in rows) {
    const { [key]: _done, ...rest } = rows
    $toolProgress.set(rest)
  }
}

/** Drop every progress entry a session still holds (its turn ended; a missed tool.complete never lingers). */
export function clearSessionToolProgress(sessionId: string): void {
  const prefix = `${sessionId}\u0000`
  const rows = $toolProgress.get()
  const kept = Object.fromEntries(Object.entries(rows).filter(([key]) => !key.startsWith(prefix)))

  if (Object.keys(kept).length !== Object.keys(rows).length) {
    $toolProgress.set(kept)
  }
}

/** `42% · refining mesh` — the text a running tool row shows beside its timer. */
export function formatToolProgress(progress: ToolProgress | undefined): string {
  if (!progress) {
    return ''
  }

  return [progress.percent === null ? '' : `${progress.percent}%`, progress.message ?? ''].filter(Boolean).join(' · ')
}
