import { atom, computed, type ReadableAtom } from 'nanostores'

/** Latest live progress of a running tool call (`tool.progress`, from MCP `notifications/progress`). */
export interface ToolProgress {
  message: null | string
  percent: null | number
}

// Keyed by tool call id. Ephemeral by design: progress only exists while a call
// runs, so it is never persisted and the row drops it once the call completes.
export const $toolProgress = atom<Record<string, ToolProgress>>({})

const progressCache = new Map<string, ReadableAtom<ToolProgress | undefined>>()

export function $toolProgressFor(toolCallId: string): ReadableAtom<ToolProgress | undefined> {
  let cached = progressCache.get(toolCallId)

  if (!cached) {
    // MCP rows render per batched call as `<call id>:<index>`; progress is reported for the call.
    const callId = toolCallId.replace(/:\d+$/, '')
    cached = computed($toolProgress, rows => rows[toolCallId] ?? rows[callId])
    progressCache.set(toolCallId, cached)
  }

  return cached
}

export function setToolProgress(toolCallId: string, progress: ToolProgress): void {
  if (toolCallId) {
    $toolProgress.set({ ...$toolProgress.get(), [toolCallId]: progress })
  }
}

export function clearToolProgress(toolCallId: string): void {
  const rows = $toolProgress.get()

  if (toolCallId && rows[toolCallId]) {
    const { [toolCallId]: _done, ...rest } = rows
    $toolProgress.set(rest)
  }
}

/** `42% · refining mesh` — the text a running tool row shows beside its timer. */
export function formatToolProgress(progress: ToolProgress | undefined): string {
  if (!progress) {
    return ''
  }

  return [progress.percent === null ? '' : `${progress.percent}%`, progress.message ?? ''].filter(Boolean).join(' · ')
}
