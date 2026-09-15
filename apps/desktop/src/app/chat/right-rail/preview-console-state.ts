import { atom, computed } from 'nanostores'

type Updater<T> = T | ((current: T) => T)

interface WritableStore<T> {
  get: () => T
  set: (value: T) => void
}

const DEFAULT_CONSOLE_HEIGHT = 240
export const PREVIEW_CONSOLE_MAX_ENTRIES = 200
export const PREVIEW_CONSOLE_MAX_MESSAGE_CHARS = 64 * 1024
export const PREVIEW_CONSOLE_MAX_SOURCE_CHARS = 4 * 1024
export const PREVIEW_CONSOLE_MAX_TOTAL_CHARS = 512 * 1024
const PREVIEW_CONSOLE_TRUNCATION_MARKER = '\n… [truncated]'

export interface ConsoleEntry {
  id: number
  level: number
  line?: number
  message: string
  source?: string
}

export interface ConsoleEntryInput {
  level: number
  line?: number
  message: string
  source?: string
}

function updateAtom<T>(store: WritableStore<T>, next: Updater<T>) {
  store.set(typeof next === 'function' ? (next as (current: T) => T)(store.get()) : next)
}

function truncateConsoleText(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value
  }

  const marker = PREVIEW_CONSOLE_TRUNCATION_MARKER.slice(0, maxChars)
  const kept = Math.max(0, maxChars - marker.length)
  let prefix = value.slice(0, kept)

  // Avoid manufacturing a lone UTF-16 high surrogate at the cut. Replacing
  // the partial code point keeps the exact code-unit budget and produces a
  // display-safe string for React/Electron.
  const last = prefix.charCodeAt(prefix.length - 1)

  if (last >= 0xd800 && last <= 0xdbff) {
    prefix = `${prefix.slice(0, -1)}\ufffd`
  }

  return `${prefix}${marker}`
}

function consoleEntryChars(entry: ConsoleEntry): number {
  return entry.message.length + (entry.source?.length ?? 0)
}

function fitConsoleHistory(entries: ConsoleEntry[]): ConsoleEntry[] {
  let kept = entries.slice(-PREVIEW_CONSOLE_MAX_ENTRIES)
  let chars = kept.reduce((total, entry) => total + consoleEntryChars(entry), 0)
  let drop = 0

  // Each field is truncated far below the aggregate cap, so the newest entry
  // always fits. Evict oldest-first to preserve the most useful recent logs.
  while (drop < kept.length - 1 && chars > PREVIEW_CONSOLE_MAX_TOTAL_CHARS) {
    chars -= consoleEntryChars(kept[drop])
    drop++
  }

  if (drop > 0) {
    kept = kept.slice(drop)
  }

  return kept
}

export function createPreviewConsoleState() {
  const $height = atom(DEFAULT_CONSOLE_HEIGHT)
  const $logs = atom<ConsoleEntry[]>([])
  const $logCount = computed($logs, logs => logs.length)
  const $open = atom(false)
  const $selectedLogIds = atom<ReadonlySet<number>>(new Set())
  let nextLogId = 0

  return {
    $height,
    $logCount,
    $logs,
    $open,
    $selectedLogIds,
    append(entry: ConsoleEntryInput) {
      const nextEntry: ConsoleEntry = {
        ...entry,
        id: ++nextLogId,
        message: truncateConsoleText(entry.message, PREVIEW_CONSOLE_MAX_MESSAGE_CHARS),
        ...(entry.source === undefined
          ? {}
          : { source: truncateConsoleText(entry.source, PREVIEW_CONSOLE_MAX_SOURCE_CHARS) })
      }
      const logs = fitConsoleHistory([...$logs.get(), nextEntry])

      $logs.set(logs)

      const selected = $selectedLogIds.get()

      if (selected.size > 0) {
        const retained = new Set(logs.map(log => log.id))
        const nextSelected = new Set([...selected].filter(id => retained.has(id)))

        if (nextSelected.size !== selected.size) {
          $selectedLogIds.set(nextSelected)
        }
      }
    },
    clear() {
      $logs.set([])
      $selectedLogIds.set(new Set())
    },
    clearSelection() {
      if ($selectedLogIds.get().size === 0) {
        return
      }

      $selectedLogIds.set(new Set())
    },
    reset() {
      nextLogId = 0
      $logs.set([])
      $selectedLogIds.set(new Set())
    },
    setHeight(next: Updater<number>) {
      updateAtom($height, next)
    },
    setOpen(next: Updater<boolean>) {
      updateAtom($open, next)
    },
    toggleSelection(id: number) {
      const next = new Set($selectedLogIds.get())

      if (!next.delete(id)) {
        next.add(id)
      }

      $selectedLogIds.set(next)
    }
  }
}

export type PreviewConsoleState = ReturnType<typeof createPreviewConsoleState>
