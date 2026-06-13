import { atom, computed } from 'nanostores'

type Updater<T> = T | ((current: T) => T)

interface WritableStore<T> {
  get: () => T
  set: (value: T) => void
}

const DEFAULT_CONSOLE_HEIGHT = 240

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
      $logs.set([...$logs.get().slice(-199), { ...entry, id: ++nextLogId }])
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
