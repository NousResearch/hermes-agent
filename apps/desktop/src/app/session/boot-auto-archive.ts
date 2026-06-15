export interface BootAutoArchiveResult {
  archived: number
  ok: boolean
  skipped?: boolean
}

export interface BootAutoArchiveState {
  started: boolean
}

interface ScheduleBootAutoArchiveOptions {
  autoArchive: (preserveIds: string[]) => Promise<BootAutoArchiveResult>
  onArchived: (result: BootAutoArchiveResult) => void
  onError?: (error: unknown) => void
  preserveIds: Iterable<string>
  schedule?: (callback: () => void, delayMs: number) => unknown
  state: BootAutoArchiveState
}

export function scheduleBootAutoArchiveOnce({
  autoArchive,
  onArchived,
  onError,
  preserveIds,
  schedule = (callback, delayMs) => window.setTimeout(callback, delayMs),
  state
}: ScheduleBootAutoArchiveOptions): boolean {
  if (state.started) {
    return false
  }

  state.started = true
  const preserved = Array.from(new Set([...preserveIds].filter(Boolean)))

  schedule(() => {
    void autoArchive(preserved)
      .then(result => {
        if (result.archived > 0) {
          onArchived(result)
        }
      })
      .catch(error => {
        onError?.(error)
      })
  }, 0)

  return true
}
