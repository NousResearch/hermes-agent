import { atom } from 'nanostores'

export const $silencedContextCacheSessionIds = atom<ReadonlySet<string>>(new Set())

export function silenceContextCacheWarningForSession(sessionId: string): void {
  if (!sessionId) {return}
  const current = $silencedContextCacheSessionIds.get()

  if (!current.has(sessionId)) {
    const next = new Set(current)
    next.add(sessionId)
    $silencedContextCacheSessionIds.set(next)
  }
}

export function clearContextCacheWarningSilence(sessionId: string): void {
  if (!sessionId) {return}
  const current = $silencedContextCacheSessionIds.get()

  if (current.has(sessionId)) {
    const next = new Set(current)
    next.delete(sessionId)
    $silencedContextCacheSessionIds.set(next)
  }
}

export function isContextCacheWarningSilenced(sessionId: string | undefined | null): boolean {
  if (!sessionId) {return false}

  return $silencedContextCacheSessionIds.get().has(sessionId)
}

export function resetSilencedContextCacheWarnings(): void {
  $silencedContextCacheSessionIds.set(new Set())
}
