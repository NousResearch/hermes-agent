import { atom, computed } from 'nanostores'

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $providerWaitSessions = atom<Record<string, string>>({})

export function sessionProviderWait(sessionId: null | string) {
  return computed($providerWaitSessions, sessions => sessions[keyFor(sessionId)] ?? '')
}

export function setSessionProviderWait(sessionId: string | null | undefined, text: string): void {
  const key = keyFor(sessionId)
  const sessions = $providerWaitSessions.get()
  const nextText = text.trim()

  if (!nextText) {
    if (!(key in sessions)) {
      return
    }

    const next = { ...sessions }
    delete next[key]
    $providerWaitSessions.set(next)

    return
  }

  if (sessions[key] === nextText) {
    return
  }

  $providerWaitSessions.set({ ...sessions, [key]: nextText })
}

/** Only the core's explained wait/reconnect frames belong in Desktop's status
 * row. Generic kawaii spinner rewrites remain presentation noise. */
export function providerWaitText(text: string): string {
  const value = text.trim()

  return /^(?:⏳|⚠|↻)\s*(?:waiting on|no (?:output|response)|model returned)/i.test(value) ? value : ''
}
