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

export function clearSessionProviderWait(sessionId: string | null | undefined): void {
  setSessionProviderWait(sessionId, '')
}

export function clearAllProviderWaits(): void {
  $providerWaitSessions.set({})
}

/** Only the core's explained wait/reconnect frames belong in Desktop's status
 * row. Generic kawaii spinner rewrites remain presentation noise. */
export function providerWaitText(text: string): string {
  const value = text.trim()

  return /^(?:⏳|⚠|↻|⚙)\s*(?:waiting on|loading|processing prompt|no (?:output|response)|model returned)/i.test(value)
    ? value
    : ''
}

/** Parse a managed-local progress frame into bar-renderable parts, or null
 * for every other wait frame. Two shapes, both minted by the backend's
 * _managed_local_load_notice (the percents are real — per-tensor load
 * callback / live prefill counter — so a determinate bar is honest):
 *   "⏳ loading <model> into memory — 43% …"  -> kind: 'load'
 *   "⚙ processing prompt — 31%"               -> kind: 'prefill'
 * A percentless prefill frame ("⚙ processing prompt") parses with
 * percent: null and renders as label-only, no fake bar. */
export function parseModelLoadWait(
  text: string
): null | { kind: 'load' | 'prefill'; model: string; percent: null | number } {
  const value = text.trim()
  const load = /^⏳\s*loading\s+(.+?)\s+into memory\s+—\s+(\d{1,3})%/i.exec(value)

  if (load) {
    return {
      kind: 'load',
      model: load[1],
      percent: Math.max(0, Math.min(100, Number(load[2])))
    }
  }

  const prefill = /^⚙\s*processing prompt(?:\s+—\s+(\d{1,3})%)?/i.exec(value)

  if (prefill) {
    return {
      kind: 'prefill',
      model: '',
      percent: prefill[1] === undefined ? null : Math.max(0, Math.min(100, Number(prefill[1])))
    }
  }

  return null
}
