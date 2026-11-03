import { atom, computed } from 'nanostores'

// Per-session flag while auto-compaction runs mid-turn. Without it the
// transcript looks like it reset; per-session so a background chat can't
// clobber the foreground view.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''
const hasSession = (sessions: Record<string, true>, key: string): boolean => Object.hasOwn(sessions, key)

export const $compactingSessions = atom<Record<string, true>>({})

/** Is `sessionId` compacting? Per-session because a transcript may be a tile,
 *  and a tile must never wear the primary chat's compaction state. */
export function sessionCompacting(sessionId: null | string) {
  return computed($compactingSessions, sessions => hasSession(sessions, keyFor(sessionId)))
}

export function setSessionCompacting(sessionId: string | null | undefined, active: boolean): void {
  const key = keyFor(sessionId)
  const sessions = $compactingSessions.get()

  if (active) {
    for (const observe of terminalObservers) {observe(key, null)}

    if (hasSession(sessions, key)) {
      return
    }

    $compactingSessions.set({ ...sessions, [key]: true })

    return
  }

  if (!hasSession(sessions, key)) {
    return
  }

  const next = { ...sessions }
  delete next[key]
  $compactingSessions.set(next)
}

// A manual /compress whose RPC answers `pending` (#97948) returns before the
// compute host is done, so the handler never reaches the branch that renders
// the summary. Claim the session here and let the `compacted`/`ready` edge
// announce the completion instead — that edge is the only one the client gets.
const deferredCompressSessions = new Set<string>()

export function markCompressDeferred(sessionId: string | null | undefined): void {
  deferredCompressSessions.add(keyFor(sessionId))
}

/** Consume the claim. True only for the first terminal edge, so a later
 *  auto-compaction on the same session cannot replay the notice. */
export function takeCompressDeferred(sessionId: string | null | undefined): boolean {
  return deferredCompressSessions.delete(keyFor(sessionId))
}

interface CompactionTerminal {
  kind: 'compacted' | 'error' | 'ready'
  text: string
}

type TerminalObserver = (sessionId: string, terminal: CompactionTerminal | null) => void
const terminalObservers = new Set<TerminalObserver>()

/** Watch only this request's lifetime. Recovery may reveal a different runtime
 *  in the reply, so retain edges by runtime and inspect only the resolved id. */
export function observeCompactionTerminal() {
  const terminals = new Map<string, CompactionTerminal>()
  const started = new Set<string>()

  const observe: TerminalObserver = (sessionId, terminal) => {
    if (!terminal) {
      started.add(sessionId)

      return
    }

    // Ready can belong to an earlier control. Require a compaction start or
    // an explicit compacted/error edge before treating it as this request's end.
    if (terminal.kind === 'ready' && !started.has(sessionId)) {return}
    const previous = terminals.get(sessionId)

    // A generic ready edge must not erase an error or a positive completion.
    if (previous?.kind !== 'error' && (terminal.kind !== 'ready' || !previous)) {
      terminals.set(sessionId, terminal)
    }
  }

  terminalObservers.add(observe)

  return {
    get: (sessionId: string) => terminals.get(sessionId),
    dispose: () => {
      terminalObservers.delete(observe)
    }
  }
}

export function recordCompactionTerminal(sessionId: string, terminal: CompactionTerminal): void {
  for (const observe of terminalObservers) {
    observe(sessionId, terminal)
  }
}

/** Clear compaction only when the gateway proves the turn resumed or ended. */
export function reconcileSessionCompacting(
  sessionId: string | null | undefined,
  _evidence: 'resumed' | 'terminal'
): void {
  setSessionCompacting(sessionId, false)
}
