import { atom } from 'nanostores'

/**
 * Per-session input history browse state.
 *
 * The user-text ring is **derived from the live session messages** on each
 * keypress — it is not mirrored anywhere. This keeps a single source of truth
 * and avoids the entire class of seeding/dedup bugs that come from trying to
 * keep a parallel ring in sync with submit/queue/voice paths.
 *
 * We only persist the cursor and the saved draft:
 *   - `cursor` — index into the derived user-text ring (0 = newest, larger = older).
 *     `-1` means "not browsing".
 *   - `draftSnapshot` — the composer text at the moment the user started
 *     browsing, so ArrowDown back to the "present" restores it.
 */
export interface SessionBrowseState {
  cursor: number
  draftSnapshot: string
}

const $perSessionBrowse = atom<Record<string, SessionBrowseState>>({})

function ensure(sessionId: string): SessionBrowseState {
  const all = { ...$perSessionBrowse.get() }
  let s = all[sessionId]

  if (!s) {
    s = { cursor: -1, draftSnapshot: '' }
    all[sessionId] = s
    $perSessionBrowse.set(all)
  }

  return s
}

function persist() {
  $perSessionBrowse.set({ ...$perSessionBrowse.get() })
}

function valid(sessionId: string | null | undefined): sessionId is string {
  return typeof sessionId === 'string' && sessionId.length > 0
}

/**
 * Derive the user-text ring (newest first) from session messages.
 * The caller is responsible for providing already-session-scoped messages.
 */
export function deriveUserHistory<T extends { role: string }>(
  messages: readonly T[],
  getText: (m: T) => string
): string[] {
  const out: string[] = []

  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]!

    if (m.role !== 'user') {
      continue
    }

    const t = getText(m).trim()

    if (t) {
      out.push(t)
    }
  }

  return out
}

/**
 * Start browsing backward, or step to the next older entry.
 * Returns the text to place in the composer, or null if already at the oldest
 * entry (or the ring is empty).
 */
export function browseBackward(
  sessionId: string | null | undefined,
  currentDraft: string,
  history: readonly string[]
): string | null {
  if (!valid(sessionId) || history.length === 0) {
    return null
  }

  const s = ensure(sessionId)

  if (s.cursor === -1) {
    s.draftSnapshot = currentDraft
    s.cursor = 0
  } else if (s.cursor < history.length - 1) {
    s.cursor += 1
  } else {
    return null
  }

  persist()

  return history[s.cursor]!
}

/**
 * Browse forward toward the present. When reaching the "newest" entry the
 * saved draft is restored and the cursor resets.
 */
export function browseForward(
  sessionId: string | null | undefined,
  history: readonly string[]
): { text: string; returnedToPresent: boolean } | null {
  if (!valid(sessionId)) {
    return null
  }

  const s = ensure(sessionId)

  if (s.cursor === -1) {
    return null
  }

  if (s.cursor > 0) {
    s.cursor -= 1
    persist()

    return { text: history[s.cursor]!, returnedToPresent: false }
  }

  // At newest; moving forward restores the saved draft.
  const text = s.draftSnapshot
  s.cursor = -1
  s.draftSnapshot = ''
  persist()

  return { text, returnedToPresent: true }
}

/** Clear browse state for a session (e.g. on session switch or new submit). */
export function resetBrowseState(sessionId: string | null | undefined) {
  if (!valid(sessionId)) {
    return
  }

  const all = { ...$perSessionBrowse.get() }
  const existing = all[sessionId]

  if (!existing) {
    return
  }

  all[sessionId] = { cursor: -1, draftSnapshot: '' }
  $perSessionBrowse.set(all)
}

/** True if the user is currently browsing history for this session. */
export function isBrowsingHistory(sessionId: string | null | undefined): boolean {
  if (!valid(sessionId)) {
    return false
  }

  const s = $perSessionBrowse.get()[sessionId]

  return s ? s.cursor >= 0 : false
}

export { $perSessionBrowse }
