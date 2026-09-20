import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $historyArrowsEnabled,
  $perSessionBrowse,
  browseBackward,
  browseForward,
  deriveUserHistory,
  isBrowsingHistory,
  resetBrowseState,
  setHistoryArrowsEnabled
} from './composer-input-history'

const ARROWS_KEY = 'hermes.desktop.composerHistory.arrowsEnabled'

const SESSION_A = 'session-a'
const SESSION_B = 'session-b'

// Newest-first user text ring, what the caller passes to browse*.
const HISTORY = ['third', 'second', 'first']

const MSG = (role: string, text: string) => ({ id: '', role, text })

beforeEach(() => {
  $perSessionBrowse.set({})
})

describe('deriveUserHistory', () => {
  it('returns user messages newest-first with empty/whitespace skipped', () => {
    const messages = [MSG('user', '   '), MSG('assistant', 'hi'), MSG('user', 'first'), MSG('user', 'second')]

    expect(deriveUserHistory(messages, m => m.text)).toEqual(['second', 'first'])
  })
})

describe('browseBackward', () => {
  it('returns null when history is empty', () => {
    expect(browseBackward(SESSION_A, '', [])).toBeNull()
  })

  it('returns the most recent entry on first press and saves the draft', () => {
    const result = browseBackward(SESSION_A, 'unsent draft', HISTORY)

    expect(result).toBe('third')
    expect($perSessionBrowse.get()[SESSION_A]!.draftSnapshot).toBe('unsent draft')
  })

  it('moves to older entries on subsequent presses and stops at the oldest', () => {
    expect(browseBackward(SESSION_A, '', HISTORY)).toBe('third')
    expect(browseBackward(SESSION_A, '', HISTORY)).toBe('second')
    expect(browseBackward(SESSION_A, '', HISTORY)).toBe('first')
    expect(browseBackward(SESSION_A, '', HISTORY)).toBeNull()
  })

  it('uses caller-provided history, not a mirrored ring', () => {
    // The store never owns the ring — the caller passes it every press.
    // If the ring changes between presses (e.g. a new message was sent),
    // the next press sees the updated ring and the cursor continues
    // from where it was within it.
    expect(browseBackward(SESSION_A, '', ['youngest', 'older'])).toBe('youngest')

    // Caller added a new message; ring is now [brand-new, youngest, older].
    // Cursor was at 0, next press advances to 1 -> "youngest".
    expect(browseBackward(SESSION_A, '', ['brand-new', 'youngest', 'older'])).toBe('youngest')

    // One more press -> "older".
    expect(browseBackward(SESSION_A, '', ['brand-new', 'youngest', 'older'])).toBe('older')
  })
})

describe('browseForward', () => {
  it('returns null when not browsing', () => {
    expect(browseForward(SESSION_A, HISTORY)).toBeNull()
  })

  it('moves toward the present', () => {
    browseBackward(SESSION_A, 'draft', HISTORY) // cursor 0 -> 'third'
    browseBackward(SESSION_A, '', HISTORY) // cursor 1 -> 'second'

    expect(browseForward(SESSION_A, HISTORY)).toEqual({
      text: 'third',
      returnedToPresent: false
    })
  })

  it('restores the saved draft and resets when reaching the present', () => {
    browseBackward(SESSION_A, 'my original draft', HISTORY)

    const result = browseForward(SESSION_A, HISTORY)

    expect(result).toEqual({ text: 'my original draft', returnedToPresent: true })
    expect(isBrowsingHistory(SESSION_A)).toBe(false)
  })
})

describe('per-session isolation', () => {
  it('tracks cursor and draft independently per session', () => {
    browseBackward(SESSION_A, 'draft-a', HISTORY)
    browseBackward(SESSION_A, '', HISTORY) // older

    browseBackward(SESSION_B, 'draft-b', HISTORY)

    const a = $perSessionBrowse.get()[SESSION_A]!
    const b = $perSessionBrowse.get()[SESSION_B]!

    expect(a.cursor).toBe(1)
    expect(a.draftSnapshot).toBe('draft-a')
    expect(b.cursor).toBe(0)
    expect(b.draftSnapshot).toBe('draft-b')
  })
})

describe('resetBrowseState', () => {
  it('clears cursor and draft snapshot', () => {
    browseBackward(SESSION_A, 'draft', HISTORY)
    resetBrowseState(SESSION_A)

    const s = $perSessionBrowse.get()[SESSION_A]!

    expect(s.cursor).toBe(-1)
    expect(s.draftSnapshot).toBe('')
  })
})

describe('session switch behavior', () => {
  it('resets the previous session cursor and lets the new session derive its own ring', () => {
    // Session A: user browsed into the past
    browseBackward(SESSION_A, '', HISTORY)
    expect(isBrowsingHistory(SESSION_A)).toBe(true)

    // Caller switches to session B; resets A's browse state
    resetBrowseState(SESSION_A)

    // Session B's ring is derived from B's messages, not A's
    const sessionBMessages = [MSG('user', 'hello-b'), MSG('user', 'world-b')]
    const sessionBHistory = deriveUserHistory(sessionBMessages, m => m.text)

    expect(browseBackward(SESSION_B, '', sessionBHistory)).toBe('world-b')
    expect(browseBackward(SESSION_B, '', sessionBHistory)).toBe('hello-b')
    expect(isBrowsingHistory(SESSION_A)).toBe(false)
  })
})

describe('arrow-history preference', () => {
  beforeEach(() => {
    window.localStorage.removeItem(ARROWS_KEY)
  })

  it('defaults to on when nothing is stored', async () => {
    // Reload-based so the assertion reads the seed of a fresh module, not the
    // atom captured whenever this file happened to be imported.
    vi.resetModules()
    const reloaded = await import('./composer-input-history')

    expect(reloaded.$historyArrowsEnabled.get()).toBe(true)
  })

  it('disabling persists the choice and drops in-flight browse state', async () => {
    browseBackward(SESSION_A, 'draft', HISTORY)

    expect(isBrowsingHistory(SESSION_A)).toBe(true)

    setHistoryArrowsEnabled(false)

    expect($historyArrowsEnabled.get()).toBe(false)
    expect($perSessionBrowse.get()).toEqual({})

    // Round-trip through a reload instead of pinning the stored string, so the
    // test checks the choice survived without freezing persistBoolean's format.
    vi.resetModules()
    const reloaded = await import('./composer-input-history')

    expect(reloaded.$historyArrowsEnabled.get()).toBe(false)
  })

  it('re-enabling persists the choice across a reload', async () => {
    setHistoryArrowsEnabled(false)
    setHistoryArrowsEnabled(true)

    vi.resetModules()
    const reloaded = await import('./composer-input-history')

    expect(reloaded.$historyArrowsEnabled.get()).toBe(true)
  })

  it('follows a sibling window toggling the preference', async () => {
    // Fresh module seeded while the key is absent → atom starts on, like a
    // window that opened before the sibling's Settings toggle.
    vi.resetModules()
    const reloaded = await import('./composer-input-history')

    expect(reloaded.$historyArrowsEnabled.get()).toBe(true)

    // The sibling's write lands first; the storage event fires after it.
    window.localStorage.setItem(ARROWS_KEY, 'false')
    window.dispatchEvent(new StorageEvent('storage', { key: ARROWS_KEY }))

    expect(reloaded.$historyArrowsEnabled.get()).toBe(false)

    // Removal falls back to the default (on).
    window.localStorage.removeItem(ARROWS_KEY)
    window.dispatchEvent(new StorageEvent('storage', { key: ARROWS_KEY }))

    expect(reloaded.$historyArrowsEnabled.get()).toBe(true)
  })

  it('a sibling disable also drops the passive window browse state', async () => {
    // Same fresh-module setup as the sibling case: the listener lives on this
    // module instance, so the browse state must be seeded on it too — the
    // top-level import's atom is a different instance after resetModules.
    vi.resetModules()
    const reloaded = await import('./composer-input-history')

    reloaded.browseBackward(SESSION_A, 'draft', HISTORY)
    expect(reloaded.isBrowsingHistory(SESSION_A)).toBe(true)

    window.localStorage.setItem(ARROWS_KEY, 'false')
    window.dispatchEvent(new StorageEvent('storage', { key: ARROWS_KEY }))

    expect(reloaded.$historyArrowsEnabled.get()).toBe(false)
    // use-composer-draft skips its stash reads while a cursor is live, so a
    // passive window that kept browsing would stop stashing new drafts.
    expect(reloaded.$perSessionBrowse.get()).toEqual({})
  })

  it('reads a stored false back as off after a reload', async () => {
    window.localStorage.setItem(ARROWS_KEY, 'false')
    vi.resetModules()

    const reloaded = await import('./composer-input-history')

    expect(reloaded.$historyArrowsEnabled.get()).toBe(false)
  })
})
