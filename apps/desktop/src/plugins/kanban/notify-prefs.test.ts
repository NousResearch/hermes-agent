import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $kanbanNotifyPrefs,
  bindNotifyPrefs,
  BLOCKED_KINDS,
  DEFAULT_KANBAN_NOTIFY_PREFS,
  normalizeNotifyPrefs,
  NOTIFY_PREFS_KEY,
  setBlockedPopups,
  silencedKindsFor
} from './notify-prefs'

const makeStorage = (stored: Record<string, unknown> = {}) => ({
  get: vi.fn(<T>(key: string, fallback: T): T => (key in stored ? (stored[key] as T) : fallback)),
  set: vi.fn(),
  remove: vi.fn()
})

beforeEach(() => {
  $kanbanNotifyPrefs.set(DEFAULT_KANBAN_NOTIFY_PREFS)
})

describe('blocked popup preference', () => {
  it('defaults to on, so upstream behaviour is unchanged', () => {
    expect(DEFAULT_KANBAN_NOTIFY_PREFS.blockedPopups).toBe(true)
    expect(silencedKindsFor(DEFAULT_KANBAN_NOTIFY_PREFS)).toEqual([])
  })

  it('off silences exactly the worker-blocked kinds', () => {
    expect(silencedKindsFor({ blockedPopups: false })).toEqual(BLOCKED_KINDS)
  })

  it('a stored value that is not a boolean falls back to on rather than silencing popups', () => {
    expect(normalizeNotifyPrefs(undefined)).toEqual({ blockedPopups: true })
    expect(normalizeNotifyPrefs({ blockedPopups: 'false' })).toEqual({ blockedPopups: true })
    expect(normalizeNotifyPrefs({ blockedPopups: false })).toEqual({ blockedPopups: false })
  })

  it('hydrates from storage and mirrors changes back to it', () => {
    const storage = makeStorage({ [NOTIFY_PREFS_KEY]: { blockedPopups: false } })
    const unsub = bindNotifyPrefs(storage as never)

    expect($kanbanNotifyPrefs.get().blockedPopups).toBe(false)

    setBlockedPopups(true)

    expect(storage.set).toHaveBeenCalledWith(NOTIFY_PREFS_KEY, { blockedPopups: true })
    unsub()
  })
})
