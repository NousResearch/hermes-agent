/**
 * `$botChatSessionIds` is the store a surface must SUBSCRIBE to when it asks
 * `isBotChatSession`.
 *
 * The composer originally subscribed to `$sessionStates` while reading this
 * set. It looked fine because `$sessionStates` churns constantly — busy flags,
 * stream ticks — so the stale answer was overwritten within a frame or two.
 * The assertion that matters is the one that made that wrong: this set moves on
 * its own, with no `$sessionStates` write anywhere near it, so a subscriber to
 * the other store can miss the transition entirely.
 */

import { beforeEach, describe, expect, it } from 'vitest'

const { $botChatSessionIds, $sessionStates, $sessionTiles, isBotChatSession, setSessionTileWorkspaceScope } =
  await import('./session-states')

const botScope = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha' }

beforeEach(() => {
  $botChatSessionIds.set(new Set())
  // isBotChatSession takes a LIVE session id and files the answer under the
  // stored one, so a binding has to exist for the lookup to resolve at all.
  $sessionTiles.set([{ runtimeId: 'runtime-1', storedSessionId: 'stored-1' } as never])
})

describe('the bot-chat set moves independently of session state', () => {
  it('notifies its own listeners without touching $sessionStates', () => {
    const seen: number[] = []
    const statesBefore = $sessionStates.get()
    const stop = $botChatSessionIds.listen(ids => seen.push(ids.size))

    try {
      setSessionTileWorkspaceScope('stored-1', botScope)
    } finally {
      stop()
    }

    expect(seen).toEqual([1])
    // The store a stale subscriber would have been waiting on never moved.
    expect($sessionStates.get()).toBe(statesBefore)
  })

  it('answers for the live id behind a marked stored one, and stops when cleared', () => {
    setSessionTileWorkspaceScope('stored-1', botScope)

    expect(isBotChatSession('runtime-1')).toBe(true)
    expect(isBotChatSession('stored-1')).toBe(true)

    setSessionTileWorkspaceScope('stored-1', { workspaceMode: 'sessions', workspaceOwnerKey: '' })

    expect(isBotChatSession('runtime-1')).toBe(false)
  })

  it('is silent for an id it never marked, and for no id at all', () => {
    expect(isBotChatSession('never-seen')).toBe(false)
    expect(isBotChatSession(null)).toBe(false)
    expect(isBotChatSession(undefined)).toBe(false)
  })

  it('resolves a marked chat only once the binding that names it exists', () => {
    // The reason a subscriber cannot watch the scope set alone: with the set
    // already marked, the answer still flips when the BINDING arrives.
    $sessionTiles.set([])
    setSessionTileWorkspaceScope('stored-1', botScope)

    expect(isBotChatSession('runtime-1')).toBe(false)

    $sessionTiles.set([{ runtimeId: 'runtime-1', storedSessionId: 'stored-1' } as never])

    expect(isBotChatSession('runtime-1')).toBe(true)
  })

  it('does not re-notify when the same scope is recorded twice', () => {
    setSessionTileWorkspaceScope('stored-1', botScope)

    const seen: number[] = []
    const stop = $botChatSessionIds.listen(ids => seen.push(ids.size))

    try {
      setSessionTileWorkspaceScope('stored-1', botScope)
    } finally {
      stop()
    }

    expect(seen).toEqual([])
  })
})
