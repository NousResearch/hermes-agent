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

const { $botChatScopes, $botChatSessionIds } = await import('./bot-chat-scope')

const { $sessionStates, $sessionTiles, isBotChatSession, rekeySessionTile, setSessionTileWorkspaceScope } =
  await import('./session-states')

const botScope = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha' }

beforeEach(() => {
  $botChatSessionIds.set(new Set())
  $botChatScopes.set({})
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
})

describe('the bot-chat scope follows a compression rotation', () => {
  // A Bot Chat in MAIN has no tile, so the scope lives only under the stored
  // id it was opened with. Compression mints a new tip id; the chat it names
  // is still the bot's, and the main tab caption + composer read the scope
  // under whichever id is selected now.
  const canonical = { ...botScope, workspaceTabTitle: 'Bot Chat' }

  it('carries the scope and the bot-chat mark onto the new tip', () => {
    $sessionTiles.set([])
    setSessionTileWorkspaceScope('segment-1', canonical)

    rekeySessionTile('segment-1', 'segment-2', 'runtime-1')

    expect($botChatScopes.get()['segment-2']).toEqual(canonical)
    expect($botChatSessionIds.get().has('segment-2')).toBe(true)
    // The composer asks by the live id, which now binds to the tip.
    $sessionTiles.set([{ runtimeId: 'runtime-1', storedSessionId: 'segment-2' } as never])
    expect(isBotChatSession('runtime-1')).toBe(true)
    // Main still names the old id until route-follow re-selects the tip.
    expect($botChatScopes.get()['segment-1']).toEqual(canonical)
  })

  it('carries the persisted mark alone after a relaunch emptied the window-local scopes', () => {
    $sessionTiles.set([])
    $botChatSessionIds.set(new Set(['segment-1']))

    rekeySessionTile('segment-1', 'segment-2', 'runtime-1')

    expect($botChatSessionIds.get().has('segment-2')).toBe(true)
    expect($botChatScopes.get()['segment-2']).toBeUndefined()
  })

  it('leaves a plain session rotation unmarked', () => {
    $sessionTiles.set([])

    rekeySessionTile('plain-1', 'plain-2', 'runtime-1')

    expect($botChatSessionIds.get().has('plain-2')).toBe(false)
    expect($botChatScopes.get()['plain-2']).toBeUndefined()
  })
})
