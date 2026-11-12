/**
 * The persisted roster selection follows the chat the user is LOOKING AT when
 * a session owns the workspace (#154651) — the same owner the row highlight
 * already follows (botRowOwnsWorkspace). Without this, a selection left behind
 * by an earlier click re-owns the workspace on the next roster hydrate or pane
 * visibility flip: the stale row re-highlights and the tab-strip "+" creates
 * the next session under a bot nobody is reading.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

interface Store<T> {
  get: () => T
  listen: (listener: (value: T) => void) => () => void
}

const { state } = vi.hoisted(() => ({
  state: {
    activeConnectionId: '' as string,
    connectionId: 'source-a',
    focusedOwner: null as null | { authoritative?: boolean; connectionId: string; profile: string },
    focusedProfile: null as null | string,
    profile: 'default'
  }
}))

const store = <T>(read: () => T): Store<T> => ({ get: read, listen: () => () => undefined })

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: {
      activeConnectionId: () => state.activeConnectionId,
      state: {
        connectionId: store(() => state.connectionId),
        get focusedSessionOwner() {
          return state.focusedOwner ? store(() => state.focusedOwner) : undefined
        },
        get focusedSessionProfile() {
          return state.focusedProfile === null ? undefined : store(() => state.focusedProfile)
        },
        profile: store(() => state.profile)
      }
    },
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

/** $focusedBotOwner binds to host.state at module load, so each topology needs
 *  a fresh graph. */
async function load() {
  vi.resetModules()

  const [botState, data] = await Promise.all([import('./bot-state'), import('./data')])

  return { ...botState, $lastRoster: data.$lastRoster }
}

const roster = [
  { connectionId: 'source-a', name: 'alpha', remoteSource: true },
  { connectionId: 'source-b', name: 'worker', remoteSource: true }
]

beforeEach(() => {
  state.activeConnectionId = ''
  state.connectionId = 'source-a'
  state.focusedOwner = null
  state.focusedProfile = null
  state.profile = 'default'
})

describe('reconcileSelectionToFocusedChat — the selection follows the chat on screen (#154651)', () => {
  it('re-points a stale persisted selection at the focused chat\'s owner', async () => {
    state.focusedOwner = { connectionId: 'source-b', profile: 'worker' }

    const {
      $botChatFocused,
      $lastRoster,
      $selectedBot,
      $selectedRosterHydrated,
      $selectedRosterKey,
      reconcileSelectionToFocusedChat
    } = await load()

    $lastRoster.set(roster)
    // Left behind by an earlier click on alpha; the worker chat is on screen.
    $selectedRosterKey.set('source-a::alpha')
    $selectedRosterHydrated.set(true)
    $botChatFocused.set(true)

    reconcileSelectionToFocusedChat()

    expect($selectedRosterKey.get()).toBe('source-b::worker')
    // The Routines tile scope follows too — botSelectionKey for a
    // source-scoped row is the same source-qualified key.
    expect($selectedBot.get()).toBe('source-b::worker')
  })

  it('leaves the selection when no session owns the workspace', async () => {
    state.focusedOwner = { connectionId: 'source-b', profile: 'worker' }

    const { $botChatFocused, $lastRoster, $selectedRosterKey, reconcileSelectionToFocusedChat } = await load()

    $lastRoster.set(roster)
    $selectedRosterKey.set('source-a::alpha')
    $botChatFocused.set(false)

    reconcileSelectionToFocusedChat()

    expect($selectedRosterKey.get()).toBe('source-a::alpha')
  })

  it('fails closed on the legacy profile-only focus fallback', async () => {
    state.focusedProfile = 'worker'
    state.profile = 'default'

    const { $botChatFocused, $lastRoster, $selectedRosterKey, reconcileSelectionToFocusedChat } = await load()

    $lastRoster.set(roster)
    $selectedRosterKey.set('source-a::alpha')
    $botChatFocused.set(true)

    reconcileSelectionToFocusedChat()

    expect($selectedRosterKey.get()).toBe('source-a::alpha')
  })

  it('leaves the selection when the focused owner has no roster row', async () => {
    state.focusedOwner = { connectionId: 'source-z', profile: 'ghost' }

    const { $botChatFocused, $lastRoster, $selectedRosterKey, reconcileSelectionToFocusedChat } = await load()

    $lastRoster.set(roster)
    $selectedRosterKey.set('source-a::alpha')
    $botChatFocused.set(true)

    reconcileSelectionToFocusedChat()

    expect($selectedRosterKey.get()).toBe('source-a::alpha')
  })
})
