/**
 * The roster highlight and the Routines tile follow the chat the user is
 * LOOKING AT — the focused session's OWNER (profile + connection) — not the
 * gateway socket's home. Tab focus moves without swapping the socket, so
 * keying off `host.state.profile` highlighted the wrong bot whenever a focused
 * tab showed another profile's chat (community report: Newsanalyst chat open,
 * Hermes highlighted).
 *
 * Newer desktops publish the complete pair as `host.state.focusedSessionOwner`.
 * Older ones expose only `focusedSessionProfile`, a legacy HALF-SHAPE with no
 * source identity: pairing that profile with the ambient connection id
 * manufactures a cross-source owner out of two unrelated atoms, which is how a
 * remote bot got highlighted as "active" on the wrong machine. The fallback
 * therefore fails closed unless the focused profile IS the active profile.
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
    focusedProfile: null as null | string,
    focusedOwner: null as null | { authoritative?: boolean; connectionId: string; profile: string },
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
        // Both focus atoms are FEATURE-DETECTED: their absence is the whole
        // signal on an older desktop, so they must be genuinely absent.
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

  return { ...botState, isActiveRosterBot: data.isActiveRosterBot }
}

beforeEach(() => {
  state.activeConnectionId = ''
  state.connectionId = 'source-a'
  state.focusedOwner = null
  state.focusedProfile = null
  state.profile = 'default'
})

describe('a desktop that publishes the complete focused owner', () => {
  it('uses it verbatim and treats it as authoritative', async () => {
    state.focusedOwner = { connectionId: 'source-b', profile: 'worker' }

    const { $focusedBotOwner, focusedRosterOwner, isActiveRosterBot } = await load()
    const owner = focusedRosterOwner($focusedBotOwner.get())

    expect(owner).toEqual({ authoritative: true, connectionId: 'source-b', name: 'worker' })
    expect(isActiveRosterBot({ connectionId: 'source-b', name: 'worker', remoteSource: true }, owner)).toBe(true)
    // Same name on ANOTHER machine is a different agent.
    expect(isActiveRosterBot({ connectionId: 'source-a', name: 'worker', remoteSource: true }, owner)).toBe(false)
  })
})

describe('a legacy desktop with focusedSessionProfile only', () => {
  it('fails closed rather than pairing foreign focus with the ambient connection', async () => {
    state.focusedProfile = 'worker'
    state.profile = 'default'

    const { $focusedBotOwner, focusedRosterOwner, isActiveRosterBot } = await load()
    const owner = focusedRosterOwner($focusedBotOwner.get())

    expect(owner).toBeNull()
    expect(isActiveRosterBot({ connectionId: 'source-a', name: 'worker', remoteSource: true }, owner)).toBe(false)
    // And nothing else gets highlighted in its place.
    expect(isActiveRosterBot({ name: 'default' }, owner)).toBe(false)
  })

  it('reuses the ambient connection only when focus IS the active profile', async () => {
    state.focusedProfile = 'default'
    state.profile = 'default'

    const { $focusedBotOwner, focusedRosterOwner } = await load()

    // Non-authoritative: it is inferred, not published by the desktop.
    expect(focusedRosterOwner($focusedBotOwner.get())).toEqual({
      authoritative: false,
      connectionId: 'source-a',
      name: 'default'
    })
  })
})

describe('a desktop with neither focus atom', () => {
  it('falls back to the socket home profile', async () => {
    state.profile = 'researcher'
    state.activeConnectionId = 'source-a'
    state.connectionId = ''

    const { $focusedBotOwner, focusedMentionProfile, focusedRosterOwner } = await load()

    expect(focusedMentionProfile()).toBe('researcher')
    expect(focusedRosterOwner($focusedBotOwner.get())).toEqual({
      authoritative: false,
      connectionId: 'source-a',
      name: 'researcher'
    })
  })
})

describe('focusedRosterOwner', () => {
  it('reports nothing for an owner with no profile', async () => {
    const { focusedRosterOwner } = await load()

    expect(focusedRosterOwner(null)).toBeNull()
    expect(focusedRosterOwner({ connectionId: 'source-a', profile: '  ' })).toBeNull()
  })
})
