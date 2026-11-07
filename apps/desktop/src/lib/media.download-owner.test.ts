import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// UI decoration is irrelevant to routing; owner resolution is exercised below.
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { setApiRequestConnection } from '@/api/client'
import * as sessionStore from '@/store/session'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint,
  setSessions
} from '@/store/session'
import { $sessionStates, $sessionTiles, clearAllSessionStates, recordSessionEventScope } from '@/store/session-states'

import { downloadGatewayMediaFile, type GatewayFileOrigin } from './media'

const save = vi.fn(async () => ({ saved: true }))
const path = '/synthetic/report.pdf'
const botOwner = { connectionId: 'local', profile: 'bot-profile' }

function tile(storedSessionId = 'stored', runtimeId = 'runtime', ownerRoute = botOwner) {
  $sessionTiles.set([{ storedSessionId, runtimeId, ownerRoute }] as never)
}

beforeEach(() => {
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set({ mode: 'remote', connectionId: 'ambient-remote', profile: 'default' } as never)
  setApiRequestConnection('ambient-remote')
  save.mockClear()
  window.hermesDesktop = { saveGatewayFile: save } as never
})

afterEach(() => {
  vi.restoreAllMocks()
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set(null)
  setApiRequestConnection(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

const saved = (sessionId = 'stored', owner = botOwner) => ({ ...owner, sessionId, path, suggestedName: 'report.pdf' })

describe('session-owned gateway downloads', () => {
  it('pins the bot owner instead of the window profile and remote connection', async () => {
    tile()
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('keeps an exact connection pin even when it equals the ambient API connection', async () => {
    tile()
    setApiRequestConnection('local')
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('uses a remote backend targetProfile, not its display alias or the local foreground', async () => {
    $connection.set({ mode: 'local', connectionId: 'local', profile: 'default' } as never)
    tile('stored', 'runtime', {
      connectionId: 'bot-remote',
      profile: 'display-alias',
      targetProfile: 'bot-profile'
    } as never)
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(
      saved('stored', { connectionId: 'bot-remote', profile: 'bot-profile' })
    )
  })

  it('maps a tile runtime to its persisted id', async () => {
    tile()
    await downloadGatewayMediaFile(path, { sessionId: 'runtime' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('prefers a persisted tile identity over a colliding stale runtime binding', async () => {
    $sessionTiles.set([
      { storedSessionId: 'stale', runtimeId: 'stored', ownerRoute: { connectionId: 'wrong', profile: 'wrong' } },
      { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: botOwner }
    ] as never)
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it.each(['mirror', 'main'])('maps the %s runtime binding through its persisted owner hint', async binding => {
    setSessionOwnerHint('stored', botOwner)

    if (binding === 'mirror') {
      $sessionStates.set({ runtime: { storedSessionId: 'stored' } } as never)
    } else {
      $activeSessionId.set('runtime')
      $selectedStoredSessionId.set('stored')
    }

    await downloadGatewayMediaFile(path, { sessionId: 'runtime' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it.each(['hint', 'row'])('keeps the exact runtime owner with a profile-only stored %s', async evidence => {
    // Reproduce the review's legacy profile-only hint at the resolver boundary;
    // today's hint writer only accepts exact routes. The row case uses its real shape.
    if (evidence === 'hint') {
      vi.spyOn(sessionStore, 'getSessionOwnerHint').mockImplementation(id =>
        id === 'stored' ? ('bot-profile' as never) : undefined
      )
    } else {
      setSessions([{ id: 'stored', profile: 'bot-profile' }] as never)
    }

    $sessionStates.set({ runtime: { storedSessionId: 'stored' } } as never)
    recordSessionEventScope({ session_id: 'runtime', connectionId: 'bot-remote', profile: 'bot-profile' })
    await downloadGatewayMediaFile(path, { sessionId: 'runtime' })
    expect(save).toHaveBeenCalledExactlyOnceWith(
      saved('stored', { connectionId: 'bot-remote', profile: 'bot-profile' })
    )
  })

  it.each(['mirror', 'main'])('keeps a closed persisted hint ahead of a stale %s mapping', async binding => {
    setSessionOwnerHint('stored', botOwner)
    setSessionOwnerHint('other', { connectionId: 'wrong', profile: 'wrong' })

    if (binding === 'mirror') {
      $sessionStates.set({ stored: { storedSessionId: 'other' } } as never)
    } else {
      $activeSessionId.set('stored')
      $selectedStoredSessionId.set('other')
    }

    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it.each(['mirror', 'main'])('keeps a closed persisted row ahead of a stale %s mapping', async binding => {
    setSessions([{ id: 'stored', profile: 'bot-profile', connection_id: 'local' }] as never)
    setSessionOwnerHint('other', { connectionId: 'wrong', profile: 'wrong' })

    if (binding === 'mirror') {
      $sessionStates.set({ stored: { storedSessionId: 'other' } } as never)
    } else {
      $activeSessionId.set('stored')
      $selectedStoredSessionId.set('other')
    }

    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('rejects a runtime owner conflicting with a stored profile rather than combining them', async () => {
    setSessions([{ id: 'stored', profile: 'bot-profile' }] as never)
    $sessionStates.set({ runtime: { storedSessionId: 'stored' } } as never)
    recordSessionEventScope({ session_id: 'runtime', connectionId: 'bot-remote', profile: 'other' })
    await expect(downloadGatewayMediaFile(path, { sessionId: 'runtime' })).rejects.toThrow(/owner.*conflict/i)
    expect(save).not.toHaveBeenCalled()
  })

  it('keeps a durable exact alias route ahead of a stale runtime owner', async () => {
    setSessionOwnerHint('stored', { connectionId: 'local', profile: 'alias', targetProfile: 'bot-profile' })
    $sessionStates.set({ runtime: { storedSessionId: 'stored' } } as never)
    recordSessionEventScope({ session_id: 'runtime', connectionId: 'wrong', profile: 'wrong' })
    await downloadGatewayMediaFile(path, { sessionId: 'runtime' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('uses a persisted session row when no tile is open', async () => {
    setSessions([{ id: 'stored', profile: 'bot-profile', connection_id: 'local' }] as never)
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it('preserves proven profile-pool routing without grafting the ambient connection', async () => {
    $sessionTiles.set([{ storedSessionId: 'stored', ownerProfile: 'bot-profile' }] as never)
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith({
      sessionId: 'stored',
      profile: 'bot-profile',
      path,
      suggestedName: 'report.pdf'
    })
  })

  it('rejects an unknown owner before invoking the save bridge', async () => {
    await expect(downloadGatewayMediaFile(path, { sessionId: 'unknown' })).rejects.toThrow(/owner.*resolved/i)
    expect(save).not.toHaveBeenCalled()
  })

  it.each([false, true])('rejects an unresolved live runtime even with explicit owner=%s', async explicit => {
    recordSessionEventScope({ session_id: 'runtime', ...botOwner })
    await expect(
      downloadGatewayMediaFile(path, {
        sessionId: 'runtime',
        ...(explicit ? { owner: botOwner } : {})
      })
    ).rejects.toThrow(/persisted session/i)
    expect(save).not.toHaveBeenCalled()
  })

  it.each([
    { owner: { connectionId: 'other', profile: 'bot-profile' } },
    { owner: { connectionId: 'local', profile: 'other' } },
    { profile: 'other' },
    { owner: botOwner, profile: 'other' }
  ])('rejects explicit ownership conflicts: %j', async origin => {
    tile()
    await expect(downloadGatewayMediaFile(path, { sessionId: 'stored', ...origin })).rejects.toThrow(/owner.*conflict/i)
    expect(save).not.toHaveBeenCalled()
  })

  it.each([{ owner: botOwner }, { profile: 'bot-profile' }])(
    'accepts matching explicit ownership: %j',
    async origin => {
      tile()
      await downloadGatewayMediaFile(path, { sessionId: 'stored', ...origin })
      expect(save).toHaveBeenCalledExactlyOnceWith(saved())
    }
  )

  it.each([{ owner: botOwner }, { profile: 'bot-profile' }])(
    'preserves explicit ownership of an unlisted persisted artifact: %j',
    async origin => {
      await downloadGatewayMediaFile(path, { sessionId: 'unlisted-stored', ...origin })
      expect(save).toHaveBeenCalledExactlyOnceWith({
        ...saved('unlisted-stored'),
        connectionId: 'owner' in origin ? 'local' : 'ambient-remote'
      })
    }
  )

  it('combines explicit connection and profile for an unlisted persisted artifact', async () => {
    await downloadGatewayMediaFile(path, {
      sessionId: 'unlisted-stored',
      owner: { connectionId: 'local' },
      profile: 'bot-profile'
    })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved('unlisted-stored'))
  })

  it('accepts the explicit display alias but sends the backend profile', async () => {
    tile('stored', 'runtime', {
      connectionId: 'local',
      profile: 'display-alias',
      targetProfile: 'bot-profile'
    } as never)
    await downloadGatewayMediaFile(path, { sessionId: 'stored', profile: 'display-alias' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it.each(['mirror', 'main'])('rejects an unbound %s runtime with no event owner', async binding => {
    if (binding === 'mirror') {
      $sessionStates.set({ runtime: { storedSessionId: null } } as never)
    } else {
      $activeSessionId.set('runtime')
    }

    await expect(downloadGatewayMediaFile(path, { sessionId: 'runtime', owner: botOwner })).rejects.toThrow(
      /persisted session/i
    )
    expect(save).not.toHaveBeenCalled()
  })

  it('keeps a canonical persisted id when runtime events also use that id', async () => {
    setSessionOwnerHint('stored', botOwner)
    recordSessionEventScope({ session_id: 'stored', ...botOwner })
    await downloadGatewayMediaFile(path, { sessionId: 'stored' })
    expect(save).toHaveBeenCalledExactlyOnceWith(saved())
  })

  it.each([{}, { owner: botOwner }, { owner: { profile: 'bot-profile' } }, { profile: 'bot-profile' }])(
    'leaves no-session downloads unchanged: %j',
    async (origin: GatewayFileOrigin) => {
      await downloadGatewayMediaFile(path, origin)
      const owner = origin.owner ?? { connectionId: 'ambient-remote', profile: origin.profile ?? 'default' }
      expect(save).toHaveBeenCalledExactlyOnceWith({ ...owner, path, suggestedName: 'report.pdf' })
    }
  )
})
