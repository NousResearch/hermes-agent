import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

// The probe reaches the backend through getSession (REST), never the gateway
// socket — mock exactly that seam and leave the rest of @/hermes real.
vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  getSession: vi.fn()
}))

// The routed leg of requestForSessionProfile: a real dial would need a live
// secondary gateway, which is not what this file is proving.
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

const { getSession } = await import('@/hermes')
const { requestGatewayForProfile } = await import('@/store/gateway')
const { $activeGatewayProfile, $profiles } = await import('./profile')
const { $projectTree } = await import('./projects')
const { $cronSessions, $messagingSessions, $sessions, $unlistedSessionOwnerRows, _resetSessionOwnerHintsForTests } =
  await import('./session')
const { $sessionTiles, knownOwnerForSession, requestForOwnedSession } = await import('./session-states')
const { isSessionOwnerResolutionError } = await import('./session-owner-resolution')

// The archived cross-profile session from FRAMEWORK-07: owned by `arthur`
// while the window's active profile is something else entirely.
const SESSION_ID = '20261017_222002_d0e1121d'
const AMBIENT_READ = { session_id: SESSION_ID }

const row = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

function resetOwnerState(): void {
  $profiles.set([{ name: 'default' }, { name: 'arthur' }] as never)
  $activeGatewayProfile.set('default')
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $projectTree.set([])
  $unlistedSessionOwnerRows.set([])
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.mocked(getSession).mockReset()
  vi.mocked(requestGatewayForProfile).mockReset()
  vi.mocked(requestGatewayForProfile).mockResolvedValue({ control: {} } as never)
}

beforeEach(resetOwnerState)
afterEach(resetOwnerState)

describe('requestForOwnedSession cross-profile probe (FRAMEWORK-07)', () => {
  it('probes before asserting and routes an ownerless-but-resolvable read to the owning profile', async () => {
    vi.mocked(getSession).mockImplementation(async (id, scope) => {
      if (scope === 'arthur') {
        return row({ archived: true, id, profile: 'arthur' })
      }

      throw new Error('404: Session not found')
    })

    // The sync ladder (tile → hint → row → runtime event) genuinely has
    // nothing for this id, which is exactly the state the banner reported.
    expect(knownOwnerForSession(SESSION_ID)).toBeUndefined()

    const ambient = vi.fn(async () => ({ via: 'ambient' }))

    await expect(
      requestForOwnedSession<{ control: unknown }>(SESSION_ID, ambient as never, 'session.control.read', AMBIENT_READ)
    ).resolves.toEqual({ control: {} })

    // The probe ran BEFORE the request was dispatched: active profile first,
    // then the cross-profile sweep that found `arthur`.
    expect(vi.mocked(getSession).mock.calls.map(call => call[1])).toEqual(['default', 'arthur'])

    // ...and the read went to the OWNING profile, never the ambient socket.
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'arthur',
      'session.control.read',
      AMBIENT_READ,
      undefined,
      undefined
    )
    expect(ambient).not.toHaveBeenCalled()

    // Fix A: the probe's by-id resolve parks the archived row off-list, so the
    // NEXT read resolves from the sync ladder with no second probe.
    expect($sessions.get()).toEqual([])
    expect($unlistedSessionOwnerRows.get().map(parked => parked.id)).toEqual([SESSION_ID])
    expect(knownOwnerForSession(SESSION_ID)).toBe('arthur')
  })

  it('still fails closed when no profile owns the session', async () => {
    vi.mocked(getSession).mockRejectedValue(new Error('404: Session not found'))
    const ambient = vi.fn()

    await expect(requestForOwnedSession(SESSION_ID, ambient, 'session.control.read', AMBIENT_READ)).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )

    // The probe really ran (it did not short-circuit into the old failure)...
    expect(vi.mocked(getSession).mock.calls.length).toBeGreaterThan(0)
    // ...and an unresolved owner still never rides the ambient socket.
    expect(ambient).not.toHaveBeenCalled()
  })

  it('does not probe on a legacy single-backend install, where the ambient gateway owns every session', async () => {
    $profiles.set([{ name: 'default' }] as never)
    const ambient = vi.fn(async () => ({ via: 'ambient' }))

    await expect(
      requestForOwnedSession<{ via: string }>(SESSION_ID, ambient as never, 'session.control.read', AMBIENT_READ)
    ).resolves.toEqual({ via: 'ambient' })

    expect(getSession).not.toHaveBeenCalled()
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
    expect(ambient).toHaveBeenCalledTimes(1)
  })

  it('leaves a known owner on the fast path — no probe, no extra backend read', async () => {
    $sessions.set([row({ archived: true, id: SESSION_ID, profile: 'arthur' })])
    const ambient = vi.fn()

    await expect(
      requestForOwnedSession<{ control: unknown }>(SESSION_ID, ambient, 'session.control.read', AMBIENT_READ)
    ).resolves.toEqual({ control: {} })

    expect(getSession).not.toHaveBeenCalled()
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'arthur',
      'session.control.read',
      AMBIENT_READ,
      undefined,
      undefined
    )
    expect(ambient).not.toHaveBeenCalled()
  })
})
