import crypto from 'node:crypto'

const PROFILE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const MAX_PROFILES = 4

export const SPECIALIST_PROFILE_QUIESCED = 'SPECIALIST_PROFILE_QUIESCED'

export class SpecialistProfileQuiescedError extends Error {
  readonly code = SPECIALIST_PROFILE_QUIESCED

  constructor(profileId: string) {
    super(`Profile "${profileId}" is temporarily quiesced for an explicit specialist capacity lease.`)
    this.name = 'SpecialistProfileQuiescedError'
  }
}

export class SpecialistQuiesceError extends Error {
  constructor(
    readonly code: 'INVALID_QUIESCE_PROFILE_SET' | 'QUIESCE_CONFLICT' | 'QUIESCE_STOP_FAILED',
    message: string
  ) {
    super(message)
    this.name = 'SpecialistQuiesceError'
  }
}

export interface SpecialistDispatchQuiesceDeps {
  /** Electron-owned teardown; callers never signal profile children directly. */
  stopProfile: (profileId: string) => Promise<void>
}

export interface SpecialistDispatchQuiesceLease {
  leaseId: string
  profileIds: string[]
}

export type SpecialistDispatchQuiesceRelease =
  | { outcome: 'released'; profileIds: string[] }
  | { outcome: 'rejected'; reasonCode: 'UNKNOWN_QUIESCE_LEASE' }

function exactProfileIds(profileIds: string[]): string[] {
  if (
    !Array.isArray(profileIds) ||
    profileIds.length < 1 ||
    profileIds.length > MAX_PROFILES ||
    profileIds.some(profileId => typeof profileId !== 'string' || !PROFILE.test(profileId)) ||
    profileIds.some((profileId, index) => index > 0 && profileIds[index - 1] >= profileId)
  ) {
    throw new SpecialistQuiesceError('INVALID_QUIESCE_PROFILE_SET', 'Quiesce profiles must be sorted, unique, valid IDs (1-4).')
  }

  return [...profileIds]
}

/**
 * Electron-owned temporary profile quiesce leases. A profile is marked before
 * its managed pool child is stopped, closing the renderer reconnect race that
 * would otherwise refill the fixed four-slot budget during certification.
 */
export function createSpecialistDispatchQuiesce(deps: SpecialistDispatchQuiesceDeps) {
  const leases = new Map<string, string[]>()
  const profileLeases = new Map<string, string>()

  function clearLease(leaseId: string, profileIds: string[]): void {
    leases.delete(leaseId)
    for (const profileId of profileIds) {
      if (profileLeases.get(profileId) === leaseId) {
        profileLeases.delete(profileId)
      }
    }
  }

  return {
    async acquire(profileIds: string[]): Promise<SpecialistDispatchQuiesceLease> {
      const exact = exactProfileIds(profileIds)
      if (leases.size > 0) {
        throw new SpecialistQuiesceError('QUIESCE_CONFLICT', 'An explicit specialist capacity lease is already active.')
      }
      const conflicting = exact.find(profileId => profileLeases.has(profileId))

      if (conflicting) {
        throw new SpecialistQuiesceError('QUIESCE_CONFLICT', `Profile "${conflicting}" is already quiesced.`)
      }

      const leaseId = `specialist-quiesce-${crypto.randomUUID()}`
      leases.set(leaseId, exact)
      for (const profileId of exact) {
        profileLeases.set(profileId, leaseId)
      }

      try {
        for (const profileId of exact) {
          await deps.stopProfile(profileId)
        }
      } catch {
        clearLease(leaseId, exact)
        throw new SpecialistQuiesceError('QUIESCE_STOP_FAILED', 'Electron could not quiesce the requested profile pool.')
      }

      return { leaseId, profileIds: exact }
    },

    assertCanStart(profileId: string): void {
      if (profileLeases.has(profileId)) {
        throw new SpecialistProfileQuiescedError(profileId)
      }

      // The named profiles are the only processes this lease tears down, but
      // any background reconnect could otherwise refill the fixed four-slot
      // pool with a different profile before the explicit runner reserves it.
      // The runner bypasses this normal profile-backend spawn path; ordinary
      // renderer reconnects are blocked until the one exact lease is released.
      if (leases.size > 0) {
        throw new SpecialistProfileQuiescedError(profileId)
      }
    },

    isQuiesced(profileId: string): boolean {
      return profileLeases.has(profileId)
    },

    release(leaseId: string): SpecialistDispatchQuiesceRelease {
      const profileIds = leases.get(leaseId)

      if (!profileIds) {
        return { outcome: 'rejected', reasonCode: 'UNKNOWN_QUIESCE_LEASE' }
      }

      clearLease(leaseId, profileIds)
      return { outcome: 'released', profileIds }
    }
  }
}