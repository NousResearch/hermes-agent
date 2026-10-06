import { describe, expect, it } from 'vitest'

import { assessDesktopUpdateApply } from './update-apply-gate'

describe('assessDesktopUpdateApply', () => {
  it('permits a compatible available update independent of freshness classification', () => {
    expect(assessDesktopUpdateApply({ updateAvailable: true, targetSha: 'a'.repeat(40), targetRequiredBackendContract: 8 }, 8))
      .toEqual({ safeToUpdate: true, compatibility: 'PASS', reason: 'READY' })
  })

  it.each([
    [{ updateAvailable: false, targetSha: 'a'.repeat(40), targetRequiredBackendContract: 8 }, 8, 'UPDATE_NOT_AVAILABLE'],
    [{ updateAvailable: true, targetRequiredBackendContract: 8 }, 8, 'TARGET_SHA_UNKNOWN'],
    [{ updateAvailable: true, targetSha: 'a'.repeat(40) }, 8, 'BACKEND_CONTRACT_UNKNOWN'],
    [{ updateAvailable: true, targetSha: 'a'.repeat(40), targetRequiredBackendContract: 9 }, 8, 'TARGET_REQUIRES_NEWER_BACKEND_CONTRACT']
  ])('fails closed when %s', (status, activeContract, reason) => {
    expect(assessDesktopUpdateApply(status, activeContract)).toMatchObject({ safeToUpdate: false, reason })
  })
})