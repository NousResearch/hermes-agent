export type BackendCompatibility = 'PASS' | 'FAIL' | 'UNKNOWN'

export interface DesktopUpdateApplyAssessment {
  compatibility: BackendCompatibility
  reason: 'READY' | 'UPDATE_NOT_AVAILABLE' | 'TARGET_SHA_UNKNOWN' | 'BACKEND_CONTRACT_UNKNOWN' | 'TARGET_REQUIRES_NEWER_BACKEND_CONTRACT'
  safeToUpdate: boolean
}

export interface DesktopUpdateApplyEvidence {
  targetRequiredBackendContract?: number | null
  targetSha?: string
  updateAvailable?: boolean
}

function contract(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
}

export function assessDesktopUpdateApply(
  status: DesktopUpdateApplyEvidence | null | undefined,
  activeBackendContract: number | null | undefined
): DesktopUpdateApplyAssessment {
  const required = contract(status?.targetRequiredBackendContract)
  const active = contract(activeBackendContract)
  const compatibility: BackendCompatibility = required === null || active === null ? 'UNKNOWN' : required <= active ? 'PASS' : 'FAIL'

  if (status?.updateAvailable !== true) return { safeToUpdate: false, compatibility, reason: 'UPDATE_NOT_AVAILABLE' }
  if (!status.targetSha) return { safeToUpdate: false, compatibility, reason: 'TARGET_SHA_UNKNOWN' }
  if (compatibility === 'UNKNOWN') return { safeToUpdate: false, compatibility, reason: 'BACKEND_CONTRACT_UNKNOWN' }
  if (compatibility === 'FAIL') return { safeToUpdate: false, compatibility, reason: 'TARGET_REQUIRES_NEWER_BACKEND_CONTRACT' }

  return { safeToUpdate: true, compatibility, reason: 'READY' }
}