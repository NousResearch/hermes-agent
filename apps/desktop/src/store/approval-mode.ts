import { atom } from 'nanostores'

export type ApprovalMode = 'manual' | 'off' | 'smart'
export type ApprovalModeRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

const APPROVAL_MODES = new Set<ApprovalMode>(['manual', 'smart', 'off'])
const revisions = new Map<string, number>()
const confirmedModes = new Map<string, ApprovalMode>()

export const $approvalModes = atom<Record<string, ApprovalMode>>({})

function profileKey(profile: string): string {
  return profile.trim() || 'default'
}

function nextRevision(profile: string): number {
  const revision = (revisions.get(profile) ?? 0) + 1
  revisions.set(profile, revision)

  return revision
}

function normalizeApprovalMode(value: unknown): ApprovalMode {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase() as ApprovalMode

  return APPROVAL_MODES.has(normalized) ? normalized : 'manual'
}

export function approvalModeForProfile(profile: string): ApprovalMode {
  return $approvalModes.get()[profileKey(profile)] ?? 'smart'
}

function cacheApprovalMode(profile: string, mode: ApprovalMode): void {
  const key = profileKey(profile)
  $approvalModes.set({ ...$approvalModes.get(), [key]: mode })
}

export function reconcileApprovalModeForProfile(profile: string, value: unknown): ApprovalMode {
  const key = profileKey(profile)
  const mode = normalizeApprovalMode(value)
  nextRevision(key)
  confirmedModes.set(key, mode)
  cacheApprovalMode(key, mode)

  return mode
}

export async function syncApprovalModeForProfile(
  requestGateway: ApprovalModeRequester,
  profile: string
): Promise<ApprovalMode> {
  const key = profileKey(profile)
  const revision = nextRevision(key)
  const result = (await requestGateway('config.get', { key: 'approvals.mode' })) as { value?: string }
  const mode = normalizeApprovalMode(result?.value)

  if (revisions.get(key) === revision) {
    confirmedModes.set(key, mode)
    cacheApprovalMode(key, mode)
  }

  return mode
}

export async function setApprovalModeForProfile(
  requestGateway: ApprovalModeRequester,
  profile: string,
  mode: ApprovalMode
): Promise<ApprovalMode> {
  const key = profileKey(profile)
  const revision = nextRevision(key)
  cacheApprovalMode(key, mode)

  try {
    const result = (await requestGateway('config.set', {
      key: 'approvals.mode',
      value: mode
    })) as { value?: string }

    const authoritative = normalizeApprovalMode(result?.value)

    if (revisions.get(key) === revision) {
      confirmedModes.set(key, authoritative)
      cacheApprovalMode(key, authoritative)
    }

    return authoritative
  } catch (error) {
    if (revisions.get(key) === revision) {
      cacheApprovalMode(key, confirmedModes.get(key) ?? 'smart')
    }

    throw error
  }
}
