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

// The gateway's `config.get` / `config.set` are `@_profile_scoped`: a request with no `profile`
// param resolves to the profile the backend was *launched* with. So an unscoped read/write targets
// the launch profile no matter which profile the menu is showing — the UI ends up displaying one
// profile's mode for all of them and a write silently edits the launch profile (#125969). Name the
// profile the call is for; a blank name IS the launch profile, so omit the param (keeping the
// backend's os.environ / session precedence) — the same rule `api/client.ts`'s `profileScoped()` uses.
function scopeToProfile(profile: string, params: Record<string, unknown>): Record<string, unknown> {
  const name = profile.trim()

  return name ? { ...params, profile: name } : params
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

  const result = (await requestGateway(
    'config.get',
    scopeToProfile(profile, { key: 'approvals.mode' })
  )) as { value?: string }

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
    const result = (await requestGateway(
      'config.set',
      scopeToProfile(profile, { key: 'approvals.mode', value: mode })
    )) as { value?: string }

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
