import { $subagentsBySession, type SubagentProgress } from './subagents'

export const SUBAGENT_LIVENESS_GRACE_MS = 30_000
export const SUBAGENT_ORPHAN_GRACE_MS = 60 * 60 * 1_000

export interface ActiveSubagentSnapshot {
  activeIds: readonly string[]
  profile: string
}

const normProfile = (profile: string | undefined) => profile?.trim() || 'default'

const isTerminal = (item: SubagentProgress) =>
  item.status === 'completed' || item.status === 'failed' || item.status === 'interrupted'

const usesStartupGrace = (item: SubagentProgress) => item.status === 'queued' || item.id.startsWith('delegate-tool:')

export function clearAllSubagents(): void {
  if (Object.keys($subagentsBySession.get()).length > 0) {
    $subagentsBySession.set({})
  }
}

/** Reconcile event-derived rows with per-profile gateway registries.
 *
 * Successful profile snapshots are authoritative after a short race grace.
 * Unavailable profiles remain untouched (fail-open). Queued rows and synthetic
 * fallback rows receive a longer grace after an authoritative snapshot because
 * they can precede backend registration. */
export function reconcileActiveSubagents(
  snapshots: readonly ActiveSubagentSnapshot[],
  now = Date.now(),
  allProfilesAuthoritative = true
): void {
  const current = $subagentsBySession.get()
  const itemsByProfile = new Map<string, Map<string, SubagentProgress>>()
  const authoritativeProfiles = new Set<string>()
  const keepIdsByProfile = new Map<string, Set<string>>()

  for (const item of Object.values(current).flat()) {
    const profile = normProfile(item.profile)
    const byId = itemsByProfile.get(profile) ?? new Map<string, SubagentProgress>()

    byId.set(item.id, item)
    itemsByProfile.set(profile, byId)
  }

  for (const snapshot of snapshots) {
    const profile = normProfile(snapshot.profile)
    const keepIds = keepIdsByProfile.get(profile) ?? new Set<string>()

    authoritativeProfiles.add(profile)

    for (const id of snapshot.activeIds) {
      if (id) {
        keepIds.add(id)
      }
    }

    keepIdsByProfile.set(profile, keepIds)
  }

  // Preserve missing parents while an authoritative descendant is still live,
  // otherwise nested children would jump to the tree root during reconciliation.
  for (const [profile, keepIds] of keepIdsByProfile) {
    const byId = itemsByProfile.get(profile)

    for (const activeId of keepIds) {
      let parentId = byId?.get(activeId)?.parentId ?? null
      const seen = new Set<string>()

      while (parentId && !seen.has(parentId)) {
        seen.add(parentId)
        keepIds.add(parentId)
        parentId = byId?.get(parentId)?.parentId ?? null
      }
    }
  }

  const keepIdsAnywhere = new Set<string>()

  for (const keepIds of keepIdsByProfile.values()) {
    keepIds.forEach(id => keepIdsAnywhere.add(id))
  }

  let changed = false
  const next: Record<string, SubagentProgress[]> = {}

  for (const [sid, list] of Object.entries(current)) {
    const retained = list.filter(item => {
      const keepIds = keepIdsByProfile.get(normProfile(item.profile))

      if (isTerminal(item) || (item.profile ? keepIds?.has(item.id) : keepIdsAnywhere.has(item.id))) {
        return true
      }

      const profileAuthoritative = item.profile
        ? authoritativeProfiles.has(normProfile(item.profile))
        : allProfilesAuthoritative

      if (!profileAuthoritative) {
        return true
      }

      const grace = usesStartupGrace(item) ? SUBAGENT_ORPHAN_GRACE_MS : SUBAGENT_LIVENESS_GRACE_MS

      return now - item.updatedAt <= grace
    })

    if (retained.length > 0) {
      next[sid] = retained
    }

    changed ||= retained.length !== list.length
  }

  if (changed) {
    $subagentsBySession.set(next)
  }
}
