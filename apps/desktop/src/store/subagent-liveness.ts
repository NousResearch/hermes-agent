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
  const items = Object.values(current).flat()
  const byId = new Map(items.map(item => [item.id, item]))
  const authoritativeProfiles = new Set(snapshots.map(item => normProfile(item.profile)))
  const keepIds = new Set(snapshots.flatMap(item => item.activeIds).filter(Boolean))

  // Preserve missing parents while an authoritative descendant is still live,
  // otherwise nested children would jump to the tree root during reconciliation.
  for (const activeId of keepIds) {
    let parentId = byId.get(activeId)?.parentId ?? null
    const seen = new Set<string>()

    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)
      keepIds.add(parentId)
      parentId = byId.get(parentId)?.parentId ?? null
    }
  }

  let changed = false
  const next: Record<string, SubagentProgress[]> = {}

  for (const [sid, list] of Object.entries(current)) {
    const retained = list.filter(item => {
      if (isTerminal(item) || keepIds.has(item.id)) {
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
