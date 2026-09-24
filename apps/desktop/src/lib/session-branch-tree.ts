import type { SessionInfo } from '@/types/hermes'

export interface SidebarSessionEntry {
  branchStem?: string
  session: SessionInfo
}

export interface FlattenSessionsOptions {
  /**
   * Keep the input root order instead of re-sorting by group recency.
   * Use for hand-ordered surfaces (pinned ids, manual recents drag) so a
   * turn completing can't float a row. Branch children still nest under
   * their parent; sibling branches stay ordered by their own recency.
   */
  preserveOrder?: boolean
}

const recency = (session: SessionInfo): number => session.last_active || session.started_at || 0

// Profile-qualified compression lineage, same key as mergeSessionPage (store/session.ts). The
// backend's `_lineage_root_id` follows compression edges only; /branch and /new children get
// their own root, so rows sharing a key are one conversation, never a fork.
const lineageKey = (session: SessionInfo): string =>
  `${(session.profile ?? '').trim() || 'default'}::${session._lineage_root_id?.trim() || session.id}`

/**
 * One row per compression lineage. A stale tip can outlive its rotation in the store (#82290);
 * keep the row nothing in its lineage continues from, then the freshest by mergeSessionPage's
 * recency rule (first in input on a tie).
 */
function collapseCompressionLineages(sessions: readonly SessionInfo[]): readonly SessionInfo[] {
  const groups = new Map<string, SessionInfo[]>()

  for (const session of sessions) {
    const key = lineageKey(session)
    const group = groups.get(key)

    if (group) {
      group.push(session)
    } else {
      groups.set(key, [session])
    }
  }

  if (groups.size === sessions.length) {
    return sessions
  }

  const winners = new Set<SessionInfo>()

  for (const group of groups.values()) {
    const continuedIds = new Set(group.map(session => session.parent_session_id?.trim()))

    const score = (session: SessionInfo) =>
      continuedIds.has(session.id) ? -Infinity : Math.max(session.last_active || 0, session.started_at || 0)

    winners.add(group.reduce((best, session) => (score(session) > score(best) ? session : best)))
  }

  return sessions.filter(session => winners.has(session))
}

/** Flat list with branch/fork sessions nested visually under their parent. */
export function flattenSessionsWithBranches(
  input: readonly SessionInfo[],
  options: FlattenSessionsOptions = {}
): SidebarSessionEntry[] {
  const sessions = collapseCompressionLineages(input)

  if (sessions.length < 2) {
    return sessions.map(session => ({ session }))
  }

  const byVisibleId = new Map<string, SessionInfo>()

  for (const session of sessions) {
    byVisibleId.set(session.id, session)
    const rootId = session._lineage_root_id?.trim()

    if (rootId) {
      byVisibleId.set(rootId, session)
    }
  }

  const childrenByParent = new Map<string, SessionInfo[]>()
  const nestedIds = new Set<string>()

  for (const session of sessions) {
    const parentId = session.parent_session_id?.trim()

    if (!parentId) {
      continue
    }

    const parent = byVisibleId.get(parentId)

    // Compression ancestry is not a branch: never nest a row under its own lineage.
    if (!parent || lineageKey(parent) === lineageKey(session)) {
      continue
    }

    nestedIds.add(session.id)
    const siblings = childrenByParent.get(parent.id) ?? []
    siblings.push(session)
    childrenByParent.set(parent.id, siblings)
  }

  for (const siblings of childrenByParent.values()) {
    siblings.sort((left, right) => recency(right) - recency(left))
  }

  // A group sorts by its freshest member, so activity on any branch lifts the
  // whole parent→branches cluster together instead of stranding the parent at
  // its own stale timestamp. Memoized — each subtree is folded at most once.
  // Skipped when preserveOrder is set: the caller already chose positions.
  const groupRecencyMemo = new Map<string, number>()

  const groupRecency = (session: SessionInfo): number => {
    const cached = groupRecencyMemo.get(session.id)

    if (cached !== undefined) {
      return cached
    }

    groupRecencyMemo.set(session.id, recency(session)) // cycle guard

    const max = (childrenByParent.get(session.id) ?? []).reduce(
      (acc, child) => Math.max(acc, groupRecency(child)),
      recency(session)
    )

    groupRecencyMemo.set(session.id, max)

    return max
  }

  // Depth-first so a branch-of-a-branch still renders under its own parent. The
  // `seen` set guards against pathological parent cycles, and the trailing sweep
  // emits anything the walk somehow missed — nothing past the lineage collapse is dropped.
  const out: SidebarSessionEntry[] = []
  const seen = new Set<string>()

  const emit = (session: SessionInfo, branchStem?: string) => {
    if (seen.has(session.id)) {
      return
    }

    seen.add(session.id)
    out.push(branchStem ? { branchStem, session } : { session })

    const children = childrenByParent.get(session.id)
    children?.forEach((child, index) => emit(child, index === children.length - 1 ? '└─ ' : '├─ '))
  }

  const roots = sessions.filter(session => !nestedIds.has(session.id)).map((session, index) => ({ index, session }))

  if (!options.preserveOrder) {
    roots.sort((a, b) => groupRecency(b.session) - groupRecency(a.session) || a.index - b.index)
  }

  roots.forEach(({ session }) => emit(session))

  for (const session of sessions) {
    if (!seen.has(session.id)) {
      out.push({ session })
    }
  }

  return out
}
