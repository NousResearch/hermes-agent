import type { SessionInfo } from '@/types/hermes'

export interface SidebarSessionEntry {
  branchDepth?: number
  branchStem?: string
  hasChildren?: boolean
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
  /** Whether to render a session's descendants. Defaults to open. */
  isOpen?: (session: SessionInfo) => boolean
}

const recency = (session: SessionInfo): number => session.last_active || session.started_at || 0

const sessionKey = (session: SessionInfo, id = session.id) => {
  const connection = session.connection_id?.trim()

  return `${!connection || connection === 'local' ? 'local' : connection}::${session.profile || 'default'}::${id}`
}

export const sessionTreeNodeId = (session: SessionInfo): string => {
  const durableId = session._lineage_root_id?.trim() || session.id

  return `session-tree:${sessionKey(session, durableId)}`
}

/** Flat list with branch and spawned sessions nested visually under their parent. */
export function flattenSessionsWithBranches(
  sessions: readonly SessionInfo[],
  options: FlattenSessionsOptions = {}
): SidebarSessionEntry[] {
  if (sessions.length < 2) {
    return sessions.map(session => ({ session }))
  }

  const byVisibleId = new Map<string, SessionInfo>()

  for (const session of sessions) {
    for (const id of session._lineage_ids ?? [session._lineage_root_id, session.id]) {
      if (id?.trim()) {
        byVisibleId.set(sessionKey(session, id), session)
      }
    }
  }

  const childrenByParent = new Map<string, SessionInfo[]>()
  const nestedIds = new Set<string>()

  for (const session of sessions) {
    const parentId = (session.spawned_by_session_id || session.parent_session_id)?.trim()

    if (!parentId) {
      continue
    }

    const parent = byVisibleId.get(sessionKey(session, parentId))

    if (!parent || parent.id === session.id) {
      continue
    }

    nestedIds.add(sessionKey(session))
    const parentKey = sessionKey(parent)
    const siblings = childrenByParent.get(parentKey) ?? []
    siblings.push(session)
    childrenByParent.set(parentKey, siblings)
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
    const key = sessionKey(session)
    const cached = groupRecencyMemo.get(key)

    if (cached !== undefined) {
      return cached
    }

    groupRecencyMemo.set(key, recency(session)) // cycle guard

    const max = (childrenByParent.get(key) ?? []).reduce(
      (acc, child) => Math.max(acc, groupRecency(child)),
      recency(session)
    )

    groupRecencyMemo.set(key, max)

    return max
  }

  // Depth-first so a branch-of-a-branch still renders under its own parent. The
  // `seen` set guards against pathological parent cycles, and the trailing sweep
  // emits anything the walk somehow missed — nothing in the input is ever dropped.
  const out: SidebarSessionEntry[] = []
  const seen = new Set<string>()

  const suppress = (session: SessionInfo) => {
    const key = sessionKey(session)

    if (seen.has(key)) {
      return
    }

    seen.add(key)
    childrenByParent.get(key)?.forEach(suppress)
  }

  const emit = (session: SessionInfo, branchDepth = 0, branchStem?: string) => {
    const key = sessionKey(session)

    if (seen.has(key)) {
      return
    }

    const children = childrenByParent.get(key)
    const entry: SidebarSessionEntry = branchStem ? { branchDepth, branchStem, session } : { session }

    seen.add(key)

    if (children?.length) {
      entry.hasChildren = true
    }

    out.push(entry)

    if (children?.length && (options.isOpen?.(session) ?? true)) {
      children.forEach((child, index) => emit(child, branchDepth + 1, index === children.length - 1 ? '└─ ' : '├─ '))
    } else {
      children?.forEach(suppress)
    }
  }

  const roots = sessions
    .filter(session => !nestedIds.has(sessionKey(session)))
    .map((session, index) => ({ index, session }))

  if (!options.preserveOrder) {
    roots.sort((a, b) => groupRecency(b.session) - groupRecency(a.session) || a.index - b.index)
  }

  roots.forEach(({ session }) => emit(session))

  for (const session of sessions) {
    if (!seen.has(sessionKey(session))) {
      out.push({ session })
    }
  }

  return out
}
