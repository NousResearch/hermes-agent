import type { SessionInfo } from '@/types/hermes'
import { sessionIdentityKey, sessionLineageIdentityKey, sessionOwnerIdentityKey } from '@/store/session'

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

/** Flat list with branch/fork sessions nested visually under their parent. */
export function flattenSessionsWithBranches(
  sessions: readonly SessionInfo[],
  options: FlattenSessionsOptions = {}
): SidebarSessionEntry[] {
  if (sessions.length < 2) {
    return sessions.map(session => ({ session }))
  }

  const byVisibleIdentity = new Map<string, SessionInfo>()
  const byVisibleId = new Map<string, SessionInfo>()
  const aliasOwners = new Map<string, string>()
  const ambiguousAliases = new Set<string>()

  const addAlias = (alias: string, session: SessionInfo) => {
    if (ambiguousAliases.has(alias)) {
      return
    }

    const owner = sessionOwnerIdentityKey(session)
    const existingOwner = aliasOwners.get(alias)

    if (!existingOwner) {
      aliasOwners.set(alias, owner)
      byVisibleId.set(alias, session)

      return
    }

    if (existingOwner === owner) {
      byVisibleId.set(alias, session)

      return
    }

    aliasOwners.delete(alias)
    ambiguousAliases.add(alias)
    byVisibleId.delete(alias)
  }

  const exactSessionKey = (session: SessionInfo, id: string): string =>
    sessionIdentityKey({ connection_id: session.connection_id, id, profile: session.profile })

  const exactLineageKey = (session: SessionInfo, id: string): string =>
    sessionLineageIdentityKey({ connection_id: session.connection_id, id, profile: session.profile })

  for (const session of sessions) {
    byVisibleIdentity.set(sessionIdentityKey(session), session)
    addAlias(session.id, session)
    const rootId = session._lineage_root_id?.trim()

    if (rootId) {
      byVisibleIdentity.set(exactLineageKey(session, rootId), session)
      addAlias(rootId, session)
    }
  }

  const findParent = (child: SessionInfo, parentId: string): SessionInfo | undefined => {
    const exactParent =
      byVisibleIdentity.get(exactSessionKey(child, parentId)) ?? byVisibleIdentity.get(exactLineageKey(child, parentId))

    if (exactParent) {
      return exactParent
    }

    // A tagged child carries authoritative routing. A bare id from another
    // gateway is not enough evidence to attach it to a visible parent.
    return child.connection_id?.trim() || child.profile?.trim() ? undefined : byVisibleId.get(parentId)
  }

  const childrenByParent = new Map<string, SessionInfo[]>()
  const nestedIds = new Set<string>()

  for (const session of sessions) {
    const parentId = session.parent_session_id?.trim()

    if (!parentId) {
      continue
    }

    const parent = findParent(session, parentId)

    if (!parent || sessionIdentityKey(parent) === sessionIdentityKey(session)) {
      continue
    }

    const childIdentity = sessionIdentityKey(session)
    const parentIdentity = sessionIdentityKey(parent)
    nestedIds.add(childIdentity)
    const siblings = childrenByParent.get(parentIdentity) ?? []
    siblings.push(session)
    childrenByParent.set(parentIdentity, siblings)
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
    const identity = sessionIdentityKey(session)
    const cached = groupRecencyMemo.get(identity)

    if (cached !== undefined) {
      return cached
    }

    groupRecencyMemo.set(identity, recency(session)) // cycle guard

    const max = (childrenByParent.get(identity) ?? []).reduce(
      (acc, child) => Math.max(acc, groupRecency(child)),
      recency(session)
    )

    groupRecencyMemo.set(identity, max)

    return max
  }

  // Depth-first so a branch-of-a-branch still renders under its own parent. The
  // `seen` set guards against pathological parent cycles, and the trailing sweep
  // emits anything the walk somehow missed — nothing in the input is ever dropped.
  const out: SidebarSessionEntry[] = []
  const seen = new Set<string>()

  const emit = (session: SessionInfo, branchStem?: string) => {
    const identity = sessionIdentityKey(session)

    if (seen.has(identity)) {
      return
    }

    seen.add(identity)
    out.push(branchStem ? { branchStem, session } : { session })

    const children = childrenByParent.get(identity)
    children?.forEach((child, index) => emit(child, index === children.length - 1 ? '└─ ' : '├─ '))
  }

  const roots = sessions
    .filter(session => !nestedIds.has(sessionIdentityKey(session)))
    .map((session, index) => ({ index, session }))

  if (!options.preserveOrder) {
    roots.sort((a, b) => groupRecency(b.session) - groupRecency(a.session) || a.index - b.index)
  }

  roots.forEach(({ session }) => emit(session))

  for (const session of sessions) {
    if (!seen.has(sessionIdentityKey(session))) {
      out.push({ session })
    }
  }

  return out
}
