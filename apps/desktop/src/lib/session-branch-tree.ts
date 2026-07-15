import type { SessionInfo } from '@/types/hermes'

import { sessionIdentityKey } from './session-identity'

export interface SidebarSessionEntry {
  branchStem?: string
  session: SessionInfo
}

const recency = (session: SessionInfo): number => session.last_active || session.started_at || 0

/** Flat list with branch/fork sessions nested visually under their parent. */
export function flattenSessionsWithBranches(sessions: readonly SessionInfo[]): SidebarSessionEntry[] {
  if (sessions.length < 2) {
    return sessions.map(session => ({ session }))
  }

  const byVisibleId = new Map<string, SessionInfo>()

  for (const session of sessions) {
    byVisibleId.set(sessionIdentityKey(session.id, session.profile), session)
    const rootId = session._lineage_root_id?.trim()

    if (rootId) {
      byVisibleId.set(sessionIdentityKey(rootId, session.profile), session)
    }
  }

  const childrenByParent = new Map<string, SessionInfo[]>()
  const nestedIds = new Set<string>()

  for (const session of sessions) {
    const parentId = session.parent_session_id?.trim()

    if (!parentId) {
      continue
    }

    const parent = byVisibleId.get(sessionIdentityKey(parentId, session.profile))

    if (!parent || parent.id === session.id) {
      continue
    }

    nestedIds.add(sessionIdentityKey(session.id, session.profile))
    const parentKey = sessionIdentityKey(parent.id, parent.profile)
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
  const groupRecencyMemo = new Map<string, number>()

  const groupRecency = (session: SessionInfo): number => {
    const key = sessionIdentityKey(session.id, session.profile)
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

  const emit = (session: SessionInfo, branchStem?: string) => {
    const key = sessionIdentityKey(session.id, session.profile)

    if (seen.has(key)) {
      return
    }

    seen.add(key)
    out.push(branchStem ? { branchStem, session } : { session })

    const children = childrenByParent.get(key)
    children?.forEach((child, index) => emit(child, index === children.length - 1 ? '└─ ' : '├─ '))
  }

  sessions
    .filter(session => !nestedIds.has(sessionIdentityKey(session.id, session.profile)))
    .map((session, index) => ({ index, session }))
    .sort((a, b) => groupRecency(b.session) - groupRecency(a.session) || a.index - b.index)
    .forEach(({ session }) => emit(session))

  for (const session of sessions) {
    if (!seen.has(sessionIdentityKey(session.id, session.profile))) {
      out.push({ session })
    }
  }

  return out
}
