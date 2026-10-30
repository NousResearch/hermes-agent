/**
 * Bulk disband for Bot Mode group chats.
 * Every room goes through the same disbandGroupChat path as single-delete
 * (tombstones + metadata plan + sync projection).
 * Sequential on purpose: shared ui_meta / profiles.configure CAS.
 */

import { disbandGroupChat } from './group-chat-view'
import type { GroupMember } from './types'

export type GroupDisbandTarget = {
  members: GroupMember[]
  name: string
}

export type DisbandGroupChatFn = (group: string, members: GroupMember[]) => void | Promise<void>

/** Disband many rooms through the same lifecycle as one row Delete. */
export async function disbandGroupChats(
  groups: GroupDisbandTarget[],
  disband: DisbandGroupChatFn = disbandGroupChat
): Promise<{ disbanded: string[] }> {
  const disbanded: string[] = []
  const seen = new Set<string>()

  for (const { members, name } of groups) {
    const key = String(name || '').trim()
    if (!key || seen.has(key)) {
      continue
    }
    seen.add(key)
    await disband(key, Array.isArray(members) ? members : [])
    disbanded.push(key)
  }

  return { disbanded }
}

/** Click helper for roster multi-select on group rows (pure, testable). */
export function nextGroupSelection(
  prev: ReadonlySet<string>,
  orderedNames: readonly string[],
  target: string,
  mod: { ctrlKey: boolean; metaKey?: boolean; shiftKey: boolean },
  anchor: null | string
): { anchor: string; selected: Set<string> } {
  const next = new Set(prev)
  const toggle = Boolean(mod.ctrlKey || mod.metaKey)

  if (mod.shiftKey && anchor && orderedNames.includes(anchor) && orderedNames.includes(target)) {
    const a = orderedNames.indexOf(anchor)
    const b = orderedNames.indexOf(target)
    const lo = Math.min(a, b)
    const hi = Math.max(a, b)
    if (!toggle) {
      next.clear()
    }
    for (let i = lo; i <= hi; i++) {
      next.add(orderedNames[i])
    }
    return { anchor, selected: next }
  }

  if (toggle) {
    if (next.has(target)) {
      next.delete(target)
    } else {
      next.add(target)
    }
    return { anchor: target, selected: next }
  }

  return { anchor: target, selected: new Set([target]) }
}

/** Build confirm targets for the current roster projection. */
export function buildGroupDisbandTargets(
  names: readonly string[],
  membersFor: (name: string) => GroupMember[]
): GroupDisbandTarget[] {
  const out: GroupDisbandTarget[] = []
  const seen = new Set<string>()
  for (const raw of names) {
    const name = String(raw || '').trim()
    if (!name || seen.has(name)) {
      continue
    }
    seen.add(name)
    out.push({ name, members: membersFor(name) || [] })
  }
  return out
}
