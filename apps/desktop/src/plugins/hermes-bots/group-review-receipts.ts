/** Review receipts belong to a member's hidden room session, not its Bot Chat.
 * Events only invalidate: never trust an event's text as room history. Re-read
 * the exact stored session on its captured source, so private chat reviews and
 * identically named bots on other connections cannot leak into a room. */
import { host } from '@hermes/plugin-sdk'

import { $groupChats, updateGroupChat } from './group-chat'
import { followGroupChat, groupMemberKey } from './group-membership'
import { botConnectionRoute } from './routing'
import type { GroupMember, GroupReviewReceipt, ProfileRoute } from './types'

export function mergeGroupReviewReceipts(
  previous: GroupReviewReceipt[], incoming: GroupReviewReceipt[]
): GroupReviewReceipt[] {
  const byId = new Map(previous.map(receipt => [receipt.id, receipt]))

  for (const receipt of incoming) {
    byId.set(receipt.id, receipt)
  }

  return [...byId.values()].sort((a, b) => a.at - b.at || a.id.localeCompare(b.id)).slice(-50)
}

export function readGroupReviewReceipts(rows: unknown, member: GroupMember, owner: string): GroupReviewReceipt[] {
  if (!Array.isArray(rows)) {
    return []
  }

  return rows.flatMap(row => {
    if (!row || row.display_kind !== 'review_summary') {
      return []
    }

    const id = row.display_metadata?.review_id
    const text = typeof row.content === 'string' ? row.content : row.text
    const at = row.timestamp

    if (typeof id !== 'string' || !id || typeof text !== 'string' || !text.trim() ||
        typeof at !== 'number' || !Number.isFinite(at) || at <= 0) {
      return []
    }

    return [{ id: `${owner}:${id}`, at: at * 1000, text, member: member.name, memberKey: groupMemberKey(member) }]
  })
}

function captureRoute(member: GroupMember): ProfileRoute | null {
  try {
    const route = botConnectionRoute(member)

    if (route) {
      return { ...route }
    }

    // Capture the ambient source ONCE for legacy local roster rows; a later
    // foreground switch must never retarget an already-started read.
    const connectionId = host.activeConnectionId?.() || 'local'

    return { connectionId, mode: connectionId === 'local' ? 'local' : 'remote',
      profile: member.name, targetProfile: member.name }
  } catch {
    return null
  }
}

export function watchGroupReviewReceipts(group: string, members: GroupMember[]): () => void {
  if (typeof host.listReviewSummaries !== 'function') {
    return () => undefined // older shell: cached receipts remain readable
  }

  let disposed = false
  const releases: Array<() => void> = []
  const binding = followGroupChat(group, name => { group = name })

  const owners = members.flatMap(member => {
    const route = captureRoute(member)

    return route ? [{ member, route, key: groupMemberKey(member) }] : []
  })

  let running = false
  let dirty = false

  const refresh = async () => {
    dirty = true

    if (running || disposed || !binding.isLive()) {
      return
    }

    running = true

    try {
      while (dirty && !disposed && binding.isLive()) {
        dirty = false
        await Promise.all(owners.map(async ({ member, route, key }) => {
          const stored = $groupChats.get()[group]?.sessions?.[key]

          if (typeof stored !== 'string' || !stored) {
            return
          }

          try {
            const result = await host.listReviewSummaries(route, stored)

            if (disposed || !binding.isLive() || $groupChats.get()[group]?.sessions?.[key] !== stored) {
              return
            }

            const receipts = readGroupReviewReceipts(result.messages, member,
              `${route.connectionId}:${route.targetProfile}:${key}`)

            const previous = $groupChats.get()[group]?.reviewReceipts || []
            const merged = mergeGroupReviewReceipts(previous, receipts)

            if (JSON.stringify(merged) !== JSON.stringify(previous)) {
              updateGroupChat(group, room => ({ ...room, reviewReceipts: merged }), { sync: false })
            }
          } catch {
            // Old backend, missing/removed profile or transient connection:
            // preserve the cache; next summary/reconnect/open retries it.
          }
        }))
      }
    } finally {
      running = false
    }
  }

  if (typeof host.onEvent === 'function') {
    for (const type of ['review.summary', 'gateway.ready', 'message.complete']) {
      releases.push(host.onEvent(type, event => {
        if (!event.connectionId || owners.some(owner => owner.route.connectionId === event.connectionId)) {
          void refresh()
        }
      }))
    }
  }

  // Keep event taps alive while the room is visible. This is not ownership
  // of agent work: the backend independently protects reviews after close.
  if (typeof host.retainProfile === 'function') {
    for (const { route } of owners) {
      void host.retainProfile(route).then(release => {
        if (disposed) {
          release()
        } else {
          releases.push(release)
          void refresh() // a newly-connected route may have missed the first read
        }
      }).catch(() => undefined)
    }
  }

  void refresh()

  return () => {
    disposed = true
    binding.dispose()

    for (const release of releases) {
      release()
    }
  }
}
