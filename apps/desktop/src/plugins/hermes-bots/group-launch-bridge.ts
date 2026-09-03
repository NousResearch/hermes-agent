/**
 * Rooms for the launchers — Quick Entry's group tiles and the HUD's room
 * switcher both ask THIS window (the primary renderer, where the plugin's
 * room store lives) for its rooms, and to open one.
 *
 * The contract is the three window events use-quick-entry-bridge already
 * dispatches; until now nothing answered them, so the launchers listed no
 * rooms. The bridge answers from `$groupChats`, opens through the same
 * `openGroupChat` a roster click uses, and announces changes so the cached
 * list in main stays fresh.
 */

import { $botMeta, cachedUnionRoster } from './data'
import { $groupChats } from './group-chat'
import { openGroupChat } from './group-chat-view'
import { sendToGroupChat } from './group-rounds'
import type { GroupChat, GroupMember, GroupMessage } from './types'

export const GROUPS_REQUEST_EVENT = 'hermes:quick-entry:groups-request'
export const GROUPS_CHANGED_EVENT = 'hermes:quick-entry:groups-changed'
export const GROUP_OPEN_EVENT = 'hermes:quick-entry:group-open'
export const AGENTS_REQUEST_EVENT = 'hermes:quick-entry:agents-request'
export const AGENTS_CHANGED_EVENT = 'hermes:quick-entry:agents-changed'
/** The HUD talking INTO a room: post a line, read the recent log. */
export const GROUP_POST_EVENT = 'hermes:quick-entry:group-post'
export const GROUP_FEED_REQUEST_EVENT = 'hermes:quick-entry:group-feed-request'
export const GROUP_FEED_CHANGED_EVENT = 'hermes:quick-entry:group-feed-changed'

export const GROUP_FEED_LIMIT = 40

export interface GroupFeedEntry {
  at: number
  author: string
  id: string
  kind: 'member' | 'user'
  text: string
}

export interface GroupFeed {
  entries: GroupFeedEntry[]
  groupId: string
  members: string[]
  running: boolean
  turn: null | string
}

/** The members a launcher post goes to: the room's durable roster, else the
 *  live roster rows filed under this group (rooms made before rosters were
 *  persisted). Same descriptors the room view hands sendToGroupChat. */
export function groupMembersForPost(group: string, rooms: Record<string, GroupChat> = $groupChats.get()): GroupMember[] {
  const durable = rooms[group]?.members

  if (Array.isArray(durable) && durable.length) {
    return durable
  }

  const meta = $botMeta.get()
  const roster = cachedUnionRoster()
  const profiles = Array.isArray(roster?.profiles) ? roster.profiles : []

  return profiles.filter(row => {
    const groups = meta[row.name]?.groups

    return Array.isArray(groups) ? groups.includes(group) : meta[row.name]?.group === group
  }) as GroupMember[]
}

function feedEntry(entry: GroupMessage, index: number): GroupFeedEntry {
  return {
    at: Number(entry.at) || 0,
    author: entry.from?.name || '',
    id: entry.id || `${entry.at}-${index}`,
    kind: entry.from?.kind === 'user' ? 'user' : 'member',
    text: String(entry.text || '')
  }
}

export function groupFeed(group: string, rooms: Record<string, GroupChat> = $groupChats.get()): GroupFeed | null {
  const room = rooms[group]

  if (!room || room.tombstone || !Array.isArray(room.log)) {
    return null
  }

  const start = Math.max(0, room.log.length - GROUP_FEED_LIMIT)

  return {
    entries: room.log.slice(start).map((entry, index) => feedEntry(entry, start + index)),
    groupId: group,
    members: (Array.isArray(room.members) ? room.members : []).map(member => member.name).filter(Boolean),
    running: room.running === true,
    turn: typeof (room as { turn?: unknown }).turn === 'string' ? ((room as { turn?: string }).turn ?? null) : null
  }
}

/** Post a line into a room from a launcher, exactly as the room view's
 *  composer would: new thread, no images. Returns false for an unknown room
 *  or one with nobody in it. */
export function postToGroupFromLauncher(group: string, text: string): boolean {
  const trimmed = String(text || '').trim()

  if (!trimmed || !liveGroupRooms().includes(group)) {
    return false
  }

  const members = groupMembersForPost(group)

  if (!members.length) {
    return false
  }

  return sendToGroupChat(group, members, trimmed, null) !== null
}

export interface AgentDecoration {
  image?: string
  title?: string
}

/** Per-profile presentation the launchers can show: the Bot Mode title and
 *  the avatar image (data URL) when the user set one. Keys are lower-cased
 *  profile names, the same normalisation the launchers route by. */
export function agentDecorations(meta = $botMeta.get()): Record<string, AgentDecoration> {
  const out: Record<string, AgentDecoration> = {}

  for (const [name, bot] of Object.entries(meta || {})) {
    if (!bot || typeof bot !== 'object') {
      continue
    }

    const title = typeof bot.title === 'string' && bot.title.trim() ? bot.title.trim() : undefined
    const image = typeof bot.image === 'string' && bot.image.startsWith('data:image/') ? bot.image : undefined

    if (title || image) {
      out[name.trim().toLowerCase()] = { ...(title ? { title } : {}), ...(image ? { image } : {}) }
    }
  }

  return out
}

export interface GroupLaunchOption {
  displayName: string
  groupId: string
  memberCount?: number
  reachable: boolean
}

/** Live rooms only: a disband tombstone is coordination state, not a room. */
export function liveGroupRooms(rooms: Record<string, GroupChat> = $groupChats.get()): string[] {
  return Object.entries(rooms || {})
    .filter(([, room]) => room && !room.tombstone && Array.isArray(room.log))
    .map(([name]) => name)
    .sort((a, b) => a.localeCompare(b))
}

export function groupLaunchOptions(rooms: Record<string, GroupChat> = $groupChats.get()): GroupLaunchOption[] {
  return liveGroupRooms(rooms).map(name => {
    const room = rooms[name]

    return {
      displayName: name,
      groupId: name,
      memberCount: Array.isArray(room?.members) ? room.members.length : undefined,
      reachable: true
    }
  })
}

interface GroupsRequestDetail {
  respond: (options: GroupLaunchOption[]) => void
}

interface GroupOpenDetail {
  groupId: string
  respond: (ok: boolean) => void
}

export function bindGroupLaunchBridge(open: (group: string) => void = openGroupChat): () => void {
  if (typeof window === 'undefined') {
    return () => {}
  }

  const onRequest = (event: Event) => {
    const detail = (event as CustomEvent<GroupsRequestDetail>).detail

    if (typeof detail?.respond === 'function') {
      detail.respond(groupLaunchOptions())
    }
  }

  const onOpen = (event: Event) => {
    const detail = (event as CustomEvent<GroupOpenDetail>).detail
    const groupId = typeof detail?.groupId === 'string' ? detail.groupId.trim() : ''
    const ok = Boolean(groupId) && liveGroupRooms().includes(groupId)

    if (ok) {
      open(groupId)
    }

    if (typeof detail?.respond === 'function') {
      detail.respond(ok)
    }
  }

  const onAgents = (event: Event) => {
    const detail = (event as CustomEvent<{ respond?: (value: Record<string, AgentDecoration>) => void }>).detail

    if (typeof detail?.respond === 'function') {
      detail.respond(agentDecorations())
    }
  }

  const onPost = (event: Event) => {
    const detail = (event as CustomEvent<{ groupId?: string; respond?: (ok: boolean) => void; text?: string }>).detail
    const ok = postToGroupFromLauncher(String(detail?.groupId ?? '').trim(), String(detail?.text ?? ''))

    if (typeof detail?.respond === 'function') {
      detail.respond(ok)
    }
  }

  const onFeed = (event: Event) => {
    const detail = (event as CustomEvent<{ groupId?: string; respond?: (feed: GroupFeed | null) => void }>).detail

    if (typeof detail?.respond === 'function') {
      detail.respond(groupFeed(String(detail?.groupId ?? '').trim()))
    }
  }

  window.addEventListener(GROUPS_REQUEST_EVENT, onRequest)
  window.addEventListener(GROUP_OPEN_EVENT, onOpen)
  window.addEventListener(AGENTS_REQUEST_EVENT, onAgents)
  window.addEventListener(GROUP_POST_EVENT, onPost)
  window.addEventListener(GROUP_FEED_REQUEST_EVENT, onFeed)

  const offRooms = $groupChats.listen(() => {
    window.dispatchEvent(new Event(GROUPS_CHANGED_EVENT))
    window.dispatchEvent(new Event(GROUP_FEED_CHANGED_EVENT))
  })

  const offMeta = $botMeta.listen(() => {
    window.dispatchEvent(new Event(AGENTS_CHANGED_EVENT))
  })

  return () => {
    window.removeEventListener(GROUPS_REQUEST_EVENT, onRequest)
    window.removeEventListener(GROUP_OPEN_EVENT, onOpen)
    window.removeEventListener(AGENTS_REQUEST_EVENT, onAgents)
    window.removeEventListener(GROUP_POST_EVENT, onPost)
    window.removeEventListener(GROUP_FEED_REQUEST_EVENT, onFeed)
    offRooms()
    offMeta()
  }
}
