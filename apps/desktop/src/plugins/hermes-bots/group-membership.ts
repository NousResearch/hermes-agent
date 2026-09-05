/**
 * Who is in a room: the membership keys, the ui_meta group lists each bot
 * carries, and the roster→member descriptor derivation the room engine and
 * the roster UI both read.
 */

import { $botMeta, $lastRoster, botFriendlyNames, botMetaKey, botRosterKey, persistBotMetaSnapshot, saveBotMeta } from './data'
import { $groupChats, GROUP_CHAT_MAX_MEMBERS, groupChatRoomKey, updateGroupChat } from './group-chat'
import { displayName } from './labels'
import { botConnectionRoute, botRosterMeta, resolveBotConnectionRoute } from './routing'
import type { BotMeta, GroupChat, GroupMember, RosterRow } from './types'

export function groupWorkspaceOwnerKey(group: string) {
  return `group:${groupChatRoomKey(group, $groupChats.get()[group])}`
}

/** Stable per-member identity inside a group room. Local members keep their
 *  bare name (compat with rooms persisted before cross-connection groups);
 *  remote members get the source-qualified key so `dixie` on the Mini and a
 *  local `dixie` never share watermarks or sessions. */
export function groupMemberKey(member: GroupMember): string {
  return member?.sourceScoped || member?.remoteSource ? botRosterKey(member) : member?.name
}

/** Serializable immutable owner captured beside every group plumbing session. */
export function groupSessionOwner(member: RosterRow) {
  const route = botConnectionRoute(member)
  const name = String(route?.profile || member?.name || '').trim() || 'default'

  if (!route) {
    return {
      name
    }
  }

  return {
    connectionId: route.connectionId,
    name,
    sourceScoped: true,
    remoteSource: route.mode !== 'local',
    route: {
      ...route
    }
  }
}

/** Canonical multi-group read with legacy scalar compatibility. Profiles that
 *  predate `groups` still fall back to `group`; once the canonical array exists,
 *  it is authoritative. Writes keep `group` as a first-membership projection so
 *  older desktops can still display one room without corrupting the array. */
export function botGroups(meta?: BotMeta | null) {
  const groups: string[] = []
  const seen = new Set<string>()
  const values = Array.isArray(meta?.groups) ? meta.groups : [meta?.group]

  for (const value of values) {
    if (typeof value !== 'string') {
      continue
    }

    const group = value.trim()

    if (group && !seen.has(group)) {
      seen.add(group)
      groups.push(group)
    }
  }

  return groups
}

interface GroupMembershipPatch {
  /** Legacy single-group scalar, kept as a first-membership projection. */
  group: null | string
  groups: string[]
}

export function groupMembershipPatch(
  meta: BotMeta | null | undefined,
  group: string,
  enabled: boolean
): GroupMembershipPatch {
  const name = String(group || '').trim()
  let groups = botGroups(meta)

  if (enabled) {
    if (name && !groups.includes(name)) {
      groups = [...groups, name]
    }
  } else {
    groups = groups.filter(existing => existing !== name)
  }

  return {
    groups,
    group: groups[0] || null
  }
}

/** Build the exact metadata cleanup for a room disband. The rendered member
 *  list is only a presentation snapshot and can already be empty on a remote
 *  connection while bot metadata still names the room. Durable room members
 *  and the full roster recover source-qualified owners; every remaining
 *  metadata record is still cleared locally, but an unresolved scoped key is
 *  never guessed into a server route. */
export function groupDisbandMetadataPlan(
  group: string,
  members: RosterRow[],
  room: GroupChat,
  roster: RosterRow[],
  metaByName: Record<string, BotMeta>
) {
  const owners = new Map<string, RosterRow>()
  const patches = new Map<string, GroupMembershipPatch>()

  const rememberOwner = (owner: RosterRow, required = false) => {
    if (!owner?.name) {
      return
    }

    let key: string

    try {
      key = botMetaKey(owner)
    } catch {
      return
    }

    const meta = metaByName?.[key] || botRosterMeta(owner, metaByName) || {}

    if (!required && !botGroups(meta).includes(group)) {
      return
    }

    if (!owners.has(key)) {
      owners.set(key, owner)
    }

    patches.set(key, groupMembershipPatch(meta, group, false))
  }

  for (const owner of members || []) {
    rememberOwner(owner, true)
  }

  for (const owner of room?.members || []) {
    rememberOwner(owner, true)
  }

  for (const owner of roster || []) {
    rememberOwner(owner)
  }

  // Metadata itself is the final source of a metadata-only `0 bots` row.
  // Clear every record that names the room even when its exact server owner is
  // temporarily absent from the roster. Known owners still get a routed
  // profiles.configure write below; unknown scoped records remain local-only.
  for (const [key, meta] of Object.entries(metaByName || {})) {
    if (botGroups(meta).includes(group)) {
      patches.set(key, groupMembershipPatch(meta, group, false))
    }
  }

  return {
    owners,
    patches
  }
}

/** Group chats that should hold a roster row: every group named in bot meta
 *  (local members) plus every room record that still has stored members or
 *  log — cross-connection rooms whose members can't ride bot-meta. */
export function groupChatNames(metaByName: Record<string, BotMeta>, rooms: Record<string, GroupChat>) {
  const names = new Set(knownGroups(metaByName))

  for (const [name, room] of Object.entries(rooms || {})) {
    if (room?.tombstone) {
      continue
    }

    if ((Array.isArray(room?.members) && room.members.length) || (Array.isArray(room?.log) && room.log.length)) {
      names.add(name)
    }
  }

  return [...names]
}

/** Names of REAL rooms in the atom — disband tombstones excluded. Feeds the
 *  create/rename collision sets so a just-disbanded name is immediately
 *  reusable even while an in-flight drive's tombstone still holds its key. */
export function liveGroupChatNames() {
  return Object.entries($groupChats.get())
    .filter(([, room]) => !room?.tombstone)
    .map(([name]) => name)
}

/** Millisecond timestamp of a room's newest log entry (0 for a silent room) —
 *  the group's recency key, competing in the same ordering as bot rows. */
export function groupLastActivity(room?: GroupChat | null) {
  const log = Array.isArray(room?.log) ? room.log : []

  return log.length ? log[log.length - 1].at || 0 : 0
}

/** A member's seat key, or '' when the descriptor carries no name. botRosterKey
 *  synthesizes `legacy::default` for a nameless row, which would seat one
 *  malformed descriptor and then silently swallow every later one as a
 *  duplicate. Callers that write membership must skip these rather than
 *  persist a bot that does not exist. */
function memberSeatKey(bot: Partial<RosterRow> | null | undefined): string {
  return String(bot?.name || '').trim() ? botRosterKey(bot) : ''
}

/** Seat a group's member roster: local bots whose meta names the group, plus
 *  the room record's stored descriptors (remote members can't ride bot-meta).
 *  Prefers the LIVE roster row for a stored descriptor when present. */
export function groupChatMemberBots(
  group: string,
  roster: RosterRow[],
  metaByName: Record<string, BotMeta>
): RosterRow[] {
  const room = $groupChats.get()[group] || {}
  const storedMembers = room.members || []

  // A roomId-backed room persists every source-qualified member, so its
  // non-empty stored roster is AUTHORITATIVE. Falling through to the legacy
  // union below would re-seat a bot that was just removed, because that bot's
  // own ui_meta still names the group until its write lands.
  if (storedMembers.length && room.roomId) {
    const seatedKeys = new Set<string>()
    const members: RosterRow[] = []

    for (const descriptor of storedMembers) {
      const resolved = resolveLegacyMemberDescriptor(descriptor, roster)
      const key = memberSeatKey(resolved)

      if (!key || seatedKeys.has(key)) {
        continue
      }

      seatedKeys.add(key)
      members.push((roster || []).find(bot => !bot?.ghost && botRosterKey(bot) === key) || resolved)
    }

    return members
  }

  const local = (roster || []).filter(bot => botGroups(botRosterMeta(bot, metaByName)).includes(group))
  const stored = storedMembers
  const seated = new Set(local.map(botRosterKey))
  const remote: RosterRow[] = []

  for (const descriptor of stored) {
    // Legacy descriptors can carry a FRIENDLY name as `name` (older builds
    // persisted display names — #92794: `name: '大司命'` for slug `taiyi`).
    // Key-matching alone then seats the descriptor as a ghost NEXT TO its own
    // live row ("4 bots" in a 2-bot room), and anything that passes ghost
    // identity onward targets a profile that does not exist on disk.
    // Normalize first: a same-connection roster row whose friendly names
    // include the descriptor's name IS this member. The next persistence
    // pass (durableGroupChatMembers writes from the seated roster) rewrites
    // the stored descriptor to the slug, so the repair is self-healing.
    const resolved = resolveLegacyMemberDescriptor(descriptor, roster)
    const key = memberSeatKey(resolved)

    if (!key || seated.has(key)) {
      continue
    }

    seated.add(key)
    // A selected-but-offline ghost intentionally carries only enough identity
    // to paint the roster. Never let it replace the room's durable descriptor,
    // which owns the full handle/title used by mentions and remote sync.
    remote.push((roster || []).find(bot => !bot?.ghost && botRosterKey(bot) === key) || resolved)
  }

  return [...local, ...remote]
}

/** A stored member descriptor, resolved against the live roster when its
 *  `name` is not a real slug (#92794). Exact key matches pass through
 *  untouched; only a descriptor whose key matches NO roster row is re-tried
 *  by friendly name against rows on the same connection. Unresolvable
 *  descriptors return as-is — they stay visible-but-degraded ghosts and must
 *  never be used as a `profile:` target. */
function resolveLegacyMemberDescriptor(descriptor: RosterRow, roster: RosterRow[]): RosterRow {
  const rows = roster || []

  if (rows.some(bot => botRosterKey(bot) === botRosterKey(descriptor))) {
    return descriptor
  }

  const wanted = String(descriptor?.name || '')
    .trim()
    .toLowerCase()

  if (!wanted) {
    return descriptor
  }

  // Connection scope: a descriptor WITH a connectionId only matches rows on
  // that connection (two `default`s on different machines must never merge).
  // A descriptor WITHOUT one predates connection scoping entirely — those
  // rooms only ever contained this machine's bots, so local (non-remote)
  // rows are the legal candidate set.
  const descriptorConnection = String(descriptor?.connectionId || '')

  const candidates = rows.filter(bot => {
    if (bot?.ghost) {
      return false
    }

    return descriptorConnection ? String(bot?.connectionId || '') === descriptorConnection : !bot?.remoteSource
  })

  const match = candidates.find(
    bot =>
      // Case-drifted slug ('Testbot' persisted for profile 'testbot') …
      String(bot?.name || '')
        .trim()
        .toLowerCase() === wanted ||
      // … or a friendly name persisted as `name` ('大司命' for 'taiyi').
      botFriendlyNames(bot).some(
        name =>
          String(name || '')
            .trim()
            .toLowerCase() === wanted
      )
  )

  return match || descriptor
}

/** Persist source-qualified identities for every selected member. The active
 *  source's row may become remote after a connection switch, so retaining it
 *  here is what keeps the same room intact across machines. */
export function durableGroupChatMembers(bots: RosterRow[]): GroupMember[] {
  return (bots || []).filter(bot => memberSeatKey(bot)).map(bot => {
    // Persistence pass over the whole seated roster: an orphaned member
    // (connection deleted) keeps its identity and simply persists with no
    // route — the same degraded shape the hydrate annotate produces. The
    // strict throw here would lose the entire room update over one row.
    const route = resolveBotConnectionRoute(bot).route

    // Keep the friendly identity on the stored descriptor: after a
    // connection switch the live roster row may be gone, and renamed-tag
    // mentions must still resolve against the persisted member.
    const title = String(
      botRosterMeta(bot, $botMeta.get())?.title || bot.ui_meta?.['hermes-bots']?.title || bot.title || ''
    ).trim()

    return {
      name: bot.name,
      handle: bot.handle || bot.name,
      ...(title
        ? {
            title
          }
        : {}),
      ...(bot.display_name
        ? {
            display_name: bot.display_name
          }
        : {}),
      connectionId: bot.connectionId,
      connectionKind: bot.connectionKind,
      connectionLabel: bot.connectionLabel,
      ...(route
        ? {
            route,
            targetProfile: route.targetProfile
          }
        : {}),
      // A swept/annotated member keeps its degraded mark across the rebuild —
      // otherwise the next room send would silently un-mark an orphaned row.
      ...(bot.sourceMissing
        ? {
            sourceMissing: true,
            sourceReachable: false
          }
        : {}),
      remoteSource: true,
      sourceScoped: Boolean(route)
    }
  })
}

/** Replace an existing room's COMPLETE member set. The durable room roster is
 *  authoritative; local profile metadata is updated as a compatibility and
 *  discovery projection, including removal from profiles no longer seated.
 *
 *  Metadata is keyed by botMetaKey (a source route key for scoped rows, the
 *  bare name otherwise), so profiles are matched on that — not on `name`,
 *  which collides across connections. */
export async function replaceGroupChatMembers(group: string, bots: RosterRow[]): Promise<RosterRow[]> {
  const unique: RosterRow[] = []
  const seen = new Set<string>()

  for (const bot of bots || []) {
    const key = memberSeatKey(bot)

    if (!key || seen.has(key)) {
      continue
    }

    seen.add(key)
    unique.push(bot)
  }

  if (!unique.length || unique.length > GROUP_CHAT_MAX_MEMBERS) {
    throw new Error(`Group chats require 1–${GROUP_CHAT_MAX_MEMBERS} bots`)
  }

  const metaSnapshot = $botMeta.get()
  const nextLocal = new Map<string, RosterRow>()

  for (const bot of unique) {
    if (bot.remoteSource) {
      continue
    }

    const key = botMetaKey(bot)

    if (key) {
      nextLocal.set(key, bot)
    }
  }

  // Every profile whose metadata must change: the new local members, plus any
  // profile still claiming this group so a removal actually un-seats it.
  const owners = new Map<string, RosterRow | string>(nextLocal)

  for (const [key, meta] of Object.entries(metaSnapshot || {})) {
    if (owners.has(key) || !botGroups(meta).includes(group)) {
      continue
    }

    // Prefer the live row: saveBotMeta routes a source-scoped write through
    // its connection, which a bare key string cannot describe.
    owners.set(key, ($lastRoster.get() || []).find(candidate => botMetaKey(candidate) === key) || key)
  }

  const entries = [...owners.entries()]

  const writes = await Promise.allSettled(
    entries.map(([key, owner]) => saveBotMeta(owner, groupMembershipPatch(metaSnapshot[key], group, nextLocal.has(key))))
  )

  const labelFor = (index: number): string => {
    const [key, owner] = entries[index]

    return typeof owner === 'string' ? key : displayName(owner, metaSnapshot[key])
  }

  const failed = writes
    .map((result, index) => ({ index, result }))
    .filter(({ result }) => result.status === 'rejected' || result.value?.serverOutcome === 'failed')

  if (failed.length) {
    // Only profiles whose new metadata was CONFIRMED remotely need a remote
    // rollback. A profile that explicitly rejected the write is already at the
    // old remote value; "restoring" it would make a failed rollback look like
    // a remote inconsistency this operation did not create.
    const rollbackIndexes = writes
      .map((result, index) => ({ index, result }))
      .filter(({ result }) => result.status === 'fulfilled' && result.value?.serverOutcome === 'persisted')
      .map(({ index }) => index)

    const rollbackResults = await Promise.allSettled(
      rollbackIndexes.map(index => {
        const [key, owner] = entries[index]
        const prior = metaSnapshot[key] || {}

        return saveBotMeta(owner, {
          group: prior.group || null,
          groups: Array.isArray(prior.groups) ? prior.groups : []
        })
      })
    )

    const rollbackFailures = rollbackResults
      .map((result, position) => ({ index: rollbackIndexes[position], result }))
      .filter(({ result }) => result.status === 'rejected' || result.value?.serverOutcome !== 'persisted')

    // saveBotMeta MERGES for normal edits. Restore the exact snapshot locally
    // as the final reconciliation step, so profiles that had no explicit group
    // fields are not left carrying synthetic empty projections.
    $botMeta.set(metaSnapshot)
    void persistBotMetaSnapshot(
      metaSnapshot,
      unique.some(bot => bot.sourceScoped)
    )

    const details = failed.map(({ index, result }) =>
      result.status === 'rejected'
        ? `${labelFor(index)}: ${result.reason?.message || 'write failed'}`
        : `${labelFor(index)}: server rejected the write`
    )

    if (rollbackFailures.length) {
      const rollbackDetails = rollbackFailures.map(({ index, result }) =>
        result.status === 'rejected'
          ? `${labelFor(index)}: ${result.reason?.message || 'rollback failed'}`
          : `${labelFor(index)}: rollback was not confirmed by the server`
      )

      throw new Error(
        `Could not save group members (${details.join(', ')}); rollback could not be confirmed (${rollbackDetails.join(', ')}), remote metadata may be inconsistent`
      )
    }

    throw new Error(`Could not save group members (${details.join(', ')}); changes were rolled back`)
  }

  updateGroupChat(group, room => {
    room.members = durableGroupChatMembers(unique)

    return room
  })

  return unique
}

/** Existing group names, alphabetical — feeds the Manage-groups dialog. */
export function knownGroups(metaByName: Record<string, BotMeta>) {
  const names = new Set<string>()

  for (const meta of Object.values(metaByName || {})) {
    for (const group of botGroups(meta)) {
      names.add(group)
    }
  }

  return [...names].sort((a, b) =>
    a.localeCompare(b, undefined, {
      sensitivity: 'base'
    })
  )
}
