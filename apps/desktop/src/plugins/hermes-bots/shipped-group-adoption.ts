import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'
import type { PluginProfileRouteLease, PluginStorage } from '@hermes/plugin-sdk'
import { sha256 } from '@noble/hashes/sha2.js'
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js'

import {
  $canonicalGroupBindings,
  bindAdoptedCanonicalGroup,
  quarantineCanonicalGroupBindings,
  revokeAdoptedCanonicalGroupsForConnection,
  revokeCanonicalGroupBinding,
  revokeStaleAdoptedCanonicalGroups
} from './canonical-group-registry'
import type {
  CanonicalGroupRoute,
  CanonicalRoom,
  ShippedGroupHeldWork,
  ShippedGroupHistoryEntry,
  ShippedGroupImportMember,
  ShippedGroupImportRequest,
  ShippedGroupImportResult
} from './canonical-groups'
import { $groupChats, groupChatHostedGateway, persistGroupChatRoomsRequired } from './group-chat'
import { classifyHostedRoomCapability } from './hosted-room-client'
import type {
  Attachment,
  GroupChat,
  GroupMember,
  GroupMessage,
  ShippedGroupAdoption,
  ShippedGroupAdoptionIssueKind,
  ShippedGroupAdoptionRoute
} from './types'

const SOURCE_PREFIX = 'hermes.plugin.hermes-bots.group-chats:'
const LEGACY_THREAD = /^legacy-\d+$/
const DATA_URL = /^data:([^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/

interface ProfileRouteCandidate {
  connectionId: string
  mode: 'local' | 'remote'
  profile: string
  targetProfile: string
}

export interface ShippedGroupOwnerChoice extends CanonicalGroupRoute {
  label: string
}

interface BuiltShippedGroupImport {
  request: ShippedGroupImportRequest
  requestHash: string
}

interface ForegroundFence {
  activation: number
  connectionId: string
  gateway: string
  profile: string
}

interface CapabilityResult {
  authorityGatewayId: string
  methods: string[]
}

let lifecycleGeneration = 0
let currentRun: null | { generation: number; restoreOnly: boolean; promise: Promise<void> } = null
let recoveryDispose: (() => void) | null = null
let recoveryRun: Promise<void> | null = null

/** Reuse the existing adopter after the predecessor settles; never revive an
 * old binding closure or turn a capability-only button into an importer. */
export function restoreShippedGroupBindings(storage: PluginStorage): Promise<void> {
  if (!recoveryDispose) { return Promise.resolve() }

  if (recoveryRun) { return recoveryRun }
  const generation = lifecycleGeneration
  const predecessor = currentRun?.promise.catch(() => undefined) ?? Promise.resolve()

  const run = predecessor.then(async () => {
    if (generation !== lifecycleGeneration) { return }

    const missing = Object.entries($groupChats.get()).some(([name, room]) =>
      !room.tombstone && room.shippedAdoption?.state === 'adopted' &&
      $canonicalGroupBindings.get()[name]?.isCurrent?.() !== true)

    if (missing) { await adoptShippedGroupChats(storage, true) }
  }).finally(() => { if (recoveryRun === run) { recoveryRun = null } })

  recoveryRun = run

  return run
}

function observeAdoptedRoutes(storage: PluginStorage): void {
  if (recoveryDispose) { return }
  // Socket retention is not execution authority. It keeps the existing bounded
  // reconnect loop alive after a dead binding's counted request lease is freed.
  const retained = new Map<string, () => void>()

  const reconcile = () => {
    const wanted = new Set<string>()

    for (const room of Object.values($groupChats.get())) {
      const adoption = room.shippedAdoption

      if (room.tombstone || adoption?.state !== 'adopted' || !adoption.route ||
          adoption.issue?.kind === 'owner-replaced') { continue }

      const { connectionId, profile } = adoption.route
      const key = JSON.stringify([connectionId, profile])
      wanted.add(key)

      if (!retained.has(key)) {
        retained.set(key, host.retainProfileSocket?.({ connectionId, mode: 'remote', profile, targetProfile: profile }) ?? (() => undefined))
      }
    }

    for (const [key, release] of retained) {
      if (!wanted.has(key)) { retained.delete(key); release() }
    }
  }

  const unbind = $groupChats.listen(reconcile)

  const unwatch = host.onProfileRouteState?.(event => {
    if (!retained.has(JSON.stringify([event.connectionId, event.profile]))) { return }
    revokeStaleAdoptedCanonicalGroups()

    if (event.state === 'open') { void restoreShippedGroupBindings(storage).catch(() => undefined) }
  })

  recoveryDispose = () => {
    unwatch?.()
    unbind()

    for (const release of retained.values()) { release() }
    retained.clear()
  }

  reconcile()
}

// Group-room authority changes publish synchronously. Retire a stale route
// lease in that same writer turn, before a mounted continuation can dispatch.
$groupChats.listen(revokeStaleAdoptedCanonicalGroups)

function digest(value: string): string {
  return bytesToHex(sha256(utf8ToBytes(value)))
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function atMilliseconds(value: unknown): number {
  const number = Number(value)

  if (!Number.isFinite(number) || number < 0) {
    return 0
  }

  return Math.floor(number < 1_000_000_000_000 && number >= 1_000_000_000 ? number * 1000 : number)
}

function memberProfile(member: GroupMember): string {
  return (
    text(member.route?.targetProfile) ||
    text(member.targetProfile) ||
    text(member.hostedIdentity?.profile) ||
    text(member.name)
  )
}

function memberDisplayName(member: GroupMember): string {
  return text(member.display_name) || text(member.title) || text(member.name) || memberProfile(member)
}

function memberConnectionId(member: GroupMember): string {
  return text(member.route?.connectionId) || text(member.connectionId)
}

function memberIsRemote(member: GroupMember): boolean {
  return member.remoteSource === true || member.sourceScoped === true
}

function sourceMemberId(sourceId: string, index: number, member: GroupMember): string {
  const identity = JSON.stringify([
    sourceId,
    index,
    text(member.name),
    memberProfile(member),
    text(member.handle),
    memberConnectionId(member),
    text(member.connectionLabel),
    memberIsRemote(member)
  ])

  return `member:${index}:${digest(identity).slice(0, 32)}`
}

function uniqueHandle(handle: string, sourceId: string, used: Set<string>): string {
  const base = handle || `member-${digest(sourceId).slice(0, 12)}`
  let candidate = base

  if (used.has(candidate.toLowerCase())) {
    const suffix = `-${digest(sourceId).slice(0, 8)}`
    candidate = base.slice(0, Math.max(1, 128 - suffix.length)) + suffix
  }

  if (used.has(candidate.toLowerCase())) {
    throw new Error('This Group Chat has member handles that cannot be distinguished safely.')
  }

  used.add(candidate.toLowerCase())

  return candidate
}

function historicalMember(
  sourceId: string,
  author: GroupMessage['from'],
  former: Map<string, ShippedGroupImportMember>,
  usedHandles: Set<string>
): ShippedGroupImportMember {
  const signature = JSON.stringify([
    sourceId,
    text(author?.name),
    text(author?.source),
    author?.hostedIdentity?.memberId || ''
  ])

  const existing = former.get(signature)

  if (existing) {
    return existing
  }

  const key = digest(signature)

  const member: ShippedGroupImportMember = {
    source_member_id: `former:${key.slice(0, 32)}`,
    name: text(author?.name) || 'Former member',
    profile: `former-${key.slice(0, 24)}`,
    handle: uniqueHandle(`former-${key.slice(0, 16)}`, `former:${key}`, usedHandles),
    remote_source: false,
    active: false,
    ...(text(author?.source) ? { connection_label: text(author.source) } : {})
  }

  former.set(signature, member)

  return member
}

function matchingMember(
  author: GroupMessage['from'],
  sourceMembers: Array<{ member: GroupMember; imported: ShippedGroupImportMember }>
): ShippedGroupImportMember | null {
  const hostedMemberId = text(author?.hostedIdentity?.memberId)

  if (hostedMemberId) {
    const exact = sourceMembers.filter(({ member }) => text(member.hostedIdentity?.memberId) === hostedMemberId)

    return exact.length === 1 ? exact[0].imported : null
  }

  const authorName = text(author?.name)
  const authorSource = text(author?.source)

  let candidates = sourceMembers.filter(({ member, imported }) => {
    const aliases = new Set(
      [text(member.name), text(member.display_name), text(member.title), text(member.handle), imported.profile].filter(
        Boolean
      )
    )

    return aliases.has(authorName)
  })

  if (authorSource) {
    candidates = candidates.filter(({ member, imported }) =>
      [text(member.connectionLabel), memberConnectionId(member), imported.connection_label, imported.connection_id]
        .filter(Boolean)
        .includes(authorSource)
    )
  }

  return candidates.length === 1 ? candidates[0].imported : null
}

function attachmentKind(attachment: Attachment, mime: string): 'file' | 'image' | 'pdf' {
  if (attachment.kind === 'file' || attachment.kind === 'image' || attachment.kind === 'pdf') {
    return attachment.kind
  }

  if (mime.startsWith('image/')) {
    return 'image'
  }

  return mime === 'application/pdf' ? 'pdf' : 'file'
}

function importAttachments(value: unknown): ShippedGroupHistoryEntry['attachments'] {
  if (value === undefined) {
    return undefined
  }

  if (!Array.isArray(value)) {
    throw new Error('This Group Chat has historical attachments that cannot be read safely.')
  }

  const attachments = value.map((raw, index) => {
    const attachment = raw as Attachment
    const data = typeof attachment?.data === 'string' ? attachment.data : ''
    const match = DATA_URL.exec(data)
    const name = text(attachment?.name)

    if (!match || !name) {
      throw new Error(`Historical attachment ${index + 1} is incomplete. The original Group Chat was kept.`)
    }

    return {
      kind: attachmentKind(attachment, match[1].toLowerCase()),
      name,
      data
    }
  })

  return attachments.length ? attachments : undefined
}

function heldWork(
  sourceId: string,
  room: GroupChat,
  sourceMembers: Array<{ member: GroupMember; imported: ShippedGroupImportMember }>
): ShippedGroupHeldWork[] {
  const latestAt = Math.max(0, ...(room.log || []).map(entry => atMilliseconds(entry?.at)))
  const held: ShippedGroupHeldWork[] = []

  for (const [key, value] of Object.entries(room.stranded || {})) {
    const matches = sourceMembers.filter(({ member, imported }) => {
      const aliases = [text(member.name), text(member.handle), imported.profile, imported.source_member_id].filter(
        Boolean
      )

      return aliases.some(alias => key === alias || key.endsWith(`::${alias}`))
    })

    const related = matches.length === 1 ? matches[0].imported : null
    const label = related?.name || key || 'a member'
    const workIdentity = JSON.stringify([sourceId, key, value])

    held.push({
      source_work_id: `work:${digest(workIdentity).slice(0, 48)}`,
      at_ms: latestAt,
      state: 'uncertain',
      description: `Work for ${label} had no confirmed outcome when this Group Chat was upgraded. Review it before retrying.`,
      ...(related ? { member_source_id: related.source_member_id } : {})
    })
  }

  return held
}

/** Map exactly one released plugin-storage room. The original array order is
 * the event order; no mapped row enters a send or execution method. */
export async function buildShippedGroupImport(
  group: string,
  room: GroupChat,
  ownerConnectionId: string
): Promise<BuiltShippedGroupImport> {
  const storedIdentity = text(room.roomId) || text(room.desktopAuthorityHash)

  if (!storedIdentity) {
    throw new Error('This Group Chat does not yet have a durable local identity. Its original data was kept.')
  }

  const sourceId = `${SOURCE_PREFIX}${storedIdentity}`
  const roomId = text(room.roomId) || `r${digest(sourceId).slice(0, 32)}`
  const usedHandles = new Set(['all', 'everyone'])
  const sourceMembers: Array<{ member: GroupMember; imported: ShippedGroupImportMember }> = []

  for (const [index, raw] of (Array.isArray(room.members) ? room.members : []).entries()) {
    const member = raw as GroupMember
    const profile = memberProfile(member)
    const name = memberDisplayName(member)

    if (!profile || !name) {
      throw new Error(`Member ${index + 1} has no stable profile identity. The original Group Chat was kept.`)
    }

    const sourceIdForMember = sourceMemberId(sourceId, index, member)
    const connectionId = memberConnectionId(member)

    // A missing source is not the selected owner. Keep the entire original room
    // until that member's route can be resolved, rather than making a same-named
    // local profile executable or inventing an importer ownership classification.
    if (member.sourceMissing || (!connectionId && memberIsRemote(member))) {
      throw new Error(`Member ${index + 1} has an unresolved source owner. The original Group Chat was kept.`)
    }

    const imported: ShippedGroupImportMember = {
      source_member_id: sourceIdForMember,
      name,
      profile,
      handle: uniqueHandle(text(member.handle) || profile, sourceIdForMember, usedHandles),
      // Released persistence intentionally labels every member remote-capable.
      // Backend import needs a different fact: remote relative to the selected
      // canonical owner connection.
      remote_source: Boolean(connectionId && connectionId !== ownerConnectionId),
      active: true,
      ...(connectionId ? { connection_id: connectionId } : {}),
      ...(text(member.connectionLabel) ? { connection_label: text(member.connectionLabel) } : {})
    }

    sourceMembers.push({ member, imported })
  }

  const former = new Map<string, ShippedGroupImportMember>()
  const history: ShippedGroupHistoryEntry[] = []

  for (const [index, entry] of (Array.isArray(room.log) ? room.log : []).entries()) {
    if (entry?.from?.kind !== 'member' && entry?.from?.kind !== 'user') {
      throw new Error(`History entry ${index + 1} has no safe author identity. The original Group Chat was kept.`)
    }

    if (typeof entry.text !== 'string') {
      throw new Error(`History entry ${index + 1} has unreadable text. The original Group Chat was kept.`)
    }

    const authorKind = entry.from.kind
    const authorName = text(entry.from.name) || (authorKind === 'user' ? 'You' : 'Former member')
    let member: ShippedGroupImportMember | null = null

    if (authorKind === 'member') {
      member = matchingMember(entry.from, sourceMembers)

      if (!member) {
        member = historicalMember(sourceId, entry.from, former, usedHandles)
      }
    }

    const thread = text(entry?.thread)
    const attachments = importAttachments(entry?.images)

    const identity = JSON.stringify([
      sourceId,
      index,
      atMilliseconds(entry?.at),
      authorKind,
      authorName,
      member?.source_member_id || '',
      text(entry?.text),
      thread && !LEGACY_THREAD.test(thread) ? thread : 'legacy',
      attachments || []
    ])

    history.push({
      source_entry_id: `entry:${index}:${digest(identity).slice(0, 40)}`,
      at_ms: atMilliseconds(entry?.at),
      author_kind: authorKind,
      author_name: authorName,
      ...(member ? { member_source_id: member.source_member_id } : {}),
      text: String(entry?.text || ''),
      thread_id: thread && !LEGACY_THREAD.test(thread) ? thread : 'legacy',
      ...(attachments ? { attachments } : {})
    })
  }

  const members = [...sourceMembers.map(item => item.imported), ...former.values()]

  if (!members.length) {
    throw new Error('This Group Chat has no retained member identities to adopt. Its original history was kept.')
  }

  const request: ShippedGroupImportRequest = {
    room_id: roomId,
    name: text(group),
    source_id: sourceId,
    members,
    history,
    held_work: heldWork(sourceId, room, sourceMembers)
  }

  return {
    request,
    requestHash: digest(JSON.stringify(request))
  }
}

function sourceCheckpoint(group: string, room: GroupChat): ShippedGroupAdoption | null {
  const storedIdentity = text(room.roomId) || text(room.desktopAuthorityHash)

  if (!storedIdentity) {
    return null
  }

  const sourceId = `${SOURCE_PREFIX}${storedIdentity}`

  return {
    version: 1,
    state: 'waiting',
    sourceId,
    roomId: text(room.roomId) || `r${digest(sourceId).slice(0, 32)}`,
    requestHash: digest(JSON.stringify([group, room.members || [], room.log || [], room.stranded || {}]))
  }
}

function foregroundFence(): ForegroundFence {
  return {
    activation: gatewayActivationEpoch(),
    connectionId: text(host.state.connectionId?.get?.()),
    gateway: text(host.state.gateway?.get?.()),
    profile: text(host.state.profile?.get?.())
  }
}

function foregroundCurrent(fence: ForegroundFence, generation: number): boolean {
  return (
    generation === lifecycleGeneration &&
    fence.activation === gatewayActivationEpoch() &&
    fence.connectionId === text(host.state.connectionId?.get?.()) &&
    fence.gateway === text(host.state.gateway?.get?.()) &&
    fence.profile === text(host.state.profile?.get?.())
  )
}

function checkpointMatches(room: GroupChat | undefined, expected: ShippedGroupAdoption): boolean {
  const current = room?.shippedAdoption

  return Boolean(
    current &&
    current.version === 1 &&
    current.state === expected.state &&
    current.sourceId === expected.sourceId &&
    current.roomId === expected.roomId &&
    current.requestHash === expected.requestHash &&
    current.route?.connectionId === expected.route?.connectionId &&
    current.route?.profile === expected.route?.profile &&
    current.route?.authorityGatewayId === expected.route?.authorityGatewayId
  )
}

async function persistRoom(
  storage: PluginStorage,
  group: string,
  expected: GroupChat,
  mutate: (room: GroupChat) => GroupChat
): Promise<boolean> {
  const before = $groupChats.get()

  if (before[group] !== expected) {
    return false
  }

  const next = { ...before, [group]: mutate(expected) }

  // Revoke old closures before publishing ownership or awaiting durable storage.
  // A failed save must not resurrect an already-mounted descriptor alias.
  if (next[group].shippedAdoption) {
    quarantineCanonicalGroupBindings(group, next[group].shippedAdoption)
  }

  $groupChats.set(next)

  try {
    await persistGroupChatRoomsRequired(next, storage)

    return $groupChats.get() === next
  } catch (error) {
    if ($groupChats.get() === next) {
      $groupChats.set(before)
    }

    throw error
  }
}

async function persistCheckpoint(
  storage: PluginStorage,
  group: string,
  expectedRoom: GroupChat,
  adoption: ShippedGroupAdoption,
  roomPatch: Partial<GroupChat> = {}
): Promise<boolean> {
  return persistRoom(storage, group, expectedRoom, room => ({
    ...room,
    ...roomPatch,
    shippedAdoption: adoption
  }))
}

async function persistIssue(
  storage: PluginStorage,
  group: string,
  adoption: ShippedGroupAdoption,
  kind: ShippedGroupAdoptionIssueKind,
  message: string
): Promise<void> {
  revokeCanonicalGroupBinding(group)
  const current = $groupChats.get()[group]

  if (!checkpointMatches(current, adoption)) {
    return
  }

  await persistCheckpoint(storage, group, current, {
    ...adoption,
    issue: { kind, message }
  })
}

function routeCandidates(value: unknown): ProfileRouteCandidate[] {
  if (!Array.isArray(value)) {
    return []
  }

  const unique = new Map<string, ProfileRouteCandidate>()

  for (const raw of value) {
    const candidate = raw as Partial<ProfileRouteCandidate>
    const connectionId = text(candidate.connectionId)
    const profile = text(candidate.profile)
    const targetProfile = text(candidate.targetProfile) || profile
    const mode = candidate.mode === 'local' ? 'local' : candidate.mode === 'remote' ? 'remote' : null

    // The shipped storage projection is owned by the exact default profile.
    // Aliases into another backend profile are not authority to guess that owner.
    if (!connectionId || profile !== 'default' || targetProfile !== 'default' || !mode) {
      continue
    }

    unique.set(`${connectionId}\0${profile}`, { connectionId, profile, targetProfile, mode })
  }

  return [...unique.values()]
}

async function chooseOwner(room: GroupChat): Promise<CanonicalGroupRoute | null> {
  const candidates = routeCandidates(await host.profileRoutes())
  const historicalConnections = new Set((room.members || []).map(memberConnectionId).filter(Boolean))
  const evidenced = candidates.filter(candidate => historicalConnections.has(candidate.connectionId))
  const evidencedLocal = evidenced.filter(candidate => candidate.mode === 'local')
  const local = candidates.filter(candidate => candidate.mode === 'local')

  const exact =
    evidencedLocal.length === 1
      ? evidencedLocal
      : evidenced.length === 1
        ? evidenced
        : historicalConnections.size === 0 && local.length === 1
          ? local
          : historicalConnections.size === 0 && local.length === 0 && candidates.length === 1
            ? candidates
            : []

  return exact.length === 1 ? { connectionId: exact[0].connectionId, profile: exact[0].profile } : null
}

export async function shippedGroupOwnerChoices(room: GroupChat): Promise<ShippedGroupOwnerChoice[]> {
  const candidates = routeCandidates(await host.profileRoutes()).sort((left, right) =>
    left.connectionId.localeCompare(right.connectionId)
  )

  const storedLabels = new Map<string, string>()

  for (const member of room.members || []) {
    const connectionId = memberConnectionId(member)
    const label = text(member.connectionLabel)

    if (connectionId && label && !storedLabels.has(connectionId)) {
      storedLabels.set(connectionId, label)
    }
  }

  try {
    const registry = await window.hermesDesktop?.connections?.list?.()

    for (const connection of registry?.connections || []) {
      if (connection?.id && connection?.label && !storedLabels.has(connection.id)) {
        storedLabels.set(connection.id, connection.label)
      }
    }
  } catch {
    // Labels are presentation only. Deterministic ordinals keep the choice
    // usable when the registry label read is unavailable.
  }

  return candidates.map((candidate, index) => ({
    connectionId: candidate.connectionId,
    profile: candidate.profile,
    label: storedLabels.get(candidate.connectionId) || `Device ${index + 1}`
  }))
}

/** Pin an explicit owner only to the exact waiting source checkpoint the user
 * saw. The selected route still runs through capability/installation proof. */
export async function selectShippedGroupOwner(
  storage: PluginStorage,
  group: string,
  expected: Pick<ShippedGroupAdoption, 'requestHash' | 'sourceId'>,
  selected: CanonicalGroupRoute
): Promise<void> {
  const generation = lifecycleGeneration
  const room = $groupChats.get()[group]
  let adoption = room?.shippedAdoption

  if (
    !room ||
    adoption?.state !== 'waiting' ||
    adoption.route ||
    adoption.issue?.kind !== 'owner-ambiguous' ||
    adoption.sourceId !== expected.sourceId ||
    adoption.requestHash !== expected.requestHash
  ) {
    throw new Error('This Group Chat owner choice is no longer current.')
  }

  const choices = await shippedGroupOwnerChoices(room)

  const owner = choices.find(
    choice => choice.connectionId === selected.connectionId && choice.profile === selected.profile
  )

  if (!owner || generation !== lifecycleGeneration || $groupChats.get()[group] !== room) {
    throw new Error('This Group Chat owner choice is no longer available.')
  }

  let routeOwner: PluginProfileRouteLease | null = null

  try {
    routeOwner = await acquireOwnerRoute(owner)
    const capability = await readCapability(routeOwner)
    routeOwner.assertCurrent()

    if (generation !== lifecycleGeneration || $groupChats.get()[group] !== room) {
      return
    }

    if (!capability.methods.includes('groups.import_history')) {
      await persistIssue(
        storage,
        group,
        adoption,
        'update-required',
        'Update the gateway that owns this Group Chat, then reconnect. Its history and members are still here.'
      )

      return
    }

    const built = await buildShippedGroupImport(group, room, owner.connectionId)

    const prepared: ShippedGroupAdoption = {
      ...adoption,
      state: 'prepared',
      requestHash: built.requestHash,
      ownerSelection: 'explicit',
      route: {
        connectionId: owner.connectionId,
        profile: owner.profile,
        authorityGatewayId: capability.authorityGatewayId
      },
      issue: undefined
    }

    if (!(await persistCheckpoint(storage, group, room, prepared))) {
      return
    }

    routeOwner.assertCurrent()
    adoption = prepared
    const preparedRoute = prepared.route!

    const transferred = await importPreparedGroup(
      storage,
      group,
      prepared,
      built,
      owner,
      routeOwner,
      generation,
      () => {
        const checkpoint = $groupChats.get()[group]?.shippedAdoption

        return (
          generation === lifecycleGeneration &&
          checkpoint?.sourceId === prepared.sourceId &&
          checkpoint.roomId === prepared.roomId &&
          checkpoint.requestHash === prepared.requestHash &&
          checkpoint.route?.connectionId === preparedRoute.connectionId &&
          checkpoint.route?.profile === preparedRoute.profile &&
          checkpoint.route?.authorityGatewayId === preparedRoute.authorityGatewayId
        )
      }
    )

    if (transferred) {
      routeOwner = null
    }
  } catch (error) {
    if (generation !== lifecycleGeneration) {
      return
    }

    const issue = issueForError(error)
    await persistIssue(storage, group, adoption, issue.kind, issue.message)
  } finally {
    routeOwner?.release()
  }
}

async function pinnedRoutePresent(route: ShippedGroupAdoptionRoute): Promise<boolean> {
  const candidates = routeCandidates(await host.profileRoutes())

  return candidates.some(
    candidate => candidate.connectionId === route.connectionId && candidate.profile === route.profile
  )
}

async function acquireOwnerRoute(route: CanonicalGroupRoute): Promise<PluginProfileRouteLease> {
  return host.acquireProfileRoute({
    connectionId: route.connectionId,
    profile: route.profile,
    targetProfile: route.profile,
    mode: route.connectionId === 'local' ? 'local' : 'remote'
  })
}

async function readCapability(owner: PluginProfileRouteLease): Promise<CapabilityResult> {
  const value = await owner.request<Record<string, unknown>>('groups.capabilities', {
    profile: owner.route.targetProfile
  })

  const methods =
    Array.isArray(value?.methods) && value.methods.every(method => typeof method === 'string')
      ? (value.methods as string[])
      : []

  const authorityGatewayId = text(value?.authority_gateway_id)

  if (!authorityGatewayId) {
    throw Object.assign(new Error('The owning gateway returned an incomplete Group Chat capability response.'), {
      adoptionIssue: 'update-required' as const
    })
  }

  return { authorityGatewayId, methods }
}

function issueForError(error: unknown): { kind: ShippedGroupAdoptionIssueKind; message: string } {
  const message = error instanceof Error ? error.message : String(error)

  const tagged =
    error && typeof error === 'object' ? (error as { adoptionIssue?: ShippedGroupAdoptionIssueKind }) : null

  if (tagged?.adoptionIssue) {
    return {
      kind: tagged.adoptionIssue,
      message
    }
  }

  if (/could not be saved|storage unavailable|available storage/i.test(message)) {
    return {
      kind: 'storage',
      message:
        'Hermes could not save the Group Chat upgrade checkpoint. Free some storage and restart; the original history was kept.'
    }
  }

  if (/This host has too many active Group Chats/i.test(message)) {
    return {
      kind: 'conflict',
      message:
        'The original gateway has reached its Group Chat limit. Remove an unused Group Chat there, then restart Hermes. The original history was kept.'
    }
  }

  if (
    /invalid (?:member|history|historical|source|room)|member .* must|historical attachment|must be a bounded|exceed(?:s|ed)|too many room members/i.test(
      message
    )
  ) {
    return {
      kind: 'conflict',
      message:
        'This Group Chat contains saved data the new gateway cannot adopt safely. Its original history and members were kept for review.'
    }
  }

  if (/different import content|already exists|conflict|changed while/i.test(message)) {
    return {
      kind: 'conflict',
      message:
        'This Group Chat changed while it was being upgraded. Its original history is still here; restart Hermes before trying again.'
    }
  }

  if (/canonical_owner_required|permission_denied|session:control/i.test(message)) {
    return {
      kind: 'auth',
      message: 'Sign in again to the original Group Chat gateway to finish the upgrade. No history was removed.'
    }
  }

  const capability = classifyHostedRoomCapability({ ok: false, error })

  if (capability.kind === 'auth-failure') {
    return {
      kind: 'auth',
      message: 'Sign in again to the original Group Chat gateway to finish the upgrade. No history was removed.'
    }
  }

  if (capability.kind === 'unsupported') {
    return {
      kind: 'update-required',
      message: 'Update the gateway that owns this Group Chat, then reconnect. Its history and members are still here.'
    }
  }

  return {
    kind: 'offline',
    message: 'Reconnect the original Group Chat gateway to finish the upgrade. Its history and members are still here.'
  }
}

async function verifyOwner(
  storage: PluginStorage,
  group: string,
  adoption: ShippedGroupAdoption,
  owner: PluginProfileRouteLease,
  requireImport: boolean,
  current: () => boolean
): Promise<{ capability: CapabilityResult; route: CanonicalGroupRoute } | null> {
  const route = adoption.route

  if (!route) {
    return null
  }

  try {
    if (!(await pinnedRoutePresent(route))) {
      if (!current()) {
        return null
      }

      await persistIssue(
        storage,
        group,
        adoption,
        'offline',
        'Reconnect the original gateway for this Group Chat. Hermes will not move it to a different connection.'
      )

      return null
    }

    owner.assertCurrent()
    const canonicalRoute = { connectionId: route.connectionId, profile: route.profile }
    const capability = await readCapability(owner)
    owner.assertCurrent()

    if (!current()) {
      return null
    }

    if (capability.authorityGatewayId !== route.authorityGatewayId) {
      await persistIssue(
        storage,
        group,
        adoption,
        'owner-replaced',
        'This connection now reaches a different gateway installation. Reconnect the original gateway to keep this Group Chat on its owner.'
      )

      return null
    }

    if (requireImport && !capability.methods.includes('groups.import_history')) {
      await persistIssue(
        storage,
        group,
        adoption,
        'update-required',
        'Update the gateway that owns this Group Chat, then reconnect. Its history and members are still here.'
      )

      return null
    }

    return { capability, route: canonicalRoute }
  } catch (error) {
    if (!current()) {
      return null
    }

    const issue = issueForError(error)
    await persistIssue(storage, group, adoption, issue.kind, issue.message)

    return null
  }
}

async function restoreAdoptedGroup(
  storage: PluginStorage,
  group: string,
  adoption: ShippedGroupAdoption,
  generation: number,
  fence: ForegroundFence
): Promise<void> {
  const sourceCurrent = () => foregroundCurrent(fence, generation)
  const route = adoption.route

  if (!route) {
    return
  }

  let routeOwner: PluginProfileRouteLease | null = null

  try {
    routeOwner = await acquireOwnerRoute(route)
    const owner = await verifyOwner(storage, group, adoption, routeOwner, false, sourceCurrent)

    if (!owner || !sourceCurrent()) {
      return
    }

    const state = await routeOwner.request<{ room: CanonicalRoom }>('groups.state', {
      profile: routeOwner.route.targetProfile,
      room_id: adoption.roomId
    })

    routeOwner.assertCurrent()

    if (
      !sourceCurrent() ||
      state.room?.room_id !== adoption.roomId ||
      text(state.room?.authority_gateway_id) !== adoption.route?.authorityGatewayId ||
      !checkpointMatches($groupChats.get()[group], adoption)
    ) {
      return
    }

    if (adoption.issue) {
      const recovered: ShippedGroupAdoption = { ...adoption, issue: undefined }
      const current = $groupChats.get()[group]

      if (
        !(await persistCheckpoint(storage, group, current, recovered)) ||
        !checkpointMatches($groupChats.get()[group], recovered)
      ) {
        return
      }

      routeOwner.assertCurrent()
    }

    routeOwner.assertCurrent()
    bindAdoptedCanonicalGroup(
      group,
      owner.route,
      state.room,
      adoption,
      generation,
      () => generation === lifecycleGeneration && checkpointMatches($groupChats.get()[group], adoption),
      routeOwner
    )
    routeOwner = null
  } catch (error) {
    if (!sourceCurrent()) {
      return
    }

    const issue = issueForError(error)
    await persistIssue(storage, group, adoption, issue.kind, issue.message)
  } finally {
    routeOwner?.release()
  }
}

async function importPreparedGroup(
  storage: PluginStorage,
  group: string,
  adoption: ShippedGroupAdoption,
  built: BuiltShippedGroupImport,
  ownerRoute: CanonicalGroupRoute,
  routeOwner: PluginProfileRouteLease,
  generation: number,
  current: () => boolean
): Promise<boolean> {
  routeOwner.assertCurrent()

  const result = await routeOwner.request<ShippedGroupImportResult>('groups.import_history', {
    ...(built.request as unknown as Record<string, unknown>),
    profile: routeOwner.route.targetProfile
  })

  routeOwner.assertCurrent()

  if (!current() || !checkpointMatches($groupChats.get()[group], adoption)) {
    return false
  }

  if (
    result.source_id !== adoption.sourceId ||
    result.room?.room_id !== adoption.roomId ||
    text(result.room?.authority_gateway_id) !== adoption.route?.authorityGatewayId ||
    !Number.isSafeInteger(result.room?.authority_epoch) ||
    Number(result.room.authority_epoch) < 1
  ) {
    throw Object.assign(new Error('The gateway returned a different Group Chat identity.'), {
      adoptionIssue: 'conflict' as const
    })
  }

  const acknowledged: ShippedGroupAdoption = {
    ...adoption,
    state: 'adopted',
    issue: undefined,
    acknowledgedAt: Date.now(),
    importedHistory: result.imported_history,
    heldWork: result.held_work,
    heldMembers: result.held_members,
    retiredMembers: result.retired_members
  }

  const room = $groupChats.get()[group]

  const persisted = await persistCheckpoint(storage, group, room, acknowledged, {
    roomId: adoption.roomId,
    hosted: adoption.route!.authorityGatewayId,
    hostedConnectionId: adoption.route!.connectionId,
    hostedEpoch: Number(result.room.authority_epoch),
    hostedMembersVerified: false,
    continuityMode: 'gateway',
    continuityIssue: null,
    epoch: Number(room.epoch || 0) + 1,
    running: false
  })

  routeOwner.assertCurrent()

  if (!persisted || !current() || !checkpointMatches($groupChats.get()[group], acknowledged)) {
    return false
  }

  routeOwner.assertCurrent()
  bindAdoptedCanonicalGroup(
    group,
    ownerRoute,
    result.room,
    acknowledged,
    generation,
    () => generation === lifecycleGeneration && checkpointMatches($groupChats.get()[group], acknowledged),
    routeOwner
  )

  return true
}

async function processGroup(storage: PluginStorage, group: string, generation: number): Promise<void> {
  let room = $groupChats.get()[group]

  if (!room || room.tombstone || (groupChatHostedGateway(room) && room.shippedAdoption?.state !== 'adopted')) {
    return
  }

  const fence = foregroundFence()
  const existing = room.shippedAdoption

  if (existing) { quarantineCanonicalGroupBindings(group, existing) }

  if (existing?.state === 'adopted') {
    revokeCanonicalGroupBinding(group)
    await restoreAdoptedGroup(storage, group, existing, generation, fence)

    return
  }

  const source = sourceCheckpoint(group, room)
  let adoption = existing

  if (!source) {
    // Classic authority activation is the prerequisite source-identity writer.
    // Retain the untouched room instead of fabricating one here.
    return
  }

  if (
    adoption &&
    adoption.state === 'waiting' &&
    (adoption.sourceId !== source.sourceId ||
      adoption.roomId !== source.roomId ||
      adoption.requestHash !== source.requestHash)
  ) {
    await persistIssue(
      storage,
      group,
      adoption,
      'conflict',
      'This Group Chat changed after its upgrade was prepared. Its original history is still here; restart Hermes before trying again.'
    )

    return
  }

  if (!adoption) {
    adoption = source

    if (!(await persistCheckpoint(storage, group, room, adoption))) {
      return
    }

    room = $groupChats.get()[group]
  }

  if (!foregroundCurrent(fence, generation) || !checkpointMatches(room, adoption)) {
    return
  }

  let built: BuiltShippedGroupImport
  let routeOwner: PluginProfileRouteLease | null = null
  let ownerRoute: CanonicalGroupRoute | null = null

  try {
    if (!adoption.route) {
      let owner: CanonicalGroupRoute | null = null

      try {
        owner = await chooseOwner(room)
      } catch (error) {
        if (!foregroundCurrent(fence, generation)) {
          return
        }

        const issue = issueForError(error)
        await persistIssue(storage, group, adoption, issue.kind, issue.message)

        return
      }

      if (!owner) {
        await persistIssue(
          storage,
          group,
          adoption,
          'owner-ambiguous',
          'Hermes cannot safely tell which gateway originally owned this Group Chat. It kept the original history and will not choose another Bot or connection.'
        )

        return
      }

      ownerRoute = owner
      routeOwner = await acquireOwnerRoute(owner)
      const capability = await readCapability(routeOwner)
      routeOwner.assertCurrent()

      if (!capability.methods.includes('groups.import_history')) {
        await persistIssue(
          storage,
          group,
          adoption,
          'update-required',
          'Update the gateway that owns this Group Chat, then reconnect. Its history and members are still here.'
        )

        return
      }

      if (!foregroundCurrent(fence, generation)) {
        return
      }

      try {
        built = await buildShippedGroupImport(group, room, owner.connectionId)
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'This Group Chat could not be mapped safely. Its original data was kept.'

        await persistIssue(storage, group, adoption, 'conflict', message)

        return
      }

      const prepared: ShippedGroupAdoption = {
        ...adoption,
        state: 'prepared',
        requestHash: built.requestHash,
        ownerSelection: 'inferred',
        route: {
          connectionId: owner.connectionId,
          profile: owner.profile,
          authorityGatewayId: capability.authorityGatewayId
        },
        issue: undefined
      }

      const current = $groupChats.get()[group]

      if (!checkpointMatches(current, adoption) || !(await persistCheckpoint(storage, group, current, prepared))) {
        return
      }

      routeOwner.assertCurrent()

      adoption = prepared
      room = $groupChats.get()[group]
    } else {
      ownerRoute = { connectionId: adoption.route.connectionId, profile: adoption.route.profile }

      try {
        built = await buildShippedGroupImport(group, room, adoption.route.connectionId)
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : 'This Group Chat could not be mapped safely. Its original data was kept.'

        await persistIssue(storage, group, adoption, 'conflict', message)

        return
      }

      if (
        adoption.sourceId !== built.request.source_id ||
        adoption.roomId !== built.request.room_id ||
        adoption.requestHash !== built.requestHash
      ) {
        await persistIssue(
          storage,
          group,
          adoption,
          'conflict',
          'This Group Chat changed after its upgrade was prepared. Its original history is still here; restart Hermes before trying again.'
        )

        return
      }

      routeOwner = await acquireOwnerRoute(ownerRoute)

      const owner = await verifyOwner(storage, group, adoption, routeOwner, true, () =>
        foregroundCurrent(fence, generation)
      )

      if (!owner) {
        return
      }
    }

    if (
      !routeOwner ||
      !ownerRoute ||
      !foregroundCurrent(fence, generation) ||
      !checkpointMatches($groupChats.get()[group], adoption)
    ) {
      return
    }

    const transferred = await importPreparedGroup(
      storage,
      group,
      adoption,
      built,
      ownerRoute,
      routeOwner,
      generation,
      () => foregroundCurrent(fence, generation)
    )

    if (transferred) {
      routeOwner = null
    }
  } catch (error) {
    if (!foregroundCurrent(fence, generation)) {
      return
    }

    const issue = issueForError(error)
    await persistIssue(storage, group, adoption, issue.kind, issue.message)
  } finally {
    routeOwner?.release()
  }
}

async function runAdoption(storage: PluginStorage, generation: number, restoreOnly = false): Promise<void> {
  for (const group of Object.keys($groupChats.get())) {
    if (generation !== lifecycleGeneration) {
      return
    }

    if (restoreOnly && $groupChats.get()[group]?.shippedAdoption?.state !== 'adopted') { continue }

    try {
      await processGroup(storage, group, generation)
    } catch (error) {
      // Required checkpoint writes already rolled the atom back. One malformed
      // or unavailable room never blocks independent safe rooms at startup.
      const current = $groupChats.get()[group]

      if (current && !current.tombstone) {
        const issue = issueForError(error)

        $groupChats.set({
          ...$groupChats.get(),
          [group]: { ...current, continuityIssue: issue.message }
        })
      }
    }
  }
}

/** Production startup/reconnect entrypoint. Re-entrant launches share one run;
 * a re-enabled lifecycle waits for the retired run before touching storage. */
export function adoptShippedGroupChats(storage: PluginStorage, restoreOnly = false): Promise<void> {
  observeAdoptedRoutes(storage)
  const generation = lifecycleGeneration

  if (currentRun?.generation === generation) {
    if (!restoreOnly && currentRun.restoreOnly) {
      return currentRun.promise.then(() => generation === lifecycleGeneration ? adoptShippedGroupChats(storage) : undefined)
    }

    return currentRun.promise
  }

  const predecessor = currentRun?.promise.catch(() => undefined) ?? Promise.resolve()
  const promise = predecessor.then(() => runAdoption(storage, generation, restoreOnly))
  currentRun = { generation, restoreOnly, promise }
  void promise.finally(() => {
    if (currentRun?.promise === promise) {
      currentRun = null
    }
  })

  return promise
}

export function stopShippedGroupAdoption(): void {
  lifecycleGeneration += 1
  recoveryDispose?.()
  recoveryDispose = null
  recoveryRun = null
  revokeAdoptedCanonicalGroupsForConnection()
}
