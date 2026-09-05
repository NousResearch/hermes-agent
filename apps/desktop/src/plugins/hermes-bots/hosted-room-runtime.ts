/**
 * Gateway-hosted Group Chat runtime.
 *
 * RPC ownership lives here: capability negotiation, the durable command
 * outbox, monotonic replay, and the bounded refresh loop. Group state remains
 * owned by `group-chat.ts`; creation, round routing, and room UI call this
 * module through narrow verbs.
 */

import { atom, host } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'

import { $lastRoster } from './data'
import {
  $groupChats,
  applyHostedRoomAuthority,
  groupChatHostedGateway,
  mergeGroupChatRoomEntries,
  uniqueGroupChatName,
  updateGroupChat
} from './group-chat'
import { assertGroupFileIntent } from './group-file-errors'
import {
  clearHostedRoomApprovalState,
  resetHostedRoomApprovalState,
  resolveHostedRoomApprovalAttention,
  syncHostedRoomApprovals
} from './hosted-room-approval-state'
import { stageHostedMessageAttachments } from './hosted-room-attachments-client'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import {
  addHostedRoomCleanup,
  armHostedRoomCleanup,
  dispatchHostedRoomCleanup,
  hostedRoomCleanupPending,
  releaseHostedRoomCleanup,
  resetHostedRoomCleanupForTests,
  startHostedRoomCleanup,
  stopHostedRoomCleanup
} from './hosted-room-cleanup'
import {
  classifyHostedRoomCapability,
  createHostedRoomOutbox,
  createHostedRoomReplayState,
  deriveFriendlyHostedRoomStatus,
  isHostedRoomContinuityEligible,
  isHostedRoomReadEligible,
  profileScopedRoomLinkEndpoint,
  replayHostedRoomPages,
  resolveAutonomousRoomPlan
} from './hosted-room-client'
import type {
  AutonomousRoomPlan,
  HostedRoomCapability,
  HostedRoomCommand,
  HostedRoomOutbox,
  HostedRoomRouteResolution,
  reduceHostedRoomOutbox
} from './hosted-room-client'
import {
  failedHostedRoomCommand,
  hostedRoomCommandFailureCode,
  pendingHostedRoomSafetyCommand,
  safetyCommandsBlockedByFailure,
  surfaceHostedRoomCommandFailure
} from './hosted-room-command-failures'
import { readHostedGroupAttachment } from './hosted-room-file-read'
import {
  hostedReadOnlyState,
  hostedRoomCapabilityFingerprint,
  hostedRoomContinuityMode,
  hostedRoomDriverDisplayStatus,
  hostedRoomPollFingerprint,
  hostedStatus,
  hostedUnavailableState,
  readHostedInventoryState,
  readHostedRoomInventory,
  hostedReplayMessages as replayMessages
} from './hosted-room-inventory'
import { revokeInvalidHostedMemberRoutes } from './hosted-room-member-inventory'
import { hostedMemberDescriptors } from './hosted-room-members'
import { HostedRoomObservations } from './hosted-room-observations'
import {
  mutateHostedRoomOutbox,
  recoverHostedRoomOutbox,
  resetHostedRoomOutboxLocksForTests,
  withHostedRoomCommandOrder,
  withHostedRoomOutboxDispatch
} from './hosted-room-outbox'
import { requestHostedConnection, withHostedRoomProbeTimeout } from './hosted-room-transport'
import { hostedUserEventReceipt, outgoingHostedUserEvent, restoreHostedUserOutboxIntents } from './hosted-user-events'
import { botsText } from './i18n'
import { requestForBot } from './routing'
import type { Attachment, GroupChat, GroupMember, GroupMessage, GroupPrompt, ProfileRoute } from './types'

export { $hostedRoomCapabilities } from './hosted-room-capability-state'
export { $hostedRoomCleanup } from './hosted-room-cleanup'
export { describeAutonomousRoomPlan, describeHostedRoomCreationError } from './hosted-room-client'
export { hostedRoomDriverDisplayStatus, hostedRoomPollFingerprint } from './hosted-room-inventory'
export { requestHostedConnection } from './hosted-room-transport'

const HOSTED_ROOM_SYNC_INTERVAL_MS = 5000
const HOSTED_ROOM_UNSUPPORTED_REPROBE_MS = 30_000

export const $hostedRoomOutbox = atom<HostedRoomOutbox>(createHostedRoomOutbox())

const hostedRoomPollCache = new Map<string, string>()
const hostedRoomPollGenerations = new Map<string, number>()
const hostedRoomMutationGenerations = new Map<string, number>()
const hostedRoomLocallyDeleted = new Set<string>()
const hostedRoomObservations = new HostedRoomObservations()
let hostedRoomSyncTimer: ReturnType<typeof setTimeout> | null = null
let hostedRoomSyncRunning = false
let hostedRoomSyncDisposed = true
let hostedRoomLifecycleGeneration = 0
let hostedOutboxDispatchPromise: Promise<void> | null = null
let hostedRoomStorage: null | PluginContext['storage'] = null
let hostedRoomHooks: HostedRoomRuntimeHooks = {}
const hostedUnsupportedUntil = new Map<string, number>()

export function hostedRoomLifecycleToken() {
  return hostedRoomLifecycleGeneration
}

export function hostedRoomLifecycleIsCurrent(token: number) {
  return !hostedRoomSyncDisposed && token === hostedRoomLifecycleGeneration
}

function hostedRoomMutationGeneration(roomId: string) {
  return Math.max(0, Number(hostedRoomMutationGenerations.get(String(roomId || '')) || 0))
}

/** Fence an asynchronous local send/Stop/delete against an older replay. */
export function beginHostedRoomMutation(roomId: string) {
  const id = String(roomId || '')
  const generation = hostedRoomMutationGeneration(id) + 1

  if (id) {
    hostedRoomMutationGenerations.set(id, generation)
  }

  return generation
}

export function hostedRoomMutationIsCurrent(roomId: string, generation: number) {
  const id = String(roomId || '')

  return Boolean(id) && !hostedRoomLocallyDeleted.has(id) && hostedRoomMutationGeneration(id) === generation
}

/** Keep an acknowledged local deletion invisible to stale in-flight polls. */
export function markHostedRoomLocallyDeleted(roomId: string) {
  const id = String(roomId || '')

  if (!id) {
    return
  }

  beginHostedRoomMutation(id)
  hostedRoomLocallyDeleted.add(id)
  hostedRoomPollCache.delete(id)
}

/** A projection-only room must not start a classic Desktop driver until each
 * member gateway has been inventoried. Existing local classic rooms carry
 * either a Desktop authority or non-projected member descriptors and remain
 * immediately usable unless this ID is already known to be hosted. */
export function groupChatContinuityReady(room: GroupChat | null | undefined) {
  if (!room) {
    return true
  }

  if (groupChatHostedGateway(room)) {
    return !['deleted', 'failed', 'read-only', 'unsupported'].includes(String(room.hostedStatus?.state || ''))
  }

  return hostedRoomObservations.classicReady(room)
}

export interface HostedRoomRuntimeHooks {
  renameGroupChat?: (oldName: string, newName: string, members: GroupMember[]) => Promise<null | string>
}

export interface HostedRoomProbe {
  attachmentParity: boolean
  attachmentUnavailableMembers: string[]
  capability: HostedRoomCapability | null
  capabilities: Record<string, HostedRoomCapability>
  eligible: boolean
  route: AutonomousRoomPlan
  routes: Record<string, ProfileRoute>
}

interface HostedRoomCreateInput {
  members: Array<{
    display_name?: string
    handle: string
    member_id: string
    profile: string
  }>
  name: string
  roomId: string
  route: HostedRoomRouteResolution
}

interface AutonomousHostedRoomMember {
  displayName?: string
  handle: string
  member: GroupMember
  profile: string
}

interface AutonomousHostedRoomCreateInput {
  members: AutonomousHostedRoomMember[]
  name: string
  probe: HostedRoomProbe
  roomId: string
}

interface HostedRoomServerState {
  authority_epoch?: unknown
  authority_gateway_id?: unknown
  disbanded_at?: unknown
  latest_seq?: unknown
  members?: unknown
  name?: unknown
  room_id?: unknown
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function activeConnectionId() {
  return String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || '')
}

async function hostedDefaultRoutes(): Promise<ProfileRoute[]> {
  if (typeof host.profileRoutes !== 'function') {
    return []
  }

  const routes = await host.profileRoutes()
  const byConnection = new Map<string, ProfileRoute>()

  for (const route of Array.isArray(routes) ? routes : []) {
    const profile = String(route?.targetProfile || route?.profile || '')
    const connectionId = String(route?.connectionId || '')

    if (!connectionId || profile !== 'default' || byConnection.has(connectionId)) {
      continue
    }

    byConnection.set(connectionId, route as ProfileRoute)
  }

  return [...byConnection.values()]
}

async function verifiedHostedAuthorityRoute(
  routes: ProfileRoute[],
  authorityId: string,
  preferredConnectionId = '',
  purpose: 'control' | 'read' = 'control',
  signal?: AbortSignal
) {
  const ordered = [...routes].sort(
    (left, right) =>
      Number(right.connectionId === preferredConnectionId) - Number(left.connectionId === preferredConnectionId)
  )

  for (const route of ordered) {
    assertGroupFileIntent(signal)
    const connectionId = String(route.connectionId || '')
    const observation = hostedRoomObservations.captureCapability(connectionId)

    try {
      const capability = classifyHostedRoomCapability(
        await hostedRoomObservations.read(observation, () =>
          withHostedRoomProbeTimeout(requestHostedConnection(route, 'groups.capabilities'))
        ),
        { connectionId }
      )

      assertGroupFileIntent(signal)

      if (!hostedRoomObservations.current(observation)) {
        continue
      }

      storeHostedCapabilities({ [connectionId]: capability })

      const eligible = purpose === 'read' ? isHostedRoomReadEligible : isHostedRoomContinuityEligible

      if (capability.authorityId === authorityId && eligible(capability)) {
        return route
      }
    } catch (error) {
      assertGroupFileIntent(signal)

      if (hostedRoomObservations.current(observation)) {
        storeHostedCapabilities({
          [connectionId]: classifyHostedRoomCapability({ ok: false, error }, { connectionId })
        })
      }

      // Capability probes reveal no room data; try the next current route.
    }
  }

  return null
}

function sourceLabel(connectionId: string) {
  const source = ($lastRoster.get() || []).find(row => String(row?.connectionId || '') === connectionId)

  return String(source?.connectionLabel || botsText().group.thisHost)
}

function markHostedConnectionUnavailable(connectionId: string, unsupported = false) {
  const connectionName = sourceLabel(connectionId)

  for (const [name, room] of Object.entries($groupChats.get())) {
    if (String(room?.hostedConnectionId || '') !== connectionId || room.hostedStatus?.state === 'deleted') {
      continue
    }

    updateGroupChat(
      name,
      current =>
        hostedUnavailableState(current, $hostedRoomCapabilities.get()[connectionId], connectionName, unsupported),
      {
        sync: false
      }
    )
    clearHostedRoomApprovalState(name)
  }
}

function isDisbanded(room: HostedRoomServerState) {
  return room.disbanded_at !== null && room.disbanded_at !== undefined
}

function storeHostedCapabilities(next: Record<string, HostedRoomCapability>, replace = false) {
  const current = $hostedRoomCapabilities.get()

  for (const [connectionId, capability] of Object.entries(next)) {
    if (hostedRoomCapabilityFingerprint(current[connectionId]) !== hostedRoomCapabilityFingerprint(capability)) {
      invalidateHostedRoomsForConnection(connectionId, capability.authorityId || '')
    }
  }

  $hostedRoomCapabilities.set(replace ? next : { ...current, ...next })

  for (const [connectionId, capability] of Object.entries(next)) {
    if (!isHostedRoomReadEligible(capability)) {
      markHostedConnectionUnavailable(connectionId, capability.kind === 'unsupported')
    }
  }

  for (const [name, room] of Object.entries($groupChats.get())) {
    const capability = next[String(room.hostedConnectionId || '')]

    if (
      capability &&
      isHostedRoomReadEligible(capability) &&
      room.hostedStatus?.state !== 'deleted' &&
      (!isHostedRoomContinuityEligible(capability) || capability.authorityId !== room.hosted)
    ) {
      updateGroupChat(name, current => ({ ...current, ...hostedReadOnlyState() }), { sync: false })
      clearHostedRoomApprovalState(name)
    }
  }
}

function invalidateHostedRoomsForConnection(connectionId: string, installationId = '') {
  hostedRoomObservations.invalidate(connectionId)

  for (const room of Object.values($groupChats.get())) {
    if (
      room.hostedConnectionId === connectionId ||
      (room.members || []).some(
        member =>
          String(member.route?.connectionId || member.connectionId || '') === connectionId ||
          (installationId && member.hostedIdentity?.installationId === installationId)
      )
    ) {
      hostedRoomPollCache.delete(String(room.roomId || ''))
    }
  }
}

export function invalidateHostedRoomPoll(roomId: string) {
  const id = String(roomId || '')

  hostedRoomPollCache.delete(id)
  hostedRoomPollGenerations.set(id, Number(hostedRoomPollGenerations.get(id) || 0) + 1)
}

export function shouldRefreshHostedRoom(room: GroupChat | undefined, listed: unknown) {
  if (!room) {
    return true
  }

  const activeStates = new Set(['queued', 'sending', 'stopping', 'working'])

  const active =
    room.running === true ||
    activeStates.has(String(room.hostedStatus?.state || '')) ||
    $hostedRoomOutbox.get().commands.some(command => command.roomId === room.roomId && command.status !== 'failed')

  const fingerprint = hostedRoomPollFingerprint(listed)

  return (
    active ||
    room.hostedMembersNeedRefresh ||
    (Boolean(groupChatHostedGateway(room)) && !room.hostedMembersVerified) ||
    hostedRoomPollCache.get(String(room.roomId || '')) !== fingerprint
  )
}

/** Replay every hosted room only after plugin storage/ui_meta hydration has
 * settled. The contiguous cursor is persisted with the room, so reconnects
 * fetch only missing events and a gap never skips unseen history. */
export async function refreshHostedRooms() {
  if (hostedRoomSyncDisposed || hostedRoomSyncRunning) {
    return
  }

  const lifecycleGeneration = hostedRoomLifecycleGeneration
  const syncStale = () => hostedRoomSyncDisposed || lifecycleGeneration !== hostedRoomLifecycleGeneration

  hostedRoomSyncRunning = true

  try {
    const routes = await hostedDefaultRoutes()

    if (syncStale()) {
      return
    }

    const routesByConnection = Object.fromEntries(routes.map(route => [String(route.connectionId || ''), route]))

    for (const id of Object.keys($hostedRoomCapabilities.get())) {
      if (!routesByConnection[id]) {
        invalidateHostedRoomsForConnection(id)
      }
    }

    const capabilities = Object.fromEntries(
      Object.entries($hostedRoomCapabilities.get()).filter(([id]) => routesByConnection[id])
    )

    hostedRoomObservations.retain(Object.keys(routesByConnection), Object.values($groupChats.get()))
    storeHostedCapabilities(capabilities, true)

    if (typeof host.profileRoutes === 'function') {
      revokeInvalidHostedMemberRoutes(routesByConnection, capabilities, invalidateHostedRoomPoll)
    }

    for (const route of routes) {
      if (syncStale()) {
        return
      }

      const connectionId = String(route.connectionId)
      let observation = hostedRoomObservations.capture(connectionId)
      let capability: HostedRoomCapability

      const cached = capabilities[connectionId]

      if (cached?.kind === 'unsupported' && Number(hostedUnsupportedUntil.get(connectionId) || 0) > Date.now()) {
        capability = cached
      } else {
        observation = hostedRoomObservations.captureCapability(connectionId)

        try {
          capability = classifyHostedRoomCapability(await requestHostedConnection(route, 'groups.capabilities'), {
            connectionId
          })
        } catch (error) {
          capability = classifyHostedRoomCapability(
            {
              ok: false,
              error
            },
            {
              connectionId
            }
          )
        }
      }

      if (syncStale()) {
        return
      }

      if (!hostedRoomObservations.current(observation)) {
        capabilities[connectionId] = $hostedRoomCapabilities.get()[connectionId]

        continue
      }

      if (capability !== cached) {
        if (capability.kind === 'unsupported') {
          hostedUnsupportedUntil.set(connectionId, Date.now() + HOSTED_ROOM_UNSUPPORTED_REPROBE_MS)
        } else {
          hostedUnsupportedUntil.delete(connectionId)
        }
      }

      storeHostedCapabilities({ [connectionId]: capability })
      capabilities[connectionId] = capability
      revokeInvalidHostedMemberRoutes(routesByConnection, capabilities, invalidateHostedRoomPoll)
    }

    if (syncStale()) {
      return
    }

    connectionLoop: for (const route of routes) {
      if (syncStale()) {
        return
      }

      const connectionId = String(route.connectionId)
      const capability = $hostedRoomCapabilities.get()[connectionId]

      if (!capability) {
        continue
      }

      const observation = hostedRoomObservations.capture(connectionId)
      const stale = () => syncStale() || !hostedRoomObservations.current(observation)

      const read = <T>(method: string, params: Record<string, unknown>) =>
        hostedRoomObservations.read(observation, () => requestHostedConnection<T>(route, method, params))

      if (!isHostedRoomReadEligible(capability)) {
        markHostedConnectionUnavailable(connectionId, capability.kind === 'unsupported')

        if (capability.reason === 'old-gateway') {
          hostedRoomObservations.publish(hostedRoomObservations.capture(connectionId), new Set(), true)
        }

        continue
      }

      let inventory: Awaited<ReturnType<typeof readHostedRoomInventory>>

      try {
        inventory = await readHostedRoomInventory(
          params => read('groups.list', params),
          ids => {
            if (!stale()) {
              hostedRoomObservations.observe(observation, ids)
            }
          }
        )
      } catch {
        if (stale()) {
          continue
        }

        invalidateHostedRoomsForConnection(connectionId)
        markHostedConnectionUnavailable(connectionId)

        continue
      }

      if (stale()) {
        continue
      }

      const listedRooms = inventory.rooms

      // IDs establish absence independently of each known room's display replay.
      hostedRoomObservations.publish(observation, inventory.ids, inventory.complete)

      const disbandedIds = new Set(
        listedRooms
          .map(raw => (record(raw) || {}) as HostedRoomServerState)
          .filter(isDisbanded)
          .map(room => String(room.room_id || ''))
          .filter(Boolean)
      )

      const caughtUpDisbandedIds = new Set<string>()

      for (const listedRaw of listedRooms) {
        if (stale()) {
          continue connectionLoop
        }

        const listedRoom = (record(listedRaw) || {}) as HostedRoomServerState
        const roomId = String(listedRoom.room_id || '')
        const serverName = String(listedRoom.name || '').trim()

        if (!roomId || !serverName || hostedRoomLocallyDeleted.has(roomId)) {
          continue
        }

        const existingEntry = Object.entries($groupChats.get()).find(
          ([, room]) => String(room?.roomId || '') === roomId
        )

        const includeDisbanded = isDisbanded(listedRoom)

        // A client that already joined the room must replay terminal events
        // committed while it was offline before painting the remote disband.
        // Unknown disbanded rooms remain invisible on newly connected clients.
        if (includeDisbanded && !existingEntry) {
          continue
        }

        if (!shouldRefreshHostedRoom(existingEntry?.[1], listedRoom)) {
          if (
            includeDisbanded &&
            Math.max(0, Number(existingEntry?.[1]?.hostedSeq || 0)) >= Math.max(0, Number(listedRoom.latest_seq || 0))
          ) {
            caughtUpDisbandedIds.add(roomId)
          }

          continue
        }

        const refreshGeneration = hostedRoomMutationGeneration(roomId)
        const pollGeneration = Number(hostedRoomPollGenerations.get(roomId) || 0)

        let stateResponse: Record<string, unknown>
        let serverRoom: Record<string, unknown>

        try {
          stateResponse = await read('groups.state', {
            room_id: roomId,
            ...(includeDisbanded ? { include_disbanded: true } : {})
          })
          serverRoom = readHostedInventoryState(stateResponse, roomId)
        } catch {
          if (stale()) {
            continue connectionLoop
          }

          markHostedConnectionUnavailable(connectionId)

          continue
        }

        if (stale()) {
          continue connectionLoop
        }

        if (!hostedRoomMutationIsCurrent(roomId, refreshGeneration)) {
          continue
        }

        const ownership = applyHostedRoomAuthority(
          existingEntry?.[1] || { roomId, log: [], watermarks: {} },
          serverRoom
        )

        if (
          ownership.hosted !== serverRoom.authority_gateway_id ||
          ownership.hostedEpoch !== serverRoom.authority_epoch
        ) {
          continue
        }

        const writable =
          isHostedRoomContinuityEligible(capability) && capability.authorityId === serverRoom.authority_gateway_id

        let existingName = existingEntry?.[0]
        let existing = existingEntry?.[1]
        const taken = new Set(Object.keys($groupChats.get()))

        let localName =
          existingName ||
          (taken.has(serverName)
            ? uniqueGroupChatName(`${serverName} (${sourceLabel(connectionId)})`, taken)
            : serverName)

        const renamePending = $hostedRoomOutbox
          .get()
          .commands.some(
            command => command.kind === 'rename' && command.roomId === roomId && command.status !== 'failed'
          )

        if (existingName && existingName !== serverName && !renamePending && hostedRoomHooks.renameGroupChat) {
          const occupant = $groupChats.get()[serverName]
          const renameTaken = new Set(taken)

          renameTaken.delete(existingName)

          const targetName =
            occupant && occupant.roomId !== roomId
              ? uniqueGroupChatName(`${serverName} (${sourceLabel(connectionId)})`, renameTaken)
              : serverName

          const renamed = await hostedRoomHooks.renameGroupChat(
            existingName,
            targetName,
            Array.isArray(existing?.members) ? existing.members : []
          )

          if (renamed) {
            existingName = renamed
            localName = renamed
            existing = $groupChats.get()[renamed]
          }

          if (stale()) {
            continue connectionLoop
          }

          if (!hostedRoomMutationIsCurrent(roomId, refreshGeneration)) {
            continue
          }
        }

        const replay = await replayHostedRoomPages({
          state: createHostedRoomReplayState({
            roomId,
            name: serverName,
            members: Array.isArray(serverRoom.members) ? (serverRoom.members as Array<Record<string, unknown>>) : [],
            authorityId: String(serverRoom.authority_gateway_id || capability.authorityId),
            authorityEpoch: Number(serverRoom.authority_epoch || 1),
            connectionId,
            cursor: Number(existing?.hostedSeq || 0)
          }),
          fetchPage: request =>
            read('groups.log', {
              room_id: roomId,
              since_seq: request.sinceSeq,
              limit: request.limit,
              ...(includeDisbanded ? { include_disbanded: true } : {})
            }),
          pageSize: capability.maxLogLimit || 100
        })

        if (stale()) {
          continue connectionLoop
        }

        if (!hostedRoomMutationIsCurrent(roomId, refreshGeneration)) {
          continue
        }

        const replayStatus = deriveFriendlyHostedRoomStatus(replay.state)
        const driver = record(stateResponse.driver_status)

        const reconnectRoute = (Array.isArray(driver?.peer_routes) ? driver.peer_routes : [])
          .map(record)
          .find(route => route?.status === 'needs_reauthorization' && String(route?.member_id || ''))

        const reconnectMemberId = String(reconnectRoute?.member_id || '')

        const reconnectMember = (Array.isArray(serverRoom.members) ? serverRoom.members : [])
          .map(record)
          .find(member => String(member?.member_id || '') === reconnectMemberId)

        const reconnectName = String(
          reconnectMember?.display_name || reconnectMember?.handle || reconnectMember?.profile || botsText().group.aBot
        )

        const reconnectTarget = record(reconnectMember?.target)
        const reconnectAuthority = String(reconnectTarget?.installation_id || reconnectTarget?.peer_id || '')

        const reconnectPrior = (existing?.members || []).find(
          member =>
            String(member.handle || member.name || '') ===
              String(reconnectMember?.handle || reconnectMember?.profile || '') &&
            String(member.targetProfile || member.name || '') ===
              String(reconnectMember?.profile || reconnectMember?.member_id || '')
        )

        const reconnectConnectionId =
          Object.entries(capabilities).find(([, candidate]) => candidate.authorityId === reconnectAuthority)?.[0] ||
          String(reconnectPrior?.route?.connectionId || reconnectPrior?.connectionId || '')

        const reconnectCapability = reconnectConnectionId ? capabilities[reconnectConnectionId] : undefined
        const reconnectCapabilityKnown = Boolean(reconnectCapability)

        const reconnectSupported = Boolean(
          capability.routeGrantFingerprint &&
          reconnectConnectionId &&
          reconnectCapability?.kind === 'driver-capable' &&
          reconnectCapability.exactPeerGrantRevoke
        )

        const reconnectUpdateConnectionId = !capability.routeGrantFingerprint
          ? connectionId
          : reconnectCapability?.kind === 'unsupported' ||
              (reconnectCapability?.kind === 'driver-capable' && !reconnectCapability.exactPeerGrantRevoke)
            ? reconnectConnectionId
            : ''

        const stopping = $hostedRoomOutbox
          .get()
          .commands.some(
            command =>
              command.roomId === roomId && ['disband', 'stop'].includes(command.kind) && command.status !== 'failed'
          )

        const friendly = reconnectMemberId
          ? {
              ...replayStatus,
              kind: 'needs-attention' as const,
              member: reconnectName,
              canRetry: false,
              canStop: false
            }
          : hostedRoomDriverDisplayStatus(replayStatus, driver, { stopping })

        const running = ['queued', 'stopping', 'working'].includes(friendly.kind)

        const pendingActions = Array.isArray(driver?.pending_actions) ? driver.pending_actions : []

        const retryAction = pendingActions
          .map(record)
          .find(action => action?.kind === 'retry' && String(action?.task_id || ''))

        const commandFailure = failedHostedRoomCommand($hostedRoomOutbox.get(), roomId)

        const memberDescriptors = hostedMemberDescriptors(
          serverRoom,
          connectionId,
          existing?.members || [],
          capabilities,
          sourceLabel
        )

        updateGroupChat(
          localName,
          current => {
            if (stale()) {
              return current
            }

            const authoritative = applyHostedRoomAuthority(current, serverRoom as Record<string, unknown>)

            return {
              ...authoritative,
              roomId,
              members: memberDescriptors,
              hostedMembersVerified: true,
              hostedMembersNeedRefresh: false,
              log: mergeGroupChatRoomEntries(
                current,
                restoreHostedUserOutboxIntents(current, $hostedRoomOutbox.get()),
                replayMessages(replay.state.messages)
              ),
              hostedConnectionId: connectionId,
              hostedSeq: replay.state.cursor,
              hostedStatus: commandFailure
                ? {
                    canRetry: true,
                    canStop: friendly.canStop,
                    label: botsText().group.hostedNeedsAttention,
                    retryCommandId: commandFailure.commandId,
                    state: 'failed'
                  }
                : {
                    ...hostedStatus(friendly, sourceLabel(connectionId)),
                    ...(retryAction && !reconnectMemberId ? { taskId: String(retryAction.task_id) } : {}),
                    ...(reconnectMemberId && reconnectSupported
                      ? {
                          canReconnect: true,
                          reconnectMemberId
                        }
                      : {}),
                    ...(reconnectMemberId &&
                    (!reconnectCapabilityKnown || reconnectCapability?.kind === 'transient-failure')
                      ? { canRetry: true }
                      : {}),
                    ...(!replay.complete && !reconnectMemberId ? { canRetry: true } : {})
                  },
              continuityMode: hostedRoomContinuityMode(serverRoom),
              continuityIssue: commandFailure
                ? botsText().group.hostRejectedCommand
                : reconnectMemberId
                  ? !reconnectCapabilityKnown || reconnectCapability?.kind === 'transient-failure'
                    ? botsText().group.reconnectFailed
                    : reconnectSupported
                      ? botsText().group.memberReconnectToContinue(reconnectName)
                      : botsText().group.hostUpdateNeeded(
                          reconnectUpdateConnectionId ? sourceLabel(reconnectUpdateConnectionId) : reconnectName
                        )
                  : replay.complete
                    ? null
                    : botsText().group.hostedSyncing,
              running,
              ...(!writable ? hostedReadOnlyState() : {})
            }
          },
          {
            sync: false
          }
        )

        if (stale()) {
          continue connectionLoop
        }

        if (writable) {
          syncHostedRoomApprovals(localName, serverRoom, memberDescriptors, pendingActions)
        } else {
          clearHostedRoomApprovalState(localName)
        }

        if (stale()) {
          continue connectionLoop
        }

        if (
          replay.complete &&
          (!reconnectMemberId || Boolean(reconnectUpdateConnectionId)) &&
          Number(hostedRoomPollGenerations.get(roomId) || 0) === pollGeneration
        ) {
          hostedRoomPollCache.set(roomId, hostedRoomPollFingerprint(listedRoom))

          if (includeDisbanded) {
            caughtUpDisbandedIds.add(roomId)
          }
        } else {
          hostedRoomPollCache.delete(roomId)
        }
      }

      // Keep the local shell long enough to explain a disband observed on
      // another client. Silently deleting only the room atom would strand an
      // open workspace and leave membership metadata half-cleaned. The normal
      // local disband action performs the complete cross-module cleanup.
      if (disbandedIds.size) {
        for (const [name, room] of Object.entries($groupChats.get())) {
          if (stale()) {
            continue connectionLoop
          }

          if (
            room.roomId &&
            disbandedIds.has(room.roomId) &&
            caughtUpDisbandedIds.has(room.roomId) &&
            room.hostedConnectionId === connectionId
          ) {
            updateGroupChat(
              name,
              current => ({
                ...current,
                running: false,
                hostedStatus: {
                  state: 'deleted',
                  label: botsText().group.hostedDeleted
                },
                continuityIssue: botsText().group.hostedDeleteLocally
              }),
              {
                sync: false
              }
            )
            clearHostedRoomApprovalState(name)
          }
        }
      }

      if (inventory.complete) {
        const listedIds = new Set(listedRooms.map(raw => String(record(raw)?.room_id || '')).filter(Boolean))

        for (const [name, room] of Object.entries($groupChats.get())) {
          if (stale()) {
            continue connectionLoop
          }

          const roomId = String(room?.roomId || '')

          if (!roomId || room.hostedConnectionId !== connectionId || listedIds.has(roomId)) {
            continue
          }

          try {
            await read('groups.state', {
              room_id: roomId,
              include_disbanded: true
            })

            continue
          } catch (error) {
            if (stale()) {
              continue connectionLoop
            }

            const message = String(record(error)?.message || record(record(error)?.error)?.message || error || '')

            if (!/history expired|permanently retired|hosted room not found/i.test(message)) {
              continue
            }
          }

          hostedRoomPollCache.delete(roomId)
          updateGroupChat(
            name,
            current => ({
              ...current,
              running: false,
              hostedStatus: {
                state: 'deleted',
                label: botsText().group.hostedDeleted
              },
              continuityIssue: botsText().group.hostedDeleteLocally
            }),
            { sync: false }
          )
          clearHostedRoomApprovalState(name)
        }
      }
    }
  } finally {
    hostedRoomSyncRunning = false
  }
}

function scheduleHostedRoomSync(delay = HOSTED_ROOM_SYNC_INTERVAL_MS) {
  if (hostedRoomSyncDisposed || typeof setTimeout !== 'function') {
    return
  }

  if (hostedRoomSyncTimer) {
    clearTimeout(hostedRoomSyncTimer)
  }

  hostedRoomSyncTimer = setTimeout(() => {
    hostedRoomSyncTimer = null
    void dispatchHostedRoomCleanup()
      .catch(() => undefined)
      .then(() => refreshHostedRooms())
      .catch(() => undefined)
      .then(() => dispatchHostedRoomOutbox())
      .catch(() => undefined)
      .then(() => scheduleHostedRoomSync())
  }, delay)

  const timer = hostedRoomSyncTimer as ReturnType<typeof setTimeout> & { unref?: () => void }
  timer?.unref?.()
}

async function transitionHostedRoomOutbox(action: Parameters<typeof reduceHostedRoomOutbox>[1]) {
  const previous = $hostedRoomOutbox.get()

  try {
    const next = await mutateHostedRoomOutbox(hostedRoomStorage, action)

    $hostedRoomOutbox.set(next)

    return next
  } catch (error) {
    $hostedRoomOutbox.set(previous)
    throw error
  }
}

async function consumeImmediateHostedRoomCommandFailure(commandId: unknown) {
  const id = String(commandId || '')

  const failed = $hostedRoomOutbox
    .get()
    .commands.find(command => command.commandId === id && command.status === 'failed')

  if (!failed) {
    return false
  }

  await transitionHostedRoomOutbox({ type: 'acknowledge', commandId: id })
  const roomName = Object.entries($groupChats.get()).find(([, room]) => room.roomId === failed.roomId)?.[0]

  if (roomName) {
    updateGroupChat(
      roomName,
      room => ({
        ...room,
        hostedStatus: {
          canStop: room.hostedStatus?.canStop,
          label: botsText().roster.ready,
          state: 'ready'
        },
        continuityIssue: null
      }),
      { sync: false }
    )
  }

  hostedRoomPollCache.delete(failed.roomId)
  await refreshHostedRooms().catch(() => undefined)

  return true
}

export function dispatchHostedRoomOutbox(): Promise<void> {
  if (hostedRoomSyncDisposed) {
    return Promise.resolve()
  }

  if (hostedOutboxDispatchPromise) {
    return hostedOutboxDispatchPromise.then(() => dispatchHostedRoomOutbox())
  }

  const run = withHostedRoomOutboxDispatch(async () => {
    if (hostedRoomSyncDisposed) {
      return
    }

    let state = await recoverHostedRoomOutbox(hostedRoomStorage)
    const routes = await hostedDefaultRoutes()

    for (const safety of safetyCommandsBlockedByFailure(state)) {
      state = await transitionHostedRoomOutbox({ type: 'enqueue-safety', command: safety })
    }

    const blockedRooms = new Set(
      state.commands.filter(command => command.status === 'failed').map(command => command.roomId)
    )

    $hostedRoomOutbox.set(state)

    for (const failed of state.commands.filter(command => command.status === 'failed')) {
      surfaceHostedRoomCommandFailure(failed)
    }

    for (const command of state.commands.filter(entry => entry.status === 'pending')) {
      if (hostedRoomSyncDisposed) {
        return
      }

      if (blockedRooms.has(command.roomId)) {
        continue
      }

      const exact = routes.find(candidate => candidate.connectionId === command.connectionId)

      const route = command.authorityId
        ? await verifiedHostedAuthorityRoute(routes, command.authorityId, command.connectionId)
        : exact

      if (!route) {
        blockedRooms.add(command.roomId)

        continue
      }

      state = await transitionHostedRoomOutbox({
        type: 'dispatch',
        commandId: command.commandId
      })

      const claimed = state.commands.find(entry => entry.commandId === command.commandId)

      if (!claimed || claimed.status !== 'in-flight') {
        continue
      }

      const method: Record<HostedRoomCommand['kind'], string> = {
        create: 'groups.create',
        retry: 'groups.retry',
        rename: 'groups.rename',
        send: 'groups.send',
        stop: 'groups.stop',
        disband: 'groups.disband'
      }

      const params =
        command.kind === 'send'
          ? {
              room_id: command.roomId,
              event_id: command.commandId,
              payload: command.payload
            }
          : command.kind === 'rename'
            ? {
                room_id: command.roomId,
                event_id: command.commandId,
                name: command.payload.name
              }
            : command.kind === 'retry'
              ? {
                  room_id: command.roomId,
                  task_id: command.payload.task_id,
                  command_id: command.commandId
                }
              : command.kind === 'stop' || command.kind === 'disband'
                ? {
                    room_id: command.roomId,
                    cancel_id: command.commandId
                  }
                : command.payload

      try {
        const reply = await requestHostedConnection(route, method[command.kind], params)

        // Keep the persisted in-flight command untouched when the window is
        // disposed mid-request. Rehydration returns it to pending with the
        // same idempotency key, covering an unknown server outcome safely.
        if (hostedRoomSyncDisposed) {
          return
        }

        const receipt = hostedUserEventReceipt(command, reply)

        const local =
          receipt &&
          Object.entries($groupChats.get()).find(
            ([, room]) =>
              room.roomId === command.roomId &&
              (!command.authorityId || groupChatHostedGateway(room) === command.authorityId)
          )

        if (receipt && local) {
          updateGroupChat(local[0], current => ({
            ...current,
            log: mergeGroupChatRoomEntries(current, current.log || [], [receipt])
          }))
        }

        state = await transitionHostedRoomOutbox({
          type: 'acknowledge',
          commandId: command.commandId
        })
      } catch (error) {
        const failureCode = hostedRoomCommandFailureCode(error, claimed)
        const terminal = Boolean(failureCode)

        state = await transitionHostedRoomOutbox(
          terminal
            ? {
                type: 'terminal-failure',
                commandId: command.commandId,
                failureCode
              }
            : {
                type: 'transient-failure',
                commandId: command.commandId
              }
        )

        if (terminal) {
          const failed = state.commands.find(entry => entry.commandId === command.commandId)

          const safety = pendingHostedRoomSafetyCommand(state, command.roomId)

          if (failed) {
            surfaceHostedRoomCommandFailure(failed)
          }

          if (safety) {
            state = await transitionHostedRoomOutbox({ type: 'enqueue-safety', command: safety })
            blockedRooms.delete(command.roomId)
          } else {
            blockedRooms.add(command.roomId)
          }
        } else {
          blockedRooms.add(command.roomId)
        }
      }
    }
  })

  let owned: Promise<void>

  owned = run.finally(() => {
    if (hostedOutboxDispatchPromise === owned) {
      hostedOutboxDispatchPromise = null
    }
  })
  hostedOutboxDispatchPromise = owned

  return owned
}

async function enqueueHostedRoomCommand(command: Partial<HostedRoomCommand>) {
  await withHostedRoomCommandOrder(String(command.roomId || ''), () =>
    transitionHostedRoomOutbox({
      type: command.kind === 'disband' || command.kind === 'stop' ? 'enqueue-safety' : 'enqueue',
      command
    })
  )
  await dispatchHostedRoomOutbox()

  if (await consumeImmediateHostedRoomCommandFailure(command.commandId)) {
    throw new Error(botsText().group.hostRejectedCommand)
  }

  const pending = $hostedRoomOutbox.get().commands.find(entry => entry.commandId === command.commandId)

  scheduleHostedRoomSync(0)

  return !pending
}

export async function hostedRouteForRoom(
  room: GroupChat,
  purpose: 'control' | 'read' = 'control',
  signal?: AbortSignal
) {
  assertGroupFileIntent(signal)
  const connectionId = String(room?.hostedConnectionId || '')
  const routes = await hostedDefaultRoutes()
  assertGroupFileIntent(signal)
  const authorityId = groupChatHostedGateway(room)

  if (authorityId) {
    return verifiedHostedAuthorityRoute(routes, authorityId, connectionId, purpose, signal)
  }

  if (connectionId) {
    const exact = routes.find(candidate => candidate.connectionId === connectionId)

    if (exact) {
      return exact
    }
  }

  return null
}

export async function approveHostedGroupChat(entry: GroupPrompt, choice: string) {
  const approval = entry.hostedApproval
  const room = $groupChats.get()[entry.group]
  const route = room ? await hostedRouteForRoom(room) : null

  if (!approval || !route || !['once', 'deny'].includes(choice)) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  await requestHostedConnection(route, 'groups.approve', {
    room_id: approval.roomId,
    member_id: approval.memberId,
    task_id: approval.taskId,
    execution_generation: approval.executionGeneration,
    choice,
    request_id: entry.requestId
  })
  await refreshHostedRooms().catch(() => undefined)
  resolveHostedRoomApprovalAttention(entry)
  scheduleHostedRoomSync(0)
}

export async function probeHostedRoomMembers(members: GroupMember[]): Promise<HostedRoomProbe> {
  const routes = Object.fromEntries(
    (await hostedDefaultRoutes()).map(route => [String(route.connectionId || ''), route])
  )

  const connectionIds = [
    ...new Set(
      (Array.isArray(members) ? members : [])
        .map(member => String(member?.route?.connectionId || member?.connectionId || activeConnectionId() || ''))
        .filter(Boolean)
    )
  ]

  const capabilities: Record<string, HostedRoomCapability> = {}
  const now = Date.now()

  await Promise.all(
    connectionIds.map(async connectionId => {
      const cached = $hostedRoomCapabilities.get()[connectionId]

      if (cached?.kind === 'unsupported' && Number(hostedUnsupportedUntil.get(connectionId) || 0) > now) {
        capabilities[connectionId] = cached

        return
      }

      const observation = hostedRoomObservations.captureCapability(connectionId)
      const route = routes[connectionId]
      let capability: HostedRoomCapability

      try {
        capability = classifyHostedRoomCapability(
          route
            ? await withHostedRoomProbeTimeout(requestHostedConnection(route, 'groups.capabilities'))
            : { ok: false, error: new Error('Gateway route unavailable') },
          { connectionId }
        )
      } catch (error) {
        capability = classifyHostedRoomCapability({ ok: false, error }, { connectionId })
      }

      if (!hostedRoomObservations.current(observation)) {
        capabilities[connectionId] = $hostedRoomCapabilities.get()[connectionId]

        return
      }

      storeHostedCapabilities({ [connectionId]: capability })
      capabilities[connectionId] = capability

      if (capability.kind === 'unsupported') {
        hostedUnsupportedUntil.set(connectionId, now + HOSTED_ROOM_UNSUPPORTED_REPROBE_MS)
      } else {
        hostedUnsupportedUntil.delete(connectionId)
      }
    })
  )

  for (const connectionId of connectionIds) {
    capabilities[connectionId] = $hostedRoomCapabilities.get()[connectionId] || capabilities[connectionId]
  }

  const route = resolveAutonomousRoomPlan(members, {
    activeConnectionId: activeConnectionId(),
    capabilities
  })

  const capability = route.connectionId ? capabilities[route.connectionId] || null : null
  const homeConnectionId = String(route.homeConnectionId || route.connectionId || '')

  const attachmentUnavailableConnections = new Set(
    connectionIds.filter(connectionId => {
      const candidate = capabilities[connectionId]

      return (
        candidate?.limits.attachments !== true ||
        (connectionId !== homeConnectionId && candidate?.roomLink?.catalog?.attachments !== true)
      )
    })
  )

  return {
    attachmentParity:
      Boolean(homeConnectionId) &&
      capabilities[homeConnectionId]?.limits.attachments === true &&
      route.remoteConnectionIds.every(
        connectionId =>
          capabilities[connectionId]?.limits.attachments === true &&
          capabilities[connectionId]?.roomLink?.catalog?.attachments === true
      ),
    attachmentUnavailableMembers: members
      .filter(member =>
        attachmentUnavailableConnections.has(
          String(member?.route?.connectionId || member?.connectionId || activeConnectionId() || '')
        )
      )
      .map(member => String(member.display_name || member.handle || member.name || botsText().group.aBot)),
    route,
    routes,
    capabilities,
    capability,
    eligible: route.kind !== 'unsupported' && isHostedRoomContinuityEligible(capability)
  }
}

export async function createHostedGroupChat({ route, roomId, name, members }: HostedRoomCreateInput): Promise<{
  authorityEpoch: number
  authorityId: string
  connectionId: string
}> {
  if ((route.kind !== 'single-gateway' && route.kind !== 'multi-gateway') || !route.connectionId) {
    throw new Error(botsText().group.botsNeedOneHost)
  }

  const profileRoute = (await hostedDefaultRoutes()).find(candidate => candidate.connectionId === route.connectionId)

  if (!profileRoute) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  let room: Record<string, unknown> | null = null

  try {
    const result = await requestHostedConnection<Record<string, unknown>>(profileRoute, 'groups.create', {
      room_id: roomId,
      name,
      members
    })

    room = record(result.room)
  } catch (createError) {
    // A dropped response has an unknown outcome. Verify the idempotent room id
    // before falling back to Desktop, or both drivers could start the first
    // user turn. A true create failure has no state and safely falls through.
    try {
      const state = await requestHostedConnection<Record<string, unknown>>(profileRoute, 'groups.state', {
        room_id: roomId
      })

      room = record(state.room)
    } catch {
      throw createError
    }
  }

  const authorityId = String(room?.authority_gateway_id || '')

  if (!authorityId) {
    throw new Error(botsText().group.hostRejectedCommand)
  }

  return {
    authorityId,
    authorityEpoch: Math.max(1, Number(room?.authority_epoch || 1)),
    connectionId: route.connectionId
  }
}

export async function createAutonomousHostedGroupChat({
  probe,
  roomId,
  name,
  members
}: AutonomousHostedRoomCreateInput) {
  const plan = probe.route
  const homeConnectionId = String(plan.homeConnectionId || '')
  const homeRoute = probe.routes[homeConnectionId]
  const homeCapability = probe.capabilities[homeConnectionId]

  if (!probe.eligible || !homeConnectionId || !homeRoute || !homeCapability?.authorityId) {
    throw new Error('This Group Chat cannot continue without Desktop yet.')
  }

  const hostedMembers: Array<Record<string, unknown>> = []
  const peerRegistrations: Array<Record<string, unknown>> = []

  try {
    await addHostedRoomCleanup({
      operationId: `${roomId}:home-disband`,
      setupId: roomId,
      kind: 'home-disband',
      connectionId: homeConnectionId,
      roomId,
      cancelId: `rollback-${roomId}`
    })

    for (const [index, item] of members.entries()) {
      const connectionId = String(item.member.route?.connectionId || item.member.connectionId || '')
      const profile = String(item.member.targetProfile || item.profile || item.member.name || 'default')
      const memberId = `member-${index + 1}-${profile}`.replace(/[^A-Za-z0-9._:-]/g, '-').slice(0, 128)

      const descriptor: Record<string, unknown> = {
        member_id: memberId,
        profile,
        handle: item.handle,
        ...(item.displayName
          ? {
              display_name: item.displayName
            }
          : {})
      }

      if (connectionId === homeConnectionId) {
        hostedMembers.push(descriptor)

        continue
      }

      const invitation = record(
        await requestForBot(item.member, 'groups.peer.invite', {
          room_id: roomId,
          home_install_id: homeCapability.authorityId,
          authority_gateway_id: homeCapability.authorityId,
          authority_epoch: 1,
          member_id: memberId,
          profile
        })
      )

      const catalog = record(invitation?.catalog)
      const invitedProfile = String(invitation?.target_profile || profile || '')

      const scopedTargetUrl = profileScopedRoomLinkEndpoint(
        probe.capabilities[connectionId]?.roomLink?.endpoint,
        invitation?.target_profile
      )

      if (invitation?.grant && invitedProfile) {
        await addHostedRoomCleanup({
          operationId: `${roomId}:peer-revoke:${memberId}`,
          setupId: roomId,
          kind: 'peer-revoke',
          connectionId,
          profile: invitedProfile,
          grant: String(invitation.grant)
        })
      }

      if (
        !scopedTargetUrl ||
        !invitation?.grant ||
        !catalog?.installation_id ||
        !catalog.catalog_digest ||
        !invitation.target_profile
      ) {
        throw new Error('One selected Bot could not prepare this Group Chat.')
      }

      hostedMembers.push({
        ...descriptor,
        profile: invitation.target_profile,
        target: {
          kind: 'peer',
          peer_id: catalog.installation_id,
          installation_id: catalog.installation_id,
          profile: invitation.target_profile,
          capability_digest: catalog.catalog_digest
        }
      })
      peerRegistrations.push({
        room_id: roomId,
        member_id: memberId,
        target_url: scopedTargetUrl,
        target_profile: invitation.target_profile,
        grant: invitation.grant,
        catalog
      })
    }

    const created = await createHostedGroupChat({
      route: plan,
      roomId,
      name,
      members: hostedMembers as HostedRoomCreateInput['members']
    })

    for (const registration of peerRegistrations) {
      await requestHostedConnection(homeRoute, 'groups.peer.register', registration)
    }

    await releaseHostedRoomCleanup(roomId)

    return {
      ...created,
      continuityMode: plan.kind === 'multi-gateway' ? ('distributed' as const) : ('gateway' as const)
    }
  } catch (error) {
    await armHostedRoomCleanup(roomId).catch(() => undefined)
    await dispatchHostedRoomCleanup().catch(() => undefined)

    if (hostedRoomCleanupPending(roomId)) {
      throw Object.assign(
        new Error('Some selected Bots could not finish cleanup. Reconnect them before trying again.', {
          cause: error
        }),
        {
          fallbackSafe: false
        }
      )
    }

    throw error
  }
}

async function hostedGroupChatSendCommand(group: string, message: GroupMessage, thread: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || !groupChatHostedGateway(room)) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  const commandId = String(message.id || '')

  if (message.from.kind === 'user' && !message.seq && !message.eventId && room.log.includes(message)) {
    updateGroupChat(group, current => ({
      ...current,
      log: (current.log || []).map(entry =>
        entry === message ? outgoingHostedUserEvent(entry, room.roomId!, commandId) : entry
      )
    }))
  }

  const route = await hostedRouteForRoom(room)
  const connectionId = String(route?.connectionId || room.hostedConnectionId || '')

  if (!connectionId) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  const attachments = Array.isArray(message.images)
    ? message.images.filter((attachment): attachment is Attachment => Boolean(attachment?.data))
    : []

  if (attachments.length) {
    const parity = await probeHostedRoomMembers(room.members || [])

    if (!route || !parity.attachmentParity) {
      throw new Error(
        botsText().group.hostedAttachmentMemberUnavailable(parity.attachmentUnavailableMembers.join(', '))
      )
    }
  }

  const manifest = attachments.length
    ? await stageHostedMessageAttachments(requestHostedConnection, route as ProfileRoute, room.roomId, attachments)
    : []

  return {
    commandId,
    kind: 'send' as const,
    roomId: room.roomId,
    authorityId: groupChatHostedGateway(room),
    connectionId,
    payload: {
      text: message.text || '',
      thread_id: thread,
      ...(manifest.length ? { attachments: manifest } : {})
    }
  }
}

export async function queueHostedGroupChat(group: string, message: GroupMessage, thread: string) {
  const roomId = String($groupChats.get()[group]?.roomId || '')

  await withHostedRoomCommandOrder(roomId, async () => {
    const command = await hostedGroupChatSendCommand(group, message, thread)

    await transitionHostedRoomOutbox({ type: 'enqueue', command })
  })
  await dispatchHostedRoomOutbox()

  if (await consumeImmediateHostedRoomCommandFailure(message.id)) {
    throw new Error(botsText().group.hostRejectedCommand)
  }

  const pending = $hostedRoomOutbox.get().commands.find(entry => entry.commandId === message.id)

  scheduleHostedRoomSync(0)

  return !pending
}

export async function sendHostedGroupChat(group: string, message: GroupMessage, thread: string) {
  const roomId = String($groupChats.get()[group]?.roomId || '')
  let command: Partial<HostedRoomCommand> = {}

  await withHostedRoomCommandOrder(roomId, async () => {
    command = await hostedGroupChatSendCommand(group, message, thread)
    await transitionHostedRoomOutbox({ type: 'enqueue', command })
  })
  await dispatchHostedRoomOutbox()

  if (await consumeImmediateHostedRoomCommandFailure(command.commandId)) {
    throw new Error(botsText().group.hostRejectedCommand)
  }

  const pending = $hostedRoomOutbox.get().commands.find(entry => entry.commandId === command.commandId)

  scheduleHostedRoomSync(0)

  return !pending
}

export async function readHostedGroupChatAttachment(
  group: string,
  message: GroupMessage,
  attachment: Attachment,
  signal?: AbortSignal
) {
  return readHostedGroupAttachment(group, message, attachment, hostedRouteForRoom, signal)
}

export async function stopHostedGroupChat(group: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || !groupChatHostedGateway(room)) {
    return false
  }

  const route = await hostedRouteForRoom(room)
  const connectionId = String(route?.connectionId || room.hostedConnectionId || '')

  if (!connectionId) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  return enqueueHostedRoomCommand({
    commandId: crypto.randomUUID(),
    kind: 'stop',
    roomId: room.roomId,
    authorityId: groupChatHostedGateway(room),
    connectionId,
    payload: {}
  })
}

export async function retryHostedGroupChat(group: string, taskId: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || !groupChatHostedGateway(room) || !String(taskId || '').trim()) {
    return false
  }

  const route = await hostedRouteForRoom(room)
  const connectionId = String(route?.connectionId || room.hostedConnectionId || '')

  if (!connectionId) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  return enqueueHostedRoomCommand({
    commandId: crypto.randomUUID(),
    kind: 'retry',
    roomId: room.roomId,
    authorityId: groupChatHostedGateway(room),
    connectionId,
    payload: { task_id: String(taskId).trim() }
  })
}

/** Resume bounded history replay without retrying any Bot work. */
export async function retryHostedRoomReplay(group: string) {
  const room = $groupChats.get()[group]
  const roomId = String(room?.roomId || '')

  if (!roomId || !groupChatHostedGateway(room)) {
    return false
  }

  hostedRoomPollCache.delete(roomId)
  await refreshHostedRooms()
  scheduleHostedRoomSync(0)

  return true
}

export async function retryFailedHostedRoomCommand(group: string, commandId: string) {
  const room = $groupChats.get()[group]
  const failed = failedHostedRoomCommand($hostedRoomOutbox.get(), String(room?.roomId || ''))

  if (!room || !failed || failed.commandId !== String(commandId || '')) {
    return false
  }

  await transitionHostedRoomOutbox({ type: 'retry', commandId: failed.commandId })
  updateGroupChat(
    group,
    current => ({
      ...current,
      hostedStatus: {
        state: 'queued',
        label: botsText().group.hostedQueued(sourceLabel(current.hostedConnectionId || ''))
      },
      continuityIssue: null
    }),
    { sync: false }
  )
  await dispatchHostedRoomOutbox()
  scheduleHostedRoomSync(0)

  return !failedHostedRoomCommand($hostedRoomOutbox.get(), failed.roomId)
}

export async function renameHostedGroupChat(group: string, name: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || !groupChatHostedGateway(room)) {
    return true
  }

  // A refresh may already be replaying the pre-rename server snapshot. Advance
  // the room fence before the request so that stale replay cannot restore the
  // old map key after the local rename completes or is queued for retry.
  beginHostedRoomMutation(room.roomId)

  const route = await hostedRouteForRoom(room)
  const connectionId = String(route?.connectionId || room.hostedConnectionId || '')

  if (!connectionId) {
    throw new Error(botsText().group.hostRouteMissing)
  }

  return enqueueHostedRoomCommand({
    commandId: crypto.randomUUID(),
    kind: 'rename',
    roomId: room.roomId,
    authorityId: groupChatHostedGateway(room),
    connectionId,
    payload: {
      name
    }
  })
}

export async function disbandHostedGroupChat(group: string) {
  const room = $groupChats.get()[group]

  if (!room?.roomId || !groupChatHostedGateway(room)) {
    return false
  }

  const route = await hostedRouteForRoom(room)

  if (!route) {
    throw new Error(
      botsText().group.hostedReconnectToDelete(
        sourceLabel(String(room.hostedConnectionId || '')) || botsText().group.thisHost
      )
    )
  }

  return enqueueHostedRoomCommand({
    commandId: crypto.randomUUID(),
    kind: 'disband',
    roomId: room.roomId,
    authorityId: groupChatHostedGateway(room),
    connectionId: route.connectionId,
    payload: {}
  })
}

export async function startHostedRoomRuntime(storage: PluginContext['storage'], hooks: HostedRoomRuntimeHooks = {}) {
  const lifecycleGeneration = ++hostedRoomLifecycleGeneration
  hostedRoomStorage = storage
  hostedRoomHooks = hooks
  hostedRoomSyncDisposed = false
  hostedRoomMutationGenerations.clear()
  hostedRoomLocallyDeleted.clear()
  hostedRoomObservations.invalidateAll()
  let persisted = createHostedRoomOutbox()

  try {
    persisted = await recoverHostedRoomOutbox(storage)
  } catch {
    /* an empty outbox is the safe fallback */
  }

  if (hostedRoomSyncDisposed || lifecycleGeneration !== hostedRoomLifecycleGeneration) {
    return
  }

  try {
    $hostedRoomOutbox.set(persisted)
  } catch {
    $hostedRoomOutbox.set(createHostedRoomOutbox())
  }

  await startHostedRoomCleanup(storage)

  if (hostedRoomSyncDisposed || lifecycleGeneration !== hostedRoomLifecycleGeneration) {
    return
  }

  await refreshHostedRooms().catch(() => undefined)
  await dispatchHostedRoomOutbox().catch(() => undefined)
  scheduleHostedRoomSync()
}

export function stopHostedRoomRuntime() {
  hostedRoomLifecycleGeneration += 1
  hostedRoomSyncDisposed = true
  stopHostedRoomCleanup()
  hostedRoomStorage = null
  hostedRoomHooks = {}
  hostedRoomPollCache.clear()
  hostedRoomPollGenerations.clear()
  hostedRoomMutationGenerations.clear()
  hostedRoomLocallyDeleted.clear()
  hostedRoomObservations.invalidateAll()
  hostedUnsupportedUntil.clear()

  if (hostedRoomSyncTimer) {
    clearTimeout(hostedRoomSyncTimer)
  }

  hostedRoomSyncTimer = null
}

/** Test-only lifecycle reset through the same public stop door. */
export function resetHostedRoomRuntimeForTests() {
  stopHostedRoomRuntime()
  hostedRoomObservations.retain([], [])
  hostedRoomSyncRunning = false
  hostedOutboxDispatchPromise = null
  resetHostedRoomOutboxLocksForTests()
  resetHostedRoomCleanupForTests()
  resetHostedRoomApprovalState()
  $hostedRoomCapabilities.set({})
  $hostedRoomOutbox.set(createHostedRoomOutbox())
}
