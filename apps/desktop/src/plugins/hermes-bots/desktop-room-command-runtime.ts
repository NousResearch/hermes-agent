import { gatewayActivationEpoch, host } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'

import { groupExecutionMode } from './canonical-group-capabilities'
import { $botMeta, $lastRoster, cachedUnionRoster } from './data'
import {
  createDesktopRoomConsumerId,
  desktopRoomDescriptors,
  desktopRoomIdentity,
  refreshDesktopRoomPresence,
  runDesktopRoomCommandCycle
} from './desktop-room-command-client'
import type { DesktopRoomCommand, DesktopRoomDescriptor } from './desktop-room-command-client'
import {
  $groupChats,
  backfillClassicGroupAuthorities,
  GROUP_CHAT_MAX_MEMBERS,
  groupChatHostedGateway,
  persistGroupChatRoomsRequired,
  scheduleGroupChatServerSync,
  updateGroupChat
} from './group-chat'
import { beginGroupCommandFence, groupCommandFenceLive, releaseGroupCommandFence } from './group-command-fence'
import type { GroupCommandFence } from './group-command-fence'
import { desktopCommandResult, settleDesktopCommand } from './group-command-receipts'
import type { DesktopCommandResult } from './group-command-receipts'
import { classicAuthorityClaim } from './group-desktop-authority'
import { groupChatBotsFromDescriptors, groupChatMemberBots } from './group-membership'
import { cancelGroupThreadForLeaseLoss, sendToGroupChat, stopGroupThread } from './group-rounds'
import { groupChatContinuityReady } from './hosted-room-runtime'
import type { GroupMember, ProfileRoute } from './types'

const DESKTOP_ROOM_COMMAND_INTERVAL_MS = 60_000
const DESKTOP_ROOM_PRESENCE_INTERVAL_MS = 30_000
const DESKTOP_ROOM_COMMAND_PUSH_DEBOUNCE_MS = 250
const DESKTOP_ROOM_DRIVE_ATTEMPTS_PER_CLAIM = 2
const DESKTOP_ROOM_MAX_CLAIMS = 2
const DESKTOP_ROOM_DRIVE_RETRY_DELAY_MS = 250

let desktopRoomCommandConsumerId = ''
let desktopRoomCommandTimer: null | ReturnType<typeof setInterval> = null
let desktopRoomPresenceTimer: null | ReturnType<typeof setInterval> = null
let desktopRoomCommandPushTimer: null | ReturnType<typeof setTimeout> = null
let desktopRoomCommandPushUnsub: null | (() => void) = null
let desktopRoomCommandRunning = false
let desktopRoomStopRunning = false
let desktopRoomCommandDisposed = true
let desktopRoomCommandRerun = false
let desktopRoomStopRerun = false
let desktopRoomPersistenceBlocked = false

const desktopRoomCommandPendingConnections = new Set<string>()
const desktopRoomStopPendingConnections = new Set<string>()
const desktopRoomCommandRetentions = new Map<string, () => void>()

const activeDesktopRoomCommands = new Map<
  string,
  { commandId: string; controller: AbortController; threadId: null | string }
>()

interface LegacyCommandSource {
  current: () => boolean
  check: () => Promise<boolean>
}

// Evidence belongs to one captured route in one discovery/activation, never a
// connection-name cache. A later wake must classify again, including old Serve.
const desktopRoomCommandSources = new WeakMap<ProfileRoute, LegacyCommandSource>()

function desktopCommandSourceCurrent() {
  const epoch = gatewayActivationEpoch()
  const connectionId = host.state.connectionId.get()
  const profile = host.state.profile.get()
  const open = host.state.gateway.get() === 'open'

  return () => open && gatewayActivationEpoch() === epoch &&
    host.state.connectionId.get() === connectionId && host.state.profile.get() === profile &&
    host.state.gateway.get() === 'open'
}

function legacyCommandSource(route: ProfileRoute, sourceCurrent = desktopCommandSourceCurrent()): LegacyCommandSource {
  const captured = Object.freeze({ ...route })
  let legacy = false
  let revoked = false

  const sourceLive = () => sourceCurrent() &&
    route.connectionId === captured.connectionId && route.profile === captured.profile &&
    route.targetProfile === captured.targetProfile && route.mode === captured.mode

  const current = () => !revoked && legacy && sourceLive()

  return {
    current,
    check: async () => {
      if (revoked || !sourceLive() || !captured.connectionId || !captured.profile || !captured.targetProfile) {
        revoked = true

        return false
      }

      try {
        const capability = await host.requestProfile(captured, 'groups.capabilities', { profile: captured.profile })

        // Revocation is sticky for this flow: a late successful read cannot
        // rescue a failed/canonical read or revive a switched-away activation.
        if (!sourceLive() || groupExecutionMode(capability) !== 'legacy') { revoked = true }
        legacy = !revoked
      } catch {
        revoked = true
      }

      return current()
    }
  }
}

function assertLegacyCommandSource(source: LegacyCommandSource) {
  if (!source.current()) {
    throw retryableDesktopRoomCommand('Legacy Desktop Group Chat control is unavailable for this source.')
  }
}

function desktopRoomEntry(roomId: string, descriptors: DesktopRoomDescriptor[]) {
  const descriptor = descriptors.find(room => room.roomId === roomId)

  if (!descriptor) {
    return null
  }

  const entries = Object.entries($groupChats.get()).filter(([name, room]) => desktopRoomIdentity(name, room) === roomId)

  if (entries.length !== 1 || groupChatHostedGateway(entries[0][1]) || entries[0][1].tombstone) {
    return null
  }

  const entry = entries[0]
  const claim = classicAuthorityClaim(entry[1])

  return claim && claim.authorityToken === descriptor.authorityToken && claim.authorityHash === descriptor.authorityHash
    ? entry
    : null
}

function desktopCommandEligibleRooms(forSend = false) {
  return Object.fromEntries(
    Object.entries($groupChats.get()).filter(([, room]) => {
      if (groupChatHostedGateway(room) || room?.tombstone) {
        return false
      }

      if (!classicAuthorityClaim(room)) {
        return false
      }

      if (forSend && !groupChatContinuityReady(room)) {
        return false
      }

      return Array.isArray(room?.members) && room.members.length > 0
    })
  )
}

function retryableDesktopRoomCommand(message: string) {
  return Object.assign(new Error(message), {
    retryable: true
  })
}

async function persistDesktopCommandState() {
  const owner = desktopRoomCommandConsumerId

  try {
    await persistGroupChatRoomsRequired()
  } catch {
    if (owner === desktopRoomCommandConsumerId) {
      desktopRoomPersistenceBlocked = true
    }

    throw retryableDesktopRoomCommand('Group Chat changes could not be saved. Check available storage and try again.')
  }
}

function currentDesktopRuntime(owner: string) {
  return !desktopRoomCommandDisposed && owner === desktopRoomCommandConsumerId
}

async function requestDesktopCommandGateway(route: ProfileRoute, method: string, params: Record<string, unknown>) {
  if (!currentDesktopRuntime(String(params.consumer_id))) {
    throw retryableDesktopRoomCommand('Desktop Group Chat control has stopped.')
  }

  const source = desktopRoomCommandSources.get(route)

  if (!source || !await source.check()) {
    throw retryableDesktopRoomCommand('Legacy Desktop Group Chat control is unavailable for this source.')
  }

  if (method === 'groups.desktop.claim' || method === 'groups.desktop.presence') {
    const forSend = method === 'groups.desktop.claim' && Array.isArray(params.actions) && params.actions.includes('send')
    const expected = desktopRoomDescriptors(desktopCommandEligibleRooms(forSend))
    await persistDesktopCommandState()

    if (!currentDesktopRuntime(String(params.consumer_id)) || desktopRoomPersistenceBlocked) {
      throw retryableDesktopRoomCommand('Desktop Group Chat control has stopped.')
    }

    // Persistence may span an owner change even when the visible route stays
    // put. Read capabilities again before publishing authority or claiming work.
    await source.check()
    assertLegacyCommandSource(source)

    // An async storage read must not authorize a replaced/conflicted room.
    const current = desktopRoomDescriptors(desktopCommandEligibleRooms(forSend))
    const authorities = Array.isArray(params.room_authorities) ? params.room_authorities : []

    const valid = authorities.filter(raw => {
      if (!raw || typeof raw !== 'object') {
        return false
      }

      const before = expected.find(room => room.roomId === raw.room_id)
      const after = current.find(room => room.roomId === raw.room_id)

      return (
        before &&
        after &&
        before.authorityToken === raw.authority_token &&
        after.authorityToken === raw.authority_token &&
        before.authorityHash === after.authorityHash &&
        desktopRoomEntry(raw.room_id, current)
      )
    })

    if (!valid.length) {
      return method === 'groups.desktop.claim' ? { commands: [] } : { room_ids: [] }
    }

    params = { ...params, room_authorities: valid }
  }

  assertLegacyCommandSource(source)

  if (!currentDesktopRuntime(String(params.consumer_id))) {
    throw retryableDesktopRoomCommand('Desktop Group Chat control has stopped.')
  }

  const result = await host.requestProfile(route, method, params)
  assertLegacyCommandSource(source)

  return result
}

function frozenRecipients(value: unknown): GroupMember[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > GROUP_CHAT_MAX_MEMBERS) {
    throw new Error('The Group Chat command has an invalid Bot list.')
  }

  return value.map(raw => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new Error('The Group Chat command has an invalid Bot list.')
    }

    const member = raw as GroupMember
    const name = typeof member.name === 'string' ? member.name.trim() : ''

    if (!name || name.length > 128) {
      throw new Error('The Group Chat command has an invalid Bot list.')
    }

    return member
  })
}

async function waitForDesktopRoomCommandSettlement(
  group: string,
  commandId: string,
  signal: AbortSignal | null,
  fence: GroupCommandFence
) {
  while (true) {
    if (fence.persistenceFailed) {
      desktopRoomPersistenceBlocked = true
      throw retryableDesktopRoomCommand('Group Chat changes could not be saved. Check available storage and try again.')
    }

    if (signal?.aborted || !groupCommandFenceLive(fence)) {
      throw retryableDesktopRoomCommand('The command moved to another Desktop.')
    }

    const room = $groupChats.get()[group]

    if (!room) {
      throw new Error('This Group Chat is no longer available.')
    }

    if (room.desktopCommandSettled?.[commandId]) {
      return true
    }

    if (!room.running) {
      return false
    }

    await new Promise(resolve => setTimeout(resolve, 250))
  }
}

interface CommandExecutionContext {
  consumerId: string
  leaseValid?: () => boolean
  request: (method: string, params: Record<string, unknown>) => Promise<unknown>
  route: ProfileRoute
  signal: AbortSignal | null
}

export async function executeDesktopRoomCommand(
  command: DesktopRoomCommand,
  descriptors: DesktopRoomDescriptor[],
  { signal, request, consumerId, route, leaseValid = () => true }: CommandExecutionContext
) {
  const runtimeOwner = desktopRoomCommandDisposed ? null : desktopRoomCommandConsumerId
  const source = desktopRoomCommandSources.get(route) ?? legacyCommandSource(route)

  const assertLiveLease = () => {
    if (
      signal?.aborted ||
      !leaseValid() ||
      (runtimeOwner && (desktopRoomCommandDisposed || runtimeOwner !== desktopRoomCommandConsumerId))
    ) {
      throw retryableDesktopRoomCommand('The command moved to another Desktop.')
    }
  }

  assertLiveLease()

  await persistDesktopCommandState()
  assertLiveLease()

  const roomId = String(command.room_id || '')
  const entry = desktopRoomEntry(roomId, descriptors)

  if (!entry) {
    throw new Error('This Group Chat is no longer available on this Desktop.')
  }

  const [group, room] = entry
  const commandId = String(command.command_id || '')
  const roomValid = () => desktopRoomEntry(roomId, descriptors)?.[0] === group

  const assertCurrentRoom = () => {
    assertLiveLease()
    assertLegacyCommandSource(source)

    if (!roomValid()) {
      throw new Error('This Group Chat is no longer available on this Desktop.')
    }
  }

  if (command.action === 'send' || command.action === 'stop') {
    // The required snapshot readback above verifies this exact retained result.
    const retained = desktopCommandResult(group, room, commandId, command.action)

    if (retained) {
      return retained
    }
  }

  await source.check()
  assertCurrentRoom()

  const finishStop = async (result: DesktopCommandResult) => {
    assertCurrentRoom()
    updateGroupChat(group, current => ({
      ...current,
      desktopCommandSettled: settleDesktopCommand(group, current, commandId, 'stop', result)
    }))
    await persistDesktopCommandState()
    assertCurrentRoom()

    return desktopCommandResult(group, $groupChats.get()[group], commandId, 'stop')!
  }

  const cached = cachedUnionRoster()
  const roster = Array.isArray(cached?.profiles) ? cached.profiles : $lastRoster.get()
  let members = groupChatMemberBots(group, roster, $botMeta.get())

  if (command.action === 'send') {
    const payload = command.payload || {}

    const recipients = frozenRecipients(payload.recipients)
    members = groupChatBotsFromDescriptors(recipients, roster)

    if (!members.length || members.length !== recipients.length) {
      throw retryableDesktopRoomCommand('Waiting for a Bot in this Group Chat to reconnect.')
    }

    assertLiveLease()

    if (typeof payload.message !== 'string') {
      throw new Error('The Group Chat message is invalid.')
    }

    const message = payload.message.trim()
    const actor = payload.actor_display_name

    if (actor !== undefined && typeof actor !== 'string') {
      throw new Error('The Group Chat sender is invalid.')
    }

    if (!message) {
      throw new Error('The Group Chat message is empty.')
    }

    const commandId = String(command.command_id || '')
    const localAbort = new AbortController()
    const fence = beginGroupCommandFence(roomId, commandId, () => leaseValid() && source.current(), roomValid)

    const cancelTurn = async () => {
      if (localAbort.signal.reason !== 'room-stop' && await source.check() && source.current()) {
        cancelGroupThreadForLeaseLoss(group, members, fence)
      }
    }

    localAbort.signal.addEventListener('abort', cancelTurn, { once: true })
    const onLeaseAbort = () => localAbort.abort('lease-lost')

    signal?.addEventListener('abort', onLeaseAbort, {
      once: true
    })

    if (signal?.aborted) {
      localAbort.abort('lease-lost')
    }

    activeDesktopRoomCommands.set(roomId, {
      commandId,
      controller: localAbort,
      threadId: null
    })

    try {
      const rawClaimAttempt = Number(command.attempts)
      const claimAttempt = Number.isSafeInteger(rawClaimAttempt) && rawClaimAttempt > 0 ? rawClaimAttempt : 1

      for (let driveAttempt = 1; driveAttempt <= DESKTOP_ROOM_DRIVE_ATTEMPTS_PER_CLAIM; driveAttempt += 1) {
        await source.check()
        assertCurrentRoom()

        if (!groupChatContinuityReady($groupChats.get()[group])) {
          throw retryableDesktopRoomCommand('Waiting for this Group Chat to finish reconnecting.')
        }

        if (localAbort.signal.aborted) {
          throw retryableDesktopRoomCommand('The command moved to another Desktop.')
        }

        const thread = await Promise.resolve(
          sendToGroupChat(group, members, message, null, undefined, {
            entryId: commandId,
            commandFence: fence,
            userName: String(actor || 'Messaging')
          })
        )

        if (!thread) {
          throw new Error('The Group Chat could not accept this message.')
        }

        const active = activeDesktopRoomCommands.get(roomId)

        if (active?.commandId === commandId) {
          active.threadId = thread
        }

        if (await waitForDesktopRoomCommandSettlement(group, commandId, localAbort.signal, fence)) {
          await persistDesktopCommandState()
          assertCurrentRoom()

          return desktopCommandResult(group, $groupChats.get()[group], commandId, 'send')!
        }

        if (driveAttempt < DESKTOP_ROOM_DRIVE_ATTEMPTS_PER_CLAIM) {
          await new Promise(resolve => setTimeout(resolve, DESKTOP_ROOM_DRIVE_RETRY_DELAY_MS))
        }
      }

      if (claimAttempt >= DESKTOP_ROOM_MAX_CLAIMS) {
        throw new Error('The Group Chat could not finish this command after repeated attempts. Retry it explicitly.')
      }

      throw retryableDesktopRoomCommand('The Group Chat command will retry after a short delay.')
    } catch (error) {
      if (!roomValid()) {
        throw new Error('This Group Chat is no longer available on this Desktop.')
      }

      if (localAbort.signal.aborted || !groupCommandFenceLive(fence)) {
        if (localAbort.signal.reason === 'room-stop') {
          await persistDesktopCommandState()

          if (!$groupChats.get()[group]?.desktopCommandSettled?.[commandId]) {
            throw retryableDesktopRoomCommand('The Stop is still being applied to this Group Chat.')
          }

          return desktopCommandResult(group, $groupChats.get()[group], commandId, 'send')!
        }

        await cancelTurn()
        throw retryableDesktopRoomCommand('The command moved to another Desktop.')
      }

      throw error
    } finally {
      releaseGroupCommandFence(fence)
      localAbort.signal.removeEventListener('abort', cancelTurn)
      signal?.removeEventListener('abort', onLeaseAbort)

      if (activeDesktopRoomCommands.get(roomId)?.controller === localAbort) {
        activeDesktopRoomCommands.delete(roomId)
      }
    }
  }

  if (command.action === 'stop') {
    const commandId = String(command.command_id || '')
    const payload = command.payload || {}
    const targetCommandId = String(payload.target_command_id || '')
    const targetThreadId = String(payload.target_thread_id || '')
    const targetMessageId = String(payload.target_message_id || '')

    for (const field of ['target_command_id', 'target_thread_id', 'target_message_id']) {
      if (payload[field] !== undefined && typeof payload[field] !== 'string') {
        throw new Error('The Group Chat Stop target is invalid.')
      }
    }

    if (!targetCommandId && !targetMessageId) {
      throw new Error('The Group Chat Stop target is missing.')
    }

    let active = activeDesktopRoomCommands.get(roomId)

    if (targetCommandId) {
      if (
        command.target_command_state === 'failed' &&
        command.target_result_code === 'superseded_by_stop' &&
        active?.commandId !== targetCommandId
      ) {
        return finishStop({
          room_name: group,
          stale: true,
          stopped: true
        })
      }

      if (
        ['completed', 'failed'].includes(String(command.target_command_state || '')) &&
        active?.commandId !== targetCommandId
      ) {
        return finishStop({
          room_name: group,
          stale: true,
          stopped: false
        })
      }

      for (let attempt = 0; attempt < 40; attempt += 1) {
        assertCurrentRoom()

        if ($groupChats.get()[group]?.desktopCommandSettled?.[targetCommandId]) {
          return finishStop({
            room_name: group,
            stale: true,
            stopped: false
          })
        }

        active = activeDesktopRoomCommands.get(roomId)

        if (active?.commandId === targetCommandId) {
          break
        }

        if (active && active.commandId !== targetCommandId) {
          return finishStop({
            room_name: group,
            stale: true,
            stopped: false
          })
        }

        if (signal?.aborted) {
          throw retryableDesktopRoomCommand('The command moved to another Desktop.')
        }

        await new Promise(resolve => setTimeout(resolve, 50))
      }

      if (active?.commandId !== targetCommandId) {
        throw retryableDesktopRoomCommand('Waiting for the earlier Group Chat message to start.')
      }
    }

    const latestUser = [...room.log].reverse().find(item => item?.thread && item?.from?.kind === 'user')
    const latestThread = latestUser?.thread || null
    const latestMessageId = String(latestUser?.eventId || latestUser?.id || '')
    const stopThread = targetCommandId ? active?.threadId || targetThreadId || latestThread : targetThreadId

    if (!targetCommandId && targetThreadId && latestThread && targetThreadId !== latestThread) {
      return finishStop({
        room_name: group,
        stale: true,
        stopped: false
      })
    }

    if (!targetCommandId && (!targetMessageId || latestMessageId !== targetMessageId)) {
      return finishStop({
        room_name: group,
        stale: true,
        stopped: false
      })
    }

    if (!stopThread) {
      return finishStop({
        room_name: group,
        stale: true,
        stopped: false
      })
    }

    await source.check()
    assertCurrentRoom()

    if (targetCommandId) {
      active?.controller.abort('room-stop')
    }

    await stopGroupThread(group, stopThread, members)
    assertCurrentRoom()

    if (targetCommandId && active?.commandId) {
      updateGroupChat(group, current => ({
        ...current,
        desktopCommandSettled: settleDesktopCommand(group, current, targetCommandId, 'send', {
          room_name: group,
          stopped: true
        })
      }))
    }

    return finishStop({ room_name: group, stopped: true })
  }

  throw new Error('Unsupported Group Chat command.')
}

async function desktopRoomCommandConnections() {
  const sourceCurrent = desktopCommandSourceCurrent()
  const owner = desktopRoomCommandConsumerId
  const byConnection = new Map<string, ProfileRoute>()

  if (typeof host.profileRoutes === 'function') {
    try {
      const routes = await host.profileRoutes()

      for (const route of Array.isArray(routes) ? routes : []) {
        const profile = String(route?.targetProfile || route?.profile || '')
        const connectionId = String(route?.connectionId || '')

        if (profile === 'default' && !byConnection.has(connectionId)) {
          byConnection.set(connectionId, route as ProfileRoute)
        }
      }
    } catch {
      /* the active route below remains a compatibility fallback */
    }
  }

  const active = String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || '')

  if (!byConnection.has(active)) {
    byConnection.set(active, {
      connectionId: active,
      mode: 'remote',
      profile: 'default',
      targetProfile: 'default'
    })
  }

  if (!sourceCurrent() || !currentDesktopRuntime(owner)) { return [] }

  const connections = await Promise.all([...byConnection.entries()].map(async ([id, candidate]) => {
    const route = Object.freeze({ ...candidate })
    const source = legacyCommandSource(route, sourceCurrent)
    desktopRoomCommandSources.set(route, source)
    await source.check()

    return { id, route, source }
  }))

  return connections.filter(connection => currentDesktopRuntime(owner) && connection.source.current())
}

function syncDesktopRoomCommandRetention(connections: Array<{ id: string; route: ProfileRoute }>) {
  if (typeof host.retainProfileSocket !== 'function') {
    return
  }

  const live = new Set(connections.map(connection => connection.id))

  for (const [id, release] of [...desktopRoomCommandRetentions]) {
    if (!live.has(id)) {
      desktopRoomCommandRetentions.delete(id)

      try {
        release()
      } catch {
        /* teardown stays best-effort */
      }
    }
  }

  if (desktopRoomCommandDisposed) {
    return
  }

  for (const connection of connections) {
    if (!desktopRoomCommandRetentions.has(connection.id)) {
      desktopRoomCommandRetentions.set(connection.id, host.retainProfileSocket(connection.route))
    }
  }
}

function releaseDesktopRoomCommandRetention() {
  for (const release of desktopRoomCommandRetentions.values()) {
    try {
      release()
    } catch {
      /* teardown stays best-effort */
    }
  }

  desktopRoomCommandRetentions.clear()
}

function scheduleDesktopRoomCommandPump(connectionId: null | string = null) {
  if (desktopRoomCommandDisposed || typeof setTimeout !== 'function') {
    return
  }

  const key = connectionId === null ? '*' : String(connectionId)

  desktopRoomCommandPendingConnections.add(key)
  desktopRoomStopPendingConnections.add(key)

  if (desktopRoomCommandPushTimer !== null) {
    return
  }

  desktopRoomCommandPushTimer = setTimeout(() => {
    desktopRoomCommandPushTimer = null

    const pending = new Set(desktopRoomCommandPendingConnections)
    const stopPending = new Set(desktopRoomStopPendingConnections)

    desktopRoomCommandPendingConnections.clear()
    desktopRoomStopPendingConnections.clear()
    void runDesktopRoomCommandPump(pending.has('*') ? null : pending)
    void runDesktopRoomStopPump(stopPending.has('*') ? null : stopPending)
  }, DESKTOP_ROOM_COMMAND_PUSH_DEBOUNCE_MS)
}

async function runDesktopRoomCommandPump(targetConnectionIds: null | Set<string> = null) {
  if (desktopRoomCommandDisposed || desktopRoomPersistenceBlocked) {
    return
  }

  if (desktopRoomCommandRunning) {
    desktopRoomCommandRerun = true

    if (targetConnectionIds === null) {
      desktopRoomCommandPendingConnections.add('*')
    } else {
      targetConnectionIds.forEach(id => desktopRoomCommandPendingConnections.add(String(id)))
    }

    return
  }

  desktopRoomCommandRunning = true
  const owner = desktopRoomCommandConsumerId

  try {
    const connections = await desktopRoomCommandConnections()

    if (!currentDesktopRuntime(owner)) {
      return
    }

    const rooms = desktopCommandEligibleRooms(true)

    if (!Object.keys(rooms).length) {
      syncDesktopRoomCommandRetention([])

      return
    }

    syncDesktopRoomCommandRetention(connections)

    const selected =
      targetConnectionIds === null
        ? connections
        : connections.filter(connection => targetConnectionIds.has(connection.id))

    await runDesktopRoomCommandCycle({
      routes: selected.map(connection => connection.route),
      consumerId: owner,
      rooms,
      request: requestDesktopCommandGateway,
      execute: executeDesktopRoomCommand,
      actions: ['send'],
      shouldContinue: () => currentDesktopRuntime(owner)
    })
  } catch {
    // A reconnect or older backend leaves durable commands pending.
  } finally {
    if (currentDesktopRuntime(owner)) {
      desktopRoomCommandRunning = false
    }

    if (desktopRoomCommandRerun && currentDesktopRuntime(owner)) {
      desktopRoomCommandRerun = false

      const pending = [...desktopRoomCommandPendingConnections]

      desktopRoomCommandPendingConnections.clear()

      if (!pending.length || pending.includes('*')) {
        scheduleDesktopRoomCommandPump()
      } else {
        pending.forEach(connectionId => scheduleDesktopRoomCommandPump(connectionId))
      }
    }
  }
}

async function runDesktopRoomStopPump(targetConnectionIds: null | Set<string> = null) {
  if (desktopRoomCommandDisposed || desktopRoomPersistenceBlocked) {
    return
  }

  if (desktopRoomStopRunning) {
    desktopRoomStopRerun = true

    if (targetConnectionIds === null) {
      desktopRoomStopPendingConnections.add('*')
    } else {
      targetConnectionIds.forEach(id => desktopRoomStopPendingConnections.add(String(id)))
    }

    return
  }

  desktopRoomStopRunning = true
  const owner = desktopRoomCommandConsumerId

  try {
    const connections = await desktopRoomCommandConnections()

    if (!currentDesktopRuntime(owner)) {
      return
    }

    const rooms = desktopCommandEligibleRooms()

    const selected =
      targetConnectionIds === null
        ? connections
        : connections.filter(connection => targetConnectionIds.has(connection.id))

    if (!Object.keys(rooms).length) {
      return
    }

    await runDesktopRoomCommandCycle({
      routes: selected.map(connection => connection.route),
      consumerId: owner,
      rooms,
      request: requestDesktopCommandGateway,
      execute: executeDesktopRoomCommand,
      actions: ['stop'],
      shouldContinue: () => currentDesktopRuntime(owner)
    })
  } catch {
    // A reconnect or older backend leaves durable Stops pending.
  } finally {
    if (currentDesktopRuntime(owner)) {
      desktopRoomStopRunning = false
    }

    if (desktopRoomStopRerun && currentDesktopRuntime(owner)) {
      desktopRoomStopRerun = false

      const pending = [...desktopRoomStopPendingConnections]

      desktopRoomStopPendingConnections.clear()
      void runDesktopRoomStopPump(!pending.length || pending.includes('*') ? null : new Set(pending))
    }
  }
}

async function refreshDesktopRoomCommandPresence() {
  if (desktopRoomCommandDisposed || desktopRoomPersistenceBlocked) {
    return
  }

  const owner = desktopRoomCommandConsumerId
  const connections = await desktopRoomCommandConnections()

  if (!currentDesktopRuntime(owner)) {
    return
  }

  const rooms = desktopCommandEligibleRooms()

  if (!Object.keys(rooms).length) {
    return
  }

  await refreshDesktopRoomPresence({
    routes: connections.map(connection => connection.route),
    consumerId: owner,
    rooms,
    request: requestDesktopCommandGateway
  })
}

export async function startDesktopRoomCommandRuntime(storage?: PluginContext['storage']) {
  if (!desktopRoomCommandDisposed || desktopRoomCommandTimer !== null || desktopRoomPresenceTimer !== null) {
    if (desktopRoomPersistenceBlocked) {
      const owner = desktopRoomCommandConsumerId
      await persistGroupChatRoomsRequired($groupChats.get(), storage)

      if (!desktopRoomCommandDisposed && owner === desktopRoomCommandConsumerId) {
        desktopRoomPersistenceBlocked = false
        scheduleDesktopRoomCommandPump()
        void refreshDesktopRoomCommandPresence()
      }
    }

    if (!desktopRoomCommandDisposed && !desktopRoomPersistenceBlocked) {
      scheduleDesktopRoomCommandPump()
    }

    return
  }

  const consumerId = createDesktopRoomConsumerId()

  desktopRoomCommandConsumerId = consumerId
  desktopRoomCommandDisposed = false

  backfillClassicGroupAuthorities()

  try {
    await persistGroupChatRoomsRequired($groupChats.get(), storage)
  } catch (error) {
    if (desktopRoomCommandConsumerId === consumerId) {
      desktopRoomCommandConsumerId = ''
      desktopRoomCommandDisposed = true
    }

    throw error
  }

  if (desktopRoomCommandDisposed || desktopRoomCommandConsumerId !== consumerId) {
    return
  }

  desktopRoomPersistenceBlocked = false
  scheduleGroupChatServerSync($groupChats.get())

  if (typeof setInterval !== 'function' || desktopRoomCommandTimer !== null) {
    return
  }

  void runDesktopRoomCommandPump()
  void runDesktopRoomStopPump()
  void refreshDesktopRoomCommandPresence()
  desktopRoomCommandTimer = setInterval(() => {
    void runDesktopRoomCommandPump()
    void runDesktopRoomStopPump()
  }, DESKTOP_ROOM_COMMAND_INTERVAL_MS)
  desktopRoomPresenceTimer = setInterval(() => {
    void refreshDesktopRoomCommandPresence()
  }, DESKTOP_ROOM_PRESENCE_INTERVAL_MS)

  if (desktopRoomCommandPushUnsub === null && typeof host.onEvent === 'function') {
    desktopRoomCommandPushUnsub = host.onEvent('desktop_rooms.commands.pending', event =>
      scheduleDesktopRoomCommandPump(event?.connectionId ?? null)
    )
  }
}

export function stopDesktopRoomCommandRuntime() {
  desktopRoomCommandDisposed = true
  desktopRoomCommandRunning = false
  desktopRoomStopRunning = false

  for (const active of activeDesktopRoomCommands.values()) {
    active.controller.abort('runtime-stopped')
  }

  activeDesktopRoomCommands.clear()
  desktopRoomCommandRerun = false
  desktopRoomStopRerun = false
  desktopRoomCommandPendingConnections.clear()
  desktopRoomStopPendingConnections.clear()
  releaseDesktopRoomCommandRetention()

  if (desktopRoomCommandTimer !== null) {
    clearInterval(desktopRoomCommandTimer)
    desktopRoomCommandTimer = null
  }

  if (desktopRoomPresenceTimer !== null) {
    clearInterval(desktopRoomPresenceTimer)
    desktopRoomPresenceTimer = null
  }

  if (desktopRoomCommandPushTimer !== null) {
    clearTimeout(desktopRoomCommandPushTimer)
    desktopRoomCommandPushTimer = null
  }

  if (desktopRoomCommandPushUnsub !== null) {
    try {
      desktopRoomCommandPushUnsub()
    } catch {
      /* older host disposer */
    }

    desktopRoomCommandPushUnsub = null
  }

  desktopRoomCommandConsumerId = ''
}
