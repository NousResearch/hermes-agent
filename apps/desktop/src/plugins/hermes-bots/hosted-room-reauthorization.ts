/** Explicitly renew one peer route after its policy or capability catalog changed. */

import { host } from '@hermes/plugin-sdk'

import { $groupChats, groupChatHostedGateway } from './group-chat'
import {
  addHostedRoomCleanup,
  armHostedRoomCleanup,
  dispatchHostedRoomCleanup,
  releaseHostedRoomCleanup
} from './hosted-room-cleanup'
import {
  classifyHostedRoomCapability,
  isHostedRoomContinuityEligible,
  profileScopedRoomLinkEndpoint
} from './hosted-room-client'
import {
  $hostedRoomCapabilities,
  hostedRoomLifecycleIsCurrent,
  hostedRoomLifecycleToken,
  invalidateHostedRoomPoll,
  refreshHostedRooms,
  requestHostedConnection
} from './hosted-room-runtime'
import { requestForBot } from './routing'
import type { GroupMember, ProfileRoute } from './types'

const reconnectingPeers = new Map<string, { lifecycle: number; task: Promise<void> }>()

function record(value: unknown): null | Record<string, unknown> {
  return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : null
}

async function sha256(value: string) {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))

  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
}

async function routes() {
  if (typeof host.profileRoutes !== 'function') {
    return [] as ProfileRoute[]
  }

  const value = await host.profileRoutes()

  return (Array.isArray(value) ? value : []) as ProfileRoute[]
}

function matchingLocalMember(
  members: GroupMember[],
  serverMember: Record<string, unknown>,
  capabilities: Record<string, { authorityId?: null | string }>
) {
  const profile = String(serverMember.profile || serverMember.member_id || '')
  const handle = String(serverMember.handle || '')
  const target = record(serverMember.target)
  const targetAuthority = String(target?.installation_id || target?.peer_id || '')
  const candidates = members.filter(
    member =>
      String(member.targetProfile || member.name || '') === profile &&
      (!handle || String(member.handle || member.name || '') === handle)
  )

  if (targetAuthority) {
    const exact = candidates.find(member => {
      const connectionId = String(member.route?.connectionId || member.connectionId || '')

      return connectionId && capabilities[connectionId]?.authorityId === targetAuthority
    })

    if (exact) {
      return exact
    }
  }

  return candidates.length === 1 ? candidates[0] : null
}

async function reconnectPeer(group: string, memberId: string, lifecycle: number) {
  const assertCurrent = () => {
    if (!hostedRoomLifecycleIsCurrent(lifecycle)) {
      throw new Error('Group Chat connections changed. Try Reconnect again.')
    }
  }
  const room = $groupChats.get()[group]
  const roomId = String(room?.roomId || '')
  const homeAuthority = groupChatHostedGateway(room)
  const allRoutes = await routes()
  assertCurrent()
  const homeRoute = allRoutes.find(route => String(route.connectionId || '') === String(room?.hostedConnectionId || ''))

  if (!room || !roomId || !homeAuthority || !homeRoute) {
    throw new Error('Open the gateway that owns this Group Chat, then try again.')
  }

  const homeCapability = $hostedRoomCapabilities.get()[String(homeRoute.connectionId || '')]

  if (!homeCapability?.routeGrantFingerprint || homeCapability.authorityId !== homeAuthority) {
    throw new Error('Update the gateway that owns this Group Chat, then try again.')
  }

  const state = record(
    await requestHostedConnection<Record<string, unknown>>(homeRoute, 'groups.state', {
      room_id: roomId
    })
  )
  assertCurrent()
  const serverRoom = record(state?.room)
  const driver = record(state?.driver_status)
  const authorityId = String(serverRoom?.authority_gateway_id || '')
  const authorityEpoch = Number(serverRoom?.authority_epoch || 0)
  const serverMember = (Array.isArray(serverRoom?.members) ? serverRoom.members : [])
    .map(record)
    .find(member => String(member?.member_id || '') === memberId)
  const target = record(serverMember?.target)
  const targetAuthority = String(target?.installation_id || target?.peer_id || '')
  const localMember = serverMember
    ? matchingLocalMember(room.members || [], serverMember, $hostedRoomCapabilities.get())
    : null
  const peerConnectionId = String(localMember?.route?.connectionId || localMember?.connectionId || '')
  const profile = String(serverMember?.profile || serverMember?.member_id || 'default')
  const currentPeerRoute = (Array.isArray(driver?.peer_routes) ? driver.peer_routes : [])
    .map(record)
    .find(route => String(route?.member_id || '') === memberId)
  const expectedGrantSha256 = String(currentPeerRoute?.grant_sha256 || '')
  const peerRoute = allRoutes.find(
    route =>
      String(route.connectionId || '') === peerConnectionId &&
      String(route.targetProfile || route.profile || '') === profile
  )

  if (
    !serverMember ||
    target?.kind !== 'peer' ||
    !authorityId ||
    authorityId !== homeAuthority ||
    !Number.isSafeInteger(authorityEpoch) ||
    authorityEpoch < 1 ||
    !localMember ||
    !targetAuthority ||
    !/^[0-9a-f]{64}$/.test(expectedGrantSha256) ||
    !peerRoute
  ) {
    throw new Error('Reconnect the Bot gateway in Sessions, then try again.')
  }

  const peerCapability = classifyHostedRoomCapability(await requestHostedConnection(peerRoute, 'groups.capabilities'), {
    connectionId: peerConnectionId
  })
  assertCurrent()

  $hostedRoomCapabilities.set({
    ...$hostedRoomCapabilities.get(),
    [peerConnectionId]: peerCapability
  })

  if (
    !isHostedRoomContinuityEligible(peerCapability) ||
    !peerCapability.exactPeerGrantRevoke ||
    peerCapability.authorityId !== targetAuthority
  ) {
    throw new Error('That Bot gateway cannot reconnect to this Group Chat yet.')
  }

  const invitation = record(
    await requestForBot(localMember, 'groups.peer.invite', {
      room_id: roomId,
      home_install_id: authorityId,
      authority_gateway_id: authorityId,
      authority_epoch: authorityEpoch,
      member_id: memberId,
      profile
    })
  )
  const catalog = record(invitation?.catalog)
  const grant = String(invitation?.grant || '')
  const targetProfile = String(invitation?.target_profile || profile)
  const targetUrl = profileScopedRoomLinkEndpoint(peerCapability.roomLink?.endpoint, invitation?.target_profile)
  const setupId = `reconnect:${roomId}:${memberId}:${globalThis.crypto?.randomUUID?.() || Date.now()}`
  const operationId = `${setupId}:grant`

  const revokeFreshGrant = async () => {
    if (!grant) {
      return
    }

    await requestHostedConnection(peerRoute, 'groups.peer.revoke_exact', {
      grant,
      profile: targetProfile
    })
  }

  if (grant) {
    try {
      await addHostedRoomCleanup({
        operationId,
        setupId,
        kind: 'peer-revoke-exact',
        connectionId: peerConnectionId,
        profile: targetProfile,
        grant,
        roomId: null,
        cancelId: null,
        homeConnectionId: null,
        homeProfile: null,
        memberId: null,
        targetUrl: null,
        catalog: null
      })
    } catch (error) {
      await revokeFreshGrant()
      throw error
    }
  }

  const abandonIfStale = async () => {
    if (hostedRoomLifecycleIsCurrent(lifecycle)) {
      return
    }

    await armHostedRoomCleanup(setupId)
    await dispatchHostedRoomCleanup()
    assertCurrent()
  }

  await abandonIfStale()

  let grantSha256 = ''

  if (grant) {
    try {
      grantSha256 = await sha256(grant)
      await abandonIfStale()
    } catch (error) {
      await armHostedRoomCleanup(setupId)
      await dispatchHostedRoomCleanup()
      throw error
    }
  }

  if (
    !grant ||
    !catalog?.installation_id ||
    String(catalog.installation_id) !== targetAuthority ||
    !catalog.catalog_digest ||
    !targetProfile ||
    targetProfile !== profile ||
    !targetUrl
  ) {
    if (grant) {
      await armHostedRoomCleanup(setupId)
      await dispatchHostedRoomCleanup()
    }
    throw new Error('That Bot gateway could not prepare a secure connection.')
  }

  try {
    await addHostedRoomCleanup({
      operationId,
      setupId,
      kind: 'peer-reconnect',
      connectionId: peerConnectionId,
      profile: targetProfile,
      grant,
      grantSha256,
      expectedGrantSha256,
      roomId,
      cancelId: null,
      homeConnectionId: String(homeRoute.connectionId || ''),
      homeProfile: String(homeRoute.targetProfile || homeRoute.profile || 'default'),
      memberId,
      targetUrl,
      catalog
    })
  } catch (error) {
    await armHostedRoomCleanup(setupId)
    await dispatchHostedRoomCleanup()
    throw error
  }

  await abandonIfStale()

  try {
    await requestHostedConnection(homeRoute, 'groups.peer.register', {
      room_id: roomId,
      member_id: memberId,
      target_url: targetUrl,
      target_profile: targetProfile,
      grant,
      catalog,
      expected_grant_sha256: expectedGrantSha256
    })
    await abandonIfStale()
  } catch (error) {
    await armHostedRoomCleanup(setupId)
    await dispatchHostedRoomCleanup()
    throw error
  }

  await abandonIfStale()
  await releaseHostedRoomCleanup(setupId).catch(() => undefined)
  assertCurrent()
  invalidateHostedRoomPoll(roomId)
  await refreshHostedRooms().catch(() => undefined)
}

export function reconnectHostedGroupChatPeer(group: string, memberId: string) {
  const key = `${group}:${memberId}`
  const lifecycle = hostedRoomLifecycleToken()
  const existing = reconnectingPeers.get(key)

  if (existing?.lifecycle === lifecycle) {
    return existing.task
  }

  const task = reconnectPeer(group, memberId, lifecycle).finally(() => {
    if (reconnectingPeers.get(key)?.task === task) {
      reconnectingPeers.delete(key)
    }
  })

  reconnectingPeers.set(key, { lifecycle, task })

  return task
}
