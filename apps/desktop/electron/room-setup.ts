import { randomUUID } from 'node:crypto'
import { isDeepStrictEqual } from 'node:util'

import { RoomSetupError } from './room-setup-store'
import type { roomSetupStore, SetupRecord, SetupRoute } from './room-setup-store'

interface Client { request(method: string, params?: Record<string, unknown>): Promise<any>; close(): void }
interface Member { member_id: string; handle: string; profile: string; display_name?: string; connectionId: string }
export interface RoomSetupInput { home: SetupRoute; name: string; members: Member[] }
const routeKey = (route: SetupRoute) => JSON.stringify([route.connectionId, route.profile])
const validRoute = (route: SetupRoute) => route && typeof route.connectionId === 'string' && route.connectionId.length > 0 &&
  route.connectionId.length <= 256 && typeof route.profile === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(route.profile)

/** Setup only. The gateway remains the sole owner of execution and history. */
export function roomSetupCoordinator(options: {
  store: ReturnType<typeof roomSetupStore>; connect: (route: SetupRoute) => Promise<Client>
}) {
  let serial = Promise.resolve()
  const exclusive = <T>(work: () => Promise<T>) => {
    const result = serial.then(work, work)
    serial = result.then(() => undefined, () => undefined)
    return result
  }
  const open = async (route: SetupRoute, expected?: string) => {
    const client = await options.connect(route)
    try {
      const capability = await client.request('groups.capabilities')
      if (!capability?.driver || capability.persistent_process !== true || !capability.methods?.includes('groups.discard') ||
          typeof capability.authority_gateway_id !== 'string' || !capability.authority_gateway_id ||
          (expected && capability.authority_gateway_id !== expected)) {throw new RoomSetupError('original_gateway_required')}
      return { client, capability }
    } catch (error) {client.close(); throw error}
  }
  const recover = async (live = new Map<string, Client>(), memory = new Map<string, SetupRecord>()) => {
    const journal = await options.store.list().catch(() => ({ records: [] as SetupRecord[], unreadable: ['journal'] }))
    const unreadable = journal.unreadable
    const records = [...new Map([...journal.records, ...memory.values()].map(record => [record.id, record])).values()]
    let pending = unreadable.length
    for (const home of records.filter(record => record.kind === 'home')) {
      const peers = records.filter(record => record.kind === 'peer' && record.setupId === home.setupId)
      if (home.committed) {
        // A sealed commit records the original owner's successful registration
        // receipts. Delete grants first; leave the commit until all deletes ACK.
        try {for (const peer of peers) {await options.store.remove(peer.id)}; if (!unreadable.length) {await options.store.remove(home.id)}}
        catch {pending++}
        continue
      }
      let allSettled = unreadable.length === 0
      for (const record of [...peers, home]) {
        let client = live.get(record.id)
        const retained = Boolean(client)
        try {
          client ||= (await open(record.route, record.installationId)).client
          if (record.kind === 'peer') {
            let grant = record.grant
            if (!grant) {
              try {grant = (await client.request('groups.peer.invite', record.invitation)).grant}
              catch (error) {
                // Target's replay window proves this old absent request cannot
                // issue again; expired retained receipts carry no live grant.
                if (!(error instanceof RoomSetupError) || error.reason !== 'invitation_request_expired') {throw error}
              }
            }
            if (grant) {
              const receipt = await client.request('groups.peer.revoke', { grant })
              if (receipt?.revoked !== true) {throw new RoomSetupError('cleanup_pending')}
            }
          } else {
            try {
              const receipt = await client.request('groups.disband', { room_id: record.roomId, cancel_id: `setup-${record.setupId}` })
              if (!receipt?.tombstone?.disbanded_at) {throw new RoomSetupError('cleanup_pending')}
            } catch (error) {
              if (error instanceof RoomSetupError && error.reason === 'room_not_found') {
                // The canonical owner's subject check ran before this typed
                // absence receipt. A refused creation left no room to disband.
              } else {
              if (!home.creation || !(error instanceof RoomSetupError) || !['invalid_params', 'permission_denied'].includes(error.reason)) {throw error}
              // A lost create reply is reconciled by the same idempotent create,
              // then ended. Never interpret permission denied as proof of absence.
              const created = await client.request('groups.create', home.creation)
              if (created?.room?.room_id !== record.roomId || created.room.authority_gateway_id !== record.installationId) {throw error}
              const receipt = await client.request('groups.disband', { room_id: record.roomId, cancel_id: `setup-${record.setupId}` })
              if (!receipt?.tombstone?.disbanded_at) {throw new RoomSetupError('cleanup_pending')}
              }
            }
          }
          if (record.kind === 'peer') {await options.store.remove(record.id)}
        } catch {allSettled = false; pending++}
        finally {if (!retained) {client?.close()}}
      }
      if (allSettled) {
        try {await options.store.remove(home.id)} catch {pending++}
      }
    }
    // Orphans are unknown obligations, never dropped as an empty journal.
    pending += records.filter(record => record.kind === 'peer' && !records.some(home => home.kind === 'home' && home.setupId === record.setupId)).length
    return { pending, reason: unreadable.length ? 'setup_journal_unreadable' : pending ? 'cleanup_pending' : undefined }
  }

  return {
    recover: () => exclusive(() => recover()),
    changeStoragePolicy: (apply: () => unknown) => exclusive(async () => apply()),
    create: (input: RoomSetupInput, assertCurrent: () => void) => exclusive(async () => {
      if (!validRoute(input?.home) || typeof input.name !== 'string' || !input.name.trim() || input.name.length > 128 ||
          !Array.isArray(input.members) || input.members.length < 2 || input.members.length > 6 ||
          input.members.some(member => !validRoute({ connectionId: member.connectionId, profile: member.profile }) ||
            !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(member.member_id) || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(member.handle)) ||
          new Set(input.members.map(member => member.handle.toLowerCase())).size !== input.members.length ||
          input.members.some(member => ['all', 'everyone'].includes(member.handle.toLowerCase()))) {
        throw new RoomSetupError('invalid_setup')
      }
      assertCurrent()
      if ((await recover()).pending) {throw new RoomSetupError('cleanup_pending')}
      const connections = new Map<string, Awaited<ReturnType<typeof open>>>()
      const live = new Map<string, Client>()
      const memory = new Map<string, SetupRecord>()
      try {
        connections.set(routeKey(input.home), await open(input.home))
        const home = connections.get(routeKey(input.home))!
        const prepared: Array<{ record: SetupRecord; client: Client; capability: any; member: Member }> = []
        const setupId = randomUUID(), roomId = randomUUID()
        const homeRecord: SetupRecord = { id: setupId, setupId, kind: 'home', route: input.home,
          installationId: home.capability.authority_gateway_id, roomId }
        const roster: Record<string, unknown>[] = []
        for (const member of input.members) {
          const descriptor: Record<string, unknown> = { member_id: member.member_id, profile: member.profile,
            handle: member.handle, ...(member.display_name ? { display_name: member.display_name } : {}) }
          if (member.connectionId === input.home.connectionId) {
            roster.push({ ...descriptor, target: { kind: 'local', profile: member.profile } }); continue
          }
          const route = { connectionId: member.connectionId, profile: member.profile }
          if (route.profile !== 'default') {throw new RoomSetupError('default_peer_profile_required')}
          if (!connections.has(routeKey(route))) {connections.set(routeKey(route), await open(route))}
          const peer = connections.get(routeKey(route))!, link = peer.capability.room_link
          if (!peer.capability.features?.includes('peer_setup_recovery') || !link?.enabled ||
              link.authentication !== 'proof-v2' || !link.endpoint?.available || !link.catalog?.persistent_process ||
              !link.catalog.text || link.catalog.attachments || link.catalog.installation_id !== peer.capability.authority_gateway_id) {
            throw new RoomSetupError('peer_gateway_not_ready')
          }
          const record: SetupRecord = { id: randomUUID(), setupId, kind: 'peer', route,
            installationId: peer.capability.authority_gateway_id, roomId, invitation: {
              request_id: randomUUID(), requested_at: peer.capability.server_time,
              room_id: roomId, member_id: member.member_id, home_install_id: homeRecord.installationId,
              authority_gateway_id: homeRecord.installationId, authority_epoch: 1,
              ttl_seconds: 3600, status_ttl_seconds: 2592000
            } }
          roster.push({ ...descriptor, target: { kind: 'peer', peer_id: record.installationId,
            installation_id: record.installationId, profile: member.profile, capability_digest: link.catalog.catalog_digest } })
          prepared.push({ record, client: peer.client, capability: peer.capability, member })
        }
        if (!prepared.length) {throw new RoomSetupError('peer_required')}
        assertCurrent()
        homeRecord.creation = { room_id: roomId, name: input.name, members: roster }
        await options.store.put(homeRecord)
        memory.set(homeRecord.id, homeRecord)
        live.set(homeRecord.id, home.client)
        const created = await home.client.request('groups.create', { room_id: roomId, name: input.name, members: roster })
        if (created?.room?.room_id !== roomId || created.room.authority_gateway_id !== homeRecord.installationId ||
            created.room.authority_epoch !== 1) {throw new RoomSetupError('original_gateway_required')}
        for (const peer of prepared) {
          assertCurrent()
          await options.store.put(peer.record)
          live.set(peer.record.id, peer.client)
          memory.set(peer.record.id, peer.record)
          const invitation = await peer.client.request('groups.peer.invite', peer.record.invitation)
          if (typeof invitation?.grant !== 'string' || !invitation.grant) {throw new RoomSetupError('invalid_invitation')}
          // Receipt custody precedes lifecycle checks: retirement cannot erase a fresh grant.
          peer.record.grant = invitation.grant
          await options.store.put(peer.record)
          assertCurrent()
          if (invitation.target_profile !== peer.member.profile ||
              !isDeepStrictEqual(invitation.catalog, peer.capability.room_link.catalog) ||
              invitation.endpoint?.url !== peer.capability.room_link.endpoint.url) {throw new RoomSetupError('peer_gateway_changed')}
          const receipt = await home.client.request('groups.peer.register', { room_id: roomId,
            member_id: peer.member.member_id, target_profile: invitation.target_profile,
            target_url: invitation.endpoint.url, catalog: invitation.catalog, grant: invitation.grant })
          if (!receipt?.registered || receipt.target_install_id !== peer.record.installationId ||
              receipt.target_profile !== peer.member.profile) {throw new RoomSetupError('invalid_registration')}
        }
        assertCurrent()
        await options.store.put({ ...homeRecord, committed: true })
        memory.set(homeRecord.id, { ...homeRecord, committed: true })
        // A deletion failure keeps the sealed successful receipt for the next cleanup pass.
        await recover(live, memory)
        return { room: created.room }
      } catch (error) {
        await recover(live, memory).catch(() => undefined)
        throw error
      } finally {for (const { client } of connections.values()) {client.close()}}
    })
  }
}
