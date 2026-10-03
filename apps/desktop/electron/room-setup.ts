import { randomUUID } from 'node:crypto'
import { isDeepStrictEqual } from 'node:util'

import { RoomSetupError } from './room-setup-store'
import type { roomSetupStore, SetupRecord } from './room-setup-store'
import type { RoomBackupInput, RoomSetupInput, RoomSetupMember, SetupRoute } from './room-setup-types'

interface Client { request(method: string, params?: Record<string, unknown>): Promise<any>; close(): void }
const routeKey = (route: SetupRoute) => JSON.stringify([route.connectionId, route.profile])
const validRoute = (route: SetupRoute) => route && typeof route.connectionId === 'string' && route.connectionId.length > 0 &&
  route.connectionId.length <= 256 && typeof route.profile === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(route.profile)
/** A computer that can keep and continue a group advertises the succession surface. */
const continues = (capability: { methods?: unknown }) => Array.isArray(capability?.methods) && capability.methods.includes('groups.succession.status')
const peerReady = (capability: any) => {
  const link = capability?.room_link
  return capability?.features?.includes('peer_setup_recovery') && link?.enabled && link.authentication === 'proof-v2' &&
    link.endpoint?.available && link.catalog?.persistent_process && link.catalog.installation_id === capability.authority_gateway_id
}

/** Setup only. The gateway remains the sole owner of execution and history. */
export function roomSetupCoordinator(options: {
  store: ReturnType<typeof roomSetupStore>; connect: (route: SetupRoute) => Promise<Client>
  beforeOperation?: () => void
}) {
  let serial = Promise.resolve()
  const exclusive = <T>(work: () => Promise<T>) => {
    const guarded = () => {options.beforeOperation?.(); return work()}
    const result = serial.then(guarded, guarded)
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
  /** A backup enrolment that didn't finish is withdrawn on the host, then its grant is revoked. Returns what remains. */
  const settleCustody = async (records: SetupRecord[], live: Map<string, Client>) => {
    let pending = 0
    for (const record of records.filter(record => record.kind === 'custody')) {
      if (record.committed) {
        try {await options.store.remove(record.id)} catch {pending++}
        continue
      }
      let client = live.get(record.id), host: Client | undefined
      const retained = Boolean(client)
      try {
        let grant = record.grant
        if (!grant) {
          client ||= (await open(record.route, record.installationId)).client
          try {grant = (await client.request('groups.peer.invite', record.invitation)).grant}
          catch (error) {if (!(error instanceof RoomSetupError) || error.reason !== 'invitation_request_expired') {throw error}}
        }
        if (grant) {
          // An add whose reply was lost may have reached the host: withdraw it there before the grant.
          host = (await open(record.home!, record.homeInstallationId)).client
          try {await host.request('groups.custody.remove', { room_id: record.roomId, install_id: record.installationId })}
          catch (error) {if (!(error instanceof RoomSetupError) || !['room_custody_invalid', 'room_not_found'].includes(error.reason)) {throw error}}
          client ||= (await open(record.route, record.installationId)).client
          const receipt = await client.request('groups.peer.revoke', { grant })
          if (receipt?.revoked !== true) {throw new RoomSetupError('cleanup_pending')}
        }
        await options.store.remove(record.id)
      } catch {pending++}
      finally {if (!retained) {client?.close()}; host?.close()}
    }
    return pending
  }
  const recover = async (live = new Map<string, Client>(), memory = new Map<string, SetupRecord>()) => {
    const journal = await options.store.list().catch(() => ({ records: [] as SetupRecord[], unreadable: ['journal'] }))
    const unreadable = journal.unreadable
    const records = [...new Map([...journal.records, ...memory.values()].map(record => [record.id, record])).values()]
    let pending = unreadable.length + await settleCustody(records, live)
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

  /** The owner's designation, after the group exists. It never undoes a created group: a failure is reported. */
  const designate = async (input: RoomSetupInput, roomId: string, home: { client: Client; capability: any },
    prepared: Array<{ record: SetupRecord; capability: any }>) => {
    const peers = [...new Set(prepared.filter(peer => peer.record.invitation?.successor === true).map(peer => peer.record.installationId))]
    if (input.successor !== true || !peers.length || !home.capability.methods?.includes('groups.custody.designate')) {return {}}
    try {
      for (const installId of peers) {
        const receipt = await home.client.request('groups.custody.designate', { room_id: roomId, install_id: installId, successor: true })
        if (receipt?.install_id !== installId || receipt.successor !== true) {throw new RoomSetupError('invalid_registration')}
      }
      return { successors: 'designated' as const }
    } catch {return { successors: 'failed' as const }}
  }

  return {
    recover: () => exclusive(() => recover()),
    /** A custodian-only grant from the backup computer, enrolled by the host. Grants never leave this process. */
    addBackup: (input: RoomBackupInput, assertCurrent: () => void) => exclusive(async () => {
      if (!validRoute(input?.home) || !validRoute(input?.backup) || routeKey(input.home) === routeKey(input.backup) ||
          typeof input.roomId !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(input.roomId) || typeof input.successor !== 'boolean') {
        throw new RoomSetupError('invalid_setup')
      }
      assertCurrent()
      if ((await recover()).pending) {throw new RoomSetupError('cleanup_pending')}
      const live = new Map<string, Client>(), memory = new Map<string, SetupRecord>()
      let home: Awaited<ReturnType<typeof open>> | undefined, backup: Awaited<ReturnType<typeof open>> | undefined
      try {
        home = await open(input.home)
        if (!home.capability.methods?.includes('groups.custody.add')) {throw new RoomSetupError('custody_unavailable')}
        backup = await open(input.backup)
        if (!continues(backup.capability) || !peerReady(backup.capability)) {throw new RoomSetupError('backup_gateway_unsupported')}
        if (backup.capability.authority_gateway_id === home.capability.authority_gateway_id) {throw new RoomSetupError('invalid_setup')}
        const epoch = (await home.client.request('groups.state', { room_id: input.roomId }))?.room?.authority_epoch
        if (!Number.isSafeInteger(epoch) || epoch < 1) {throw new RoomSetupError('original_gateway_required')}
        const link = backup.capability.room_link, homeInstall = home.capability.authority_gateway_id
        const setupId = randomUUID()
        const record: SetupRecord = { id: setupId, setupId, kind: 'custody', route: input.backup, roomId: input.roomId,
          installationId: backup.capability.authority_gateway_id, home: input.home, homeInstallationId: homeInstall, invitation: {
            request_id: randomUUID(), requested_at: backup.capability.server_time, room_id: input.roomId,
            home_install_id: homeInstall, authority_gateway_id: homeInstall, authority_epoch: epoch, custody_only: true,
            ttl_seconds: 3600, status_ttl_seconds: 2592000, ...input.successor ? { successor: true } : {}
          } }
        assertCurrent()
        await options.store.put(record)
        memory.set(record.id, record)
        live.set(record.id, backup.client)
        const invitation = await backup.client.request('groups.peer.invite', record.invitation)
        if (typeof invitation?.grant !== 'string' || !invitation.grant) {throw new RoomSetupError('invalid_invitation')}
        record.grant = invitation.grant
        await options.store.put(record)
        assertCurrent()
        if (!isDeepStrictEqual(invitation.catalog, link.catalog) || invitation.endpoint?.url !== link.endpoint.url) {throw new RoomSetupError('peer_gateway_changed')}
        const receipt = await home.client.request('groups.custody.add', { room_id: input.roomId, target_url: invitation.endpoint.url,
          catalog: invitation.catalog, grant: invitation.grant, successor: input.successor })
        if (receipt?.room_id !== input.roomId || receipt.install_id !== record.installationId) {throw new RoomSetupError('invalid_registration')}
        // The host now holds the grant; the journal entry was only the obligation to undo it.
        const committed = { ...record, committed: true }
        await options.store.put(committed)
        memory.set(record.id, committed)
        await recover(live, memory)
        return { install_id: record.installationId }
      } catch (error) {
        await recover(live, memory).catch(() => undefined)
        throw error
      } finally {home?.client.close(); backup?.client.close()}
    }),
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
        const prepared: Array<{ record: SetupRecord; client: Client; capability: any; member: RoomSetupMember }> = []
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
          if (!peerReady(peer.capability) || !link.catalog.text || link.catalog.attachments) {
            throw new RoomSetupError('peer_gateway_not_ready')
          }
          // Your own computer's consent travels in its grant; only one that can continue a group is asked.
          const successor = input.successor === true && continues(peer.capability)
          const record: SetupRecord = { id: randomUUID(), setupId, kind: 'peer', route,
            installationId: peer.capability.authority_gateway_id, roomId, invitation: {
              request_id: randomUUID(), requested_at: peer.capability.server_time,
              room_id: roomId, member_id: member.member_id, home_install_id: homeRecord.installationId,
              authority_gateway_id: homeRecord.installationId, authority_epoch: 1,
              ttl_seconds: 3600, status_ttl_seconds: 2592000, ...successor ? { successor: true } : {}
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
        return { room: created.room, ...await designate(input, roomId, home, prepared) }
      } catch (error) {
        await recover(live, memory).catch(() => undefined)
        throw error
      } finally {for (const { client } of connections.values()) {client.close()}}
    })
  }
}
