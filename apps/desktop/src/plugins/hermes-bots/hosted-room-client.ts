import { atom, host } from '@hermes/plugin-sdk'

import {
  applyHostedPage, hostedRecord, hostedRoomKey, parseHostedCapabilities, parseHostedRoom
} from './hosted-room-protocol'
import type { HostedCapabilities, HostedReplay, HostedRoomIdentity, HostedRoomSummary } from './hosted-room-protocol'
import { getPluginCtx } from './shared'
import type { ProfileRoute } from './types'

interface PendingInput {
  eventId: string
  text: string
  threadId: string
}

export interface HostedRoomCache extends HostedReplay {
  identity: HostedRoomIdentity
  name: string
  room?: HostedRoomSummary
  capabilities?: HostedCapabilities
  driverStatus?: Record<string, unknown>
  pending?: PendingInput
  busy: boolean
  loading: boolean
  error?: string
}

interface HostedDirectory {
  loading: boolean
  keys: string[]
  error?: string
}

// Separate from legacy group-chats/ui_meta: neither its renderer driver nor
// its cross-gateway projection may adopt a hosted room.
export const $hostedRooms = atom<Record<string, HostedRoomCache>>({})
export const $hostedDirectories = atom<Record<string, HostedDirectory>>({})
const generations = new Map<string, number>()
const operations = new Set<string>()
const discoveries = new Map<string, number>()

function bump(key: string): number {
  const generation = (generations.get(key) || 0) + 1
  generations.set(key, generation)

  return generation
}

function patchRoom(key: string, patch: Partial<HostedRoomCache>) {
  const current = $hostedRooms.get()[key]

  if (current) {$hostedRooms.set({ ...$hostedRooms.get(), [key]: { ...current, ...patch } })}
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Hosted room transport failed. Retry when the gateway is available.'
}

function storage() {
  const value = getPluginCtx()?.storage

  if (!value) {throw new Error('Hosted room storage unavailable; input was not sent')}

  return value
}

const pendingKey = (key: string) => `hosted-input:${key}`
const directoryKey = (connectionId: string) => `hosted-rooms:${connectionId}`

async function routeFor(connectionId: string): Promise<ProfileRoute> {
  if (typeof host.profileRoutes !== 'function' || typeof host.requestProfile !== 'function') {
    throw new Error('This Desktop cannot route hosted rooms. Update Desktop to open this room.')
  }

  const routes = await host.profileRoutes()
  const route = routes.find(candidate => candidate.connectionId === connectionId && (candidate.targetProfile || candidate.profile) === 'default')

  if (!route) {throw new Error('The owning gateway is unavailable. Restore its connection and retry.')}

  return route
}

async function negotiate(route: ProfileRoute, identity?: HostedRoomIdentity): Promise<HostedCapabilities> {
  let raw: unknown

  try {
    raw = await host.requestProfile(route, 'groups.capabilities', {})
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === -32601) {
      throw new Error('This gateway does not support hosted Group Chats')
    }

    throw error
  }

  const capabilities = parseHostedCapabilities(raw)

  if (identity && capabilities.authorityGatewayId !== identity.authorityGatewayId) {
    throw new Error('The connection now reaches a different authority. This room will not be retargeted.')
  }

  return capabilities
}

/** Discovery uses the ordinary registered connection, not renderer-local imports
 * or hand-written storage. Old saved identities remain openable during outages. */
export async function discoverHostedRooms(connectionId: string): Promise<void> {
  const generation = (discoveries.get(connectionId) || 0) + 1
  discoveries.set(connectionId, generation)
  const current = () => discoveries.get(connectionId) === generation

  const publish = (value: HostedDirectory) => {
    if (current()) {$hostedDirectories.set({ ...$hostedDirectories.get(), [connectionId]: value })}
  }

  const previous = $hostedDirectories.get()[connectionId]
  publish({ keys: previous?.keys || [], loading: true })
  let keys = previous?.keys || []

  try {
    const saved = await storage().get<Array<{ identity: HostedRoomIdentity; name: string }>>(directoryKey(connectionId), [])

    if (!current()) {return}

    for (const bookmark of Array.isArray(saved) ? saved : []) {
      if (bookmark?.identity?.connectionId !== connectionId || !bookmark.identity.authorityGatewayId || !bookmark.identity.roomId) {continue}
      const key = hostedRoomKey(bookmark.identity)

      if (!$hostedRooms.get()[key]) {
        $hostedRooms.set({ ...$hostedRooms.get(), [key]: { ...bookmark, cursor: 0, events: [], busy: false, loading: false } })
      }

      if (!keys.includes(key)) {keys = [...keys, key]}
    }

    publish({ keys, loading: true })
    const route = await routeFor(connectionId)
    const capabilities = await negotiate(route)
    let offset = 0
    const found: HostedRoomSummary[] = []

    // Bound a corrupt pagination stream; the server currently caps the room inventory.
    for (let pageIndex = 0; pageIndex < 100; pageIndex++) {
      const page = hostedRecord(await host.requestProfile(route, 'groups.list', { offset, limit: 100, include_disbanded: false }))

      if (!current()) {return}

      if (!Array.isArray(page.rooms)) {throw new Error('Invalid hosted room directory')}

      for (const raw of page.rooms) {
        const room = parseHostedRoom(raw)

        if (room.authority_gateway_id === capabilities.authorityGatewayId) {found.push(room)}
      }

      if (page.next_offset === null) {break}

      if (typeof page.next_offset !== 'number' || !Number.isSafeInteger(page.next_offset) || page.next_offset <= offset || !page.rooms.length || pageIndex === 99) {
        throw new Error('Invalid hosted room directory pagination')
      }

      offset = page.next_offset
    }

    const bookmarks = found.map(room => ({
      identity: { connectionId, authorityGatewayId: room.authority_gateway_id, roomId: room.room_id }, name: room.name
    }))

    await storage().set(directoryKey(connectionId), bookmarks)

    if (!current()) {return}
    const all = { ...$hostedRooms.get() }
    keys = bookmarks.map((bookmark, index) => {
      const key = hostedRoomKey(bookmark.identity)
      // Discovery may rename a row but cannot overwrite a live replay/status.
      all[key] = all[key] ? { ...all[key], name: bookmark.name } : {
        ...bookmark, room: found[index], capabilities, cursor: 0, events: [], busy: false, loading: false
      }

      return key
    })
    $hostedRooms.set(all)
    publish({ keys, loading: false })
  } catch (error) {
    publish({ keys, loading: false, error: message(error) })
  }
}

/** State + fully contiguous replay are a single guarded publication. Failure
 * preserves the previous cache/cursor; reopening reconstructs from seq zero. */
export async function refreshHostedRoom(key: string): Promise<void> {
  const cache = $hostedRooms.get()[key]

  if (!cache || operations.has(key)) {return}
  const generation = bump(key)
  const current = () => generations.get(key) === generation
  patchRoom(key, { loading: true })

  try {
    const pending = await storage().get<PendingInput | null>(pendingKey(key), null)
    const route = await routeFor(cache.identity.connectionId)
    const capabilities = await negotiate(route, cache.identity)
    const state = hostedRecord(await host.requestProfile(route, 'groups.state', { room_id: cache.identity.roomId }))

    if (!current()) {return}
    const room = parseHostedRoom(state.room, cache.identity)
    let replay: HostedReplay = cache
    let hasMore = true

    for (let index = 0; hasMore && index < 1000; index++) {
      const raw = await host.requestProfile(route, 'groups.log', { room_id: cache.identity.roomId, since_seq: replay.cursor, limit: capabilities.logLimit })

      if (!current()) {return}
      const page = applyHostedPage(replay, raw, room)
      replay = page
      hasMore = page.hasMore
    }

    if (hasMore) {throw new Error('Hosted replay exceeded the page budget. Retry to reconnect.')}

    if (!current()) {return}
    patchRoom(key, {
      ...replay, room, name: room.name, capabilities,
      driverStatus: state.driver_status ? hostedRecord(state.driver_status) : undefined,
      pending: pending || undefined, loading: false, error: undefined
    })
  } catch (error) {
    if (current()) {patchRoom(key, { loading: false, error: message(error) })}
  }
}

export function invalidateHostedRoom(key: string): void {
  bump(key)
}

/** One unresolved input per room. An uncertain acceptance is NOT a new input;
 * retry its exact persisted id and payload, even after a renderer restart. */
export async function sendHostedInput(key: string, text?: string, threadId?: string | null): Promise<boolean> {
  const cache = $hostedRooms.get()[key]

  if (!cache || operations.has(key)) {return false}
  operations.add(key)
  bump(key)
  patchRoom(key, { busy: true, error: undefined })
  let sent = false

  try {
    let pending = await storage().get<PendingInput | null>(pendingKey(key), null)

    if (pending && text !== undefined && (pending.text !== text.trim() || (threadId && pending.threadId !== threadId))) {
      throw new Error('Resolve the pending input before sending another message')
    }

    if (!pending) {
      if (!text?.trim()) {throw new Error('Enter a text message')}
      pending = { eventId: crypto.randomUUID(), text: text.trim(), threadId: threadId || crypto.randomUUID() }
      // Failure to persist is fatal: never risk minting a second id on reload.
      await storage().set(pendingKey(key), pending)
    }

    patchRoom(key, { pending })
    const route = await routeFor(cache.identity.connectionId)
    const capabilities = await negotiate(route, cache.identity)

    if (!capabilities.driver) {throw new Error('The hosted room worker is unavailable; your input is saved for retry')}
    const state = hostedRecord(await host.requestProfile(route, 'groups.state', { room_id: cache.identity.roomId }))
    const room = parseHostedRoom(state.room, cache.identity)

    const result = hostedRecord(await host.requestProfile(route, 'groups.send', {
      room_id: cache.identity.roomId, event_id: pending.eventId,
      payload: { text: pending.text, thread_id: pending.threadId }
    }))

    const event = hostedRecord(result.event)
    const payload = hostedRecord(event.payload)

    if (result.accepted !== true || result.client_event_id !== pending.eventId || event.room_id !== cache.identity.roomId ||
        event.kind !== 'message.user' || payload.text !== pending.text || payload.thread_id !== pending.threadId ||
        typeof event.event_id !== 'string' || typeof event.seq !== 'number' || !Number.isSafeInteger(event.seq) || event.seq < 1) {
      throw new Error('Invalid hosted input acknowledgement; retry the saved input')
    }

    // Verify the exact committed event, without jumping over intervening events
    // or treating a write acknowledgement as the transcript.
    const page = await host.requestProfile(route, 'groups.log', {
      room_id: cache.identity.roomId, since_seq: event.seq - 1, limit: 1
    })
    // Reuse replay validation for authority, sequence and high-water marks.
    // This targeted proof never advances the transcript's replay cursor.
    const proof = applyHostedPage({ cursor: event.seq - 1, events: [] }, page, room)
    const committed = proof.events[0]
    const actor = hostedRecord(event.actor)

    if (proof.events.length !== 1 || committed.event_id !== event.event_id || committed.kind !== 'message.user' ||
        committed.payload.text !== pending.text || committed.payload.thread_id !== pending.threadId ||
        committed.actor.kind !== 'user' || committed.actor.id !== actor.id ||
        committed.created_at !== event.created_at || committed.authority_epoch !== event.authority_epoch) {
      throw new Error('Hosted input acknowledgement was not found in the committed log')
    }

    await storage().set(pendingKey(key), null)
    patchRoom(key, { pending: undefined })
    sent = true
  } catch (error) {
    patchRoom(key, { error: message(error) })
  } finally {
    operations.delete(key)
    patchRoom(key, { busy: false, loading: false })
  }

  if (sent) {await refreshHostedRoom(key)}

  return sent
}

export async function stopHostedRoom(key: string): Promise<void> {
  const cache = $hostedRooms.get()[key]

  if (!cache || operations.has(key)) {return}
  operations.add(key)
  bump(key)
  patchRoom(key, { busy: true, error: undefined })
  let stopped = false

  try {
    const route = await routeFor(cache.identity.connectionId)
    await negotiate(route, cache.identity)
    const state = hostedRecord(await host.requestProfile(route, 'groups.state', { room_id: cache.identity.roomId }))
    parseHostedRoom(state.room, cache.identity)
    await host.requestProfile(route, 'groups.stop', { room_id: cache.identity.roomId, cancel_id: crypto.randomUUID() })
    stopped = true
  } catch (error) {
    patchRoom(key, { error: message(error) })
  } finally {
    operations.delete(key)
    patchRoom(key, { busy: false })
  }

  if (stopped) {await refreshHostedRoom(key)}
}
