/** Verified, cross-window persistence for gateway-hosted Group Chat commands. */

import { host } from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'

import { createHostedRoomOutbox, reduceHostedRoomOutbox } from './hosted-room-client'
import type { HostedRoomOutbox, HostedRoomOutboxAction } from './hosted-room-client'
import { botsText } from './i18n'

export const HOSTED_ROOM_OUTBOX_KEY = 'hosted-room-outbox-v1'

const OUTBOX_MUTATION_LOCK = 'hermes-bots-hosted-room-outbox'
const OUTBOX_DISPATCH_LOCK = 'hermes-bots-hosted-room-outbox-dispatch'
const OUTBOX_ROOM_LOCK_PREFIX = 'hermes-bots-hosted-room-order:'

interface LockManager {
  request<T>(
    name: string,
    options: { mode: 'exclusive' },
    callback: (lock: null | object) => Promise<T> | T
  ): Promise<T>
}

let mutationTail: Promise<void> = Promise.resolve()
let dispatchTail: Promise<void> = Promise.resolve()
const roomTails = new Map<string, Promise<void>>()
let repairNotified = false

function lockManager() {
  return (globalThis.navigator as (Navigator & { locks?: LockManager }) | undefined)?.locks
}

function processLock<T>(kind: 'dispatch' | 'mutation', callback: () => Promise<T>) {
  const tail = kind === 'mutation' ? mutationTail : dispatchTail
  const result = tail.then(callback, callback)

  const settled = result.then(
    () => undefined,
    () => undefined
  )

  if (kind === 'mutation') {
    mutationTail = settled
  } else {
    dispatchTail = settled
  }

  return result
}

function withLock<T>(name: string, kind: 'dispatch' | 'mutation', callback: () => Promise<T>) {
  const locks = lockManager()

  return locks?.request ? locks.request(name, { mode: 'exclusive' }, callback) : processLock(kind, callback)
}

function sameOutbox(left: HostedRoomOutbox, right: HostedRoomOutbox) {
  return JSON.stringify(left) === JSON.stringify(right)
}

export async function readHostedRoomOutbox(storage: null | PluginContext['storage']) {
  if (typeof storage?.get !== 'function' || typeof storage.set !== 'function') {
    throw new Error('Desktop storage is unavailable, so Group Chat changes cannot be secured.')
  }

  const raw = await storage.get(HOSTED_ROOM_OUTBOX_KEY, null)
  const candidate = raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as { commands?: unknown }) : null
  const rows = Array.isArray(candidate?.commands) ? candidate.commands : []
  const outbox = createHostedRoomOutbox(raw, false, true)
  const invalidEnvelope = raw !== null && (!candidate || !Array.isArray(candidate.commands))
  const dropped = Math.max(invalidEnvelope ? 1 : 0, rows.length - outbox.commands.length)

  if (!dropped) {
    return outbox
  }

  await storage.set(HOSTED_ROOM_OUTBOX_KEY, outbox)
  const repaired = createHostedRoomOutbox(await storage.get(HOSTED_ROOM_OUTBOX_KEY, null), false)

  if (!sameOutbox(repaired, outbox)) {
    throw new Error('Desktop storage could not repair the Group Chat queue.')
  }

  if (!repairNotified) {
    repairNotified = true
    host.notify({ kind: 'warning', message: botsText().group.hostedQueueRepaired(dropped) })
  }

  return repaired
}

async function persistOutbox(storage: PluginContext['storage'], next: HostedRoomOutbox) {
  await storage.set(HOSTED_ROOM_OUTBOX_KEY, next)
  const persisted = await readHostedRoomOutbox(storage)

  if (!sameOutbox(persisted, next)) {
    throw new Error('Desktop storage did not persist the Group Chat change.')
  }

  return persisted
}

export async function mutateHostedRoomOutbox(storage: null | PluginContext['storage'], action: HostedRoomOutboxAction) {
  return withLock(OUTBOX_MUTATION_LOCK, 'mutation', async () => {
    if (typeof storage?.set !== 'function') {
      throw new Error('Desktop storage is unavailable, so Group Chat changes cannot be secured.')
    }

    const current = await readHostedRoomOutbox(storage)
    const next = reduceHostedRoomOutbox(current, action)

    if (sameOutbox(current, next)) {
      return current
    }

    return persistOutbox(storage, next)
  })
}

export async function recoverHostedRoomOutbox(storage: null | PluginContext['storage']) {
  return withLock(OUTBOX_MUTATION_LOCK, 'mutation', async () => {
    if (typeof storage?.set !== 'function') {
      throw new Error('Desktop storage is unavailable, so Group Chat changes cannot be secured.')
    }

    const current = await readHostedRoomOutbox(storage)
    const recovered = createHostedRoomOutbox(current)

    return sameOutbox(current, recovered) ? current : persistOutbox(storage, recovered)
  })
}

export function withHostedRoomOutboxDispatch<T>(callback: () => Promise<T>) {
  return withLock(OUTBOX_DISPATCH_LOCK, 'dispatch', callback)
}

export function withHostedRoomCommandOrder<T>(roomId: string, callback: () => Promise<T>) {
  const id = String(roomId || '')
  const locks = lockManager()

  if (!id) {
    return Promise.reject(new Error('Group Chat command order requires a room id.'))
  }

  if (locks?.request) {
    return locks.request(`${OUTBOX_ROOM_LOCK_PREFIX}${id}`, { mode: 'exclusive' }, callback)
  }

  const previous = roomTails.get(id) || Promise.resolve()
  const result = previous.then(callback, callback)

  roomTails.set(
    id,
    result.then(
      () => undefined,
      () => undefined
    )
  )

  return result
}

export function resetHostedRoomOutboxLocksForTests() {
  mutationTail = Promise.resolve()
  dispatchTail = Promise.resolve()
  roomTails.clear()
  repairNotified = false
}
