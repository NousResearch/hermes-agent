import { sha256 } from '@noble/hashes/sha2.js'
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js'

import { groupChatSyncSequence } from './group-message-author'
import { createHostedRoomReplayState, reduceHostedRoomEvents } from './hosted-room-client'
import type { HostedRoomCommand, HostedRoomOutbox } from './hosted-room-client'
import type { GroupChat, GroupMessage } from './types'

// Local spreads preserve proof, but JSON cannot manufacture it. Every storage
// and projection boundary also explicitly removes it, including in-memory copies.
const outgoingProof = Symbol('hosted-user-outgoing-proof')
interface OutgoingProof {
  roomId: string
  key: string
  eventId?: string
  seq?: number
}
type ProvenUserMessage = GroupMessage & { [outgoingProof]?: OutgoingProof }

function clientKey(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  const key = value.trim()

  return key.length <= 128 && /^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(key) ? key : null
}

/** The gateway's normalized retry-key namespace, not a transform of event IDs. */
function userEventId(key: string) {
  return `user:${bytesToHex(sha256(utf8ToBytes(key)))}`
}

function stableId(entry: GroupMessage): string | null {
  if (entry.id && entry.eventId && entry.id !== entry.eventId) {
    return null
  }

  return clientKey(entry.eventId || entry.id)
}

export function outgoingHostedUserEvent(entry: GroupMessage, roomId: string, commandId: string): GroupMessage {
  const key = clientKey(commandId)

  if (
    !key ||
    entry.from?.kind !== 'user' ||
    entry.eventId ||
    groupChatSyncSequence(entry) !== null ||
    stableId(entry) !== key ||
    (entry.roomId && entry.roomId !== roomId)
  ) {
    return entry
  }

  return { ...entry, roomId, [outgoingProof]: { roomId, key } } as ProvenUserMessage
}

function provenClientKey(entry: GroupMessage, roomId: string): string | null {
  const proof = (entry as ProvenUserMessage)[outgoingProof]

  if (!proof || proof.roomId !== roomId || entry.roomId !== roomId) {
    return null
  }

  const seq = groupChatSyncSequence(entry)
  const id = stableId(entry)

  return proof.seq !== undefined
    ? seq === proof.seq && id === proof.eventId
      ? proof.key
      : null
    : seq === null && !entry.eventId && id === proof.key
      ? proof.key
      : null
}

export function storedHostedUserEvent(entry: GroupMessage): GroupMessage {
  if (!(entry as ProvenUserMessage)[outgoingProof] && !('clientEventId' in entry)) {
    return entry
  }

  const { [outgoingProof]: _proof, clientEventId: _legacyClaim, ...display } = entry as ProvenUserMessage

  return display
}

function userPayload(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const payload = value as Record<string, unknown>

  if (payload.text !== undefined && typeof payload.text !== 'string') {
    return null
  }

  for (const field of ['thread_id', 'thread']) {
    if (payload[field] !== undefined && typeof payload[field] !== 'string') {
      return null
    }
  }

  if (payload.thread_id && payload.thread && String(payload.thread_id).trim() !== String(payload.thread).trim()) {
    return null
  }

  const attachments = payload.attachments === undefined ? [] : payload.attachments

  if (
    !Array.isArray(attachments) ||
    attachments.length > 8 ||
    attachments.some(
      item =>
        !item ||
        typeof item !== 'object' ||
        Array.isArray(item) ||
        typeof item.size !== 'number' ||
        !['attachment_id', 'kind', 'name', 'mime'].every(field => typeof item[field] === 'string')
    )
  ) {
    return null
  }

  return payload
}

/** Re-establish intent only from the independently persisted command outbox,
 * never from a message's clientEventId claim after a cold restore. */
export function restoreHostedUserOutboxIntents(room: GroupChat, outbox: HostedRoomOutbox): GroupMessage[] {
  return room.log.map(entry => {
    const command = outbox.commands.find(
      candidate =>
        candidate.kind === 'send' &&
        candidate.status !== 'failed' &&
        candidate.roomId === room.roomId &&
        (candidate.authorityId
          ? candidate.authorityId === room.hosted
          : candidate.connectionId === room.hostedConnectionId) &&
        clientKey(candidate.commandId) === stableId(entry)
    )

    const payload = command && userPayload(command.payload)

    if (
      !room.roomId ||
      !command ||
      !payload ||
      entry.text !== (payload.text || '') ||
      (entry.thread || 'legacy') !== (String(payload.thread_id || payload.thread || '').trim() || 'legacy')
    ) {
      return entry
    }

    const attachments = (payload.attachments || []) as Record<string, unknown>[]

    if (
      (entry.images?.length || 0) !== attachments.length ||
      attachments.some((item, index) => entry.images?.[index]?.attachmentId !== item.attachment_id)
    ) {
      return entry
    }

    return outgoingHostedUserEvent(entry, room.roomId, command.commandId)
  })
}

/** Reconcile only within an identified hosted room. Mirrors never supply payload
 * fields to accepted user events, and every sequenced record stays intact. */
export function reconcileHostedUserEvents(roomId: string, ...logs: GroupMessage[][]): GroupMessage[] {
  // Bind every legacy actor to its known container before shared matching.
  // A genuine member sequence must never be lent to an unscoped user hint.
  const entries = logs.flat().map(entry => (!entry.roomId ? { ...entry, roomId } : entry))

  const canonical = new Map<string, Set<string>>()
  const receiptedKeys = new Set<string>()
  const inRoom = (entry: GroupMessage) => !entry.roomId || entry.roomId === roomId

  for (const entry of entries) {
    const seq = groupChatSyncSequence(entry)
    const id = stableId(entry)

    if (entry.from?.kind === 'user' && inRoom(entry) && seq !== null && id) {
      const events = canonical.get(id) || new Set<string>()
      events.add(`${id}:${seq}`)
      canonical.set(id, events)
      const key = provenClientKey(entry, roomId)

      if (key && id === userEventId(key)) {
        receiptedKeys.add(key)
      }
    }
  }

  return entries.filter(entry => {
    if (entry.from?.kind !== 'user' || !inRoom(entry) || groupChatSyncSequence(entry) !== null) {
      return true
    }

    const id = stableId(entry)

    if (!id) {
      return true
    }

    // A raw UUID may itself be an old backend's genuine canonical event. Its
    // own sequence/marker takes precedence over any possible hashed neighbor.
    if (canonical.has(id)) {
      return canonical.get(id)!.size !== 1
    }

    if (entry.eventId) {
      return true
    }

    const explicitClient = receiptedKeys.has(id) || provenClientKey(entry, roomId) === id

    if (/^user:[0-9a-f]{64}$/.test(id) && !explicitClient) {
      return true
    }

    return canonical.get(userEventId(id))?.size !== 1
  })
}

/** Display projections carry no authoritative sequence, regardless of actor.
 * Keep supplied canonical markers, but never promote an imported sequence. */
export function projectedGroupMessage(entry: GroupMessage, roomId?: null | string): GroupMessage {
  const display = storedHostedUserEvent(entry)
  const projected = display.seq !== undefined ? { ...display, seq: undefined } : display

  return roomId && !projected.roomId ? { ...projected, roomId } : projected
}

/** Validate the response against the exact command before using the normal replay
 * decoder. A receipt does not advance the room cursor past unobserved events. */
export function hostedUserEventReceipt(command: HostedRoomCommand, value: unknown): GroupMessage | null {
  if (command.kind !== 'send' || !value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const reply = value as Record<string, unknown>
  const event = reply.event as Record<string, unknown> | undefined
  const key = clientKey(command.commandId)

  if (
    !key ||
    !event ||
    typeof event !== 'object' ||
    Array.isArray(event) ||
    (reply.client_event_id !== undefined && clientKey(reply.client_event_id) !== key) ||
    event.room_id !== command.roomId ||
    event.kind !== 'message.user' ||
    (event.event_id !== key && event.event_id !== userEventId(key)) ||
    typeof event.seq !== 'number' ||
    !Number.isSafeInteger(event.seq) ||
    event.seq < 1
  ) {
    return null
  }

  const payload = userPayload(event.payload)

  if (!payload) {
    return null
  }

  const state = reduceHostedRoomEvents(createHostedRoomReplayState({ roomId: command.roomId, cursor: event.seq - 1 }), [
    event
  ])

  const message = state.messages[0]
  const attachments = (payload.attachments || []) as unknown[]

  // The legacy decoder drops malformed attachments. A receipt must be complete,
  // not a lossy display projection that could suppress valid pending content.
  if (
    !message ||
    attachments.length !== (message.images?.length || 0) ||
    (!message.text.trim() && !message.images?.length)
  ) {
    return null
  }

  return {
    ...message,
    id: message.eventId,
    roomId: command.roomId,
    clientEventId: key,
    [outgoingProof]: { roomId: command.roomId, key, eventId: message.eventId, seq: message.seq }
  } as ProvenUserMessage
}
