/** Strict metadata-only client for the hosted Group Chat shared-files RPC. */

import { $groupChats, groupChatHostedGateway } from './group-chat'
import { hostedRouteForRoom, requestHostedConnection } from './hosted-room-runtime'
import type { Attachment, GroupChat } from './types'

const ATTACHMENT_ID_RE = /^att_[0-9a-f]{32}$/
const IDENTIFIER_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/
const MIME_RE = /^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/i
const MAX_ATTACHMENT_BYTES = 15_000_000
const MAX_CURSOR_LENGTH = 2048
export const GROUP_FILES_PAGE_SIZE = 8
export const GROUP_FILES_MAX_PAGE_SIZE = 32
export const GROUP_FILES_MAX_QUERY_LENGTH = 255

export interface GroupFileProducer {
  identity: string
  kind: 'member' | 'user'
  label: string
}

export interface GroupFileItem {
  attachment: Attachment
  eventId: string
  producer: GroupFileProducer
  seq: number
  sharedAt: number
}

export interface GroupFilesPage {
  authority: { epoch: number; gatewayId: string }
  hasMore: boolean
  items: GroupFileItem[]
  nextCursor: null | string
  snapshotSeq: number
}

export interface GroupFilesListInput {
  cursor?: string
  limit?: number
  query?: string
}

export function isGroupFilesCursorError(error: unknown): boolean {
  const outer = record(error)
  const inner = record(outer?.error)
  const message = String(outer?.message || inner?.message || '')

  return /attachment list cursor (?:is invalid|does not match this request|must be|is too large)/i.test(message)
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function requiredText(value: unknown, label: string, maxLength = 4096): string {
  if (typeof value !== 'string' || !value.trim() || [...value].length > maxLength) {
    throw new Error(`Invalid shared-files ${label}`)
  }

  return value
}

function integer(value: unknown, label: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum) {
    throw new Error(`Invalid shared-files ${label}`)
  }

  return Number(value)
}

async function boundedRequest<T>(request: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      request,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Shared-files request timed out')), 10_000)
      })
    ])
  } finally {
    clearTimeout(timer)
  }
}

// Backend wire naming is intentionally contained here. Reconciliation of the
// producer envelope should never leak into view state or download identity.
function parseProducer(value: unknown): GroupFileProducer {
  const producer = record(value)
  const kind = producer?.kind
  const identity = requiredText(producer?.id, 'producer identity', 128)

  if ((kind !== 'member' && kind !== 'user') || !IDENTIFIER_RE.test(identity)) {
    throw new Error('Invalid shared-files producer kind')
  }

  return {
    identity,
    kind,
    label: requiredText(producer?.label, 'producer label', 256)
  }
}

function parseItem(value: unknown, snapshotSeq: number): GroupFileItem {
  const item = record(value)
  const attachmentId = requiredText(item?.attachment_id, 'attachment id', 64)
  const eventId = requiredText(item?.event_id, 'event id', 128)
  const kind = requiredText(item?.kind, 'kind', 16)
  const name = requiredText(item?.name, 'name', 255)
  const mime = requiredText(item?.mime, 'MIME', 127)
  const size = integer(item?.size, 'size')
  const seq = integer(item?.seq, 'sequence', 1)
  const sharedAt = item?.shared_at

  if (
    !ATTACHMENT_ID_RE.test(attachmentId) ||
    !IDENTIFIER_RE.test(eventId) ||
    !['file', 'image', 'pdf'].includes(kind) ||
    !MIME_RE.test(mime) ||
    size > MAX_ATTACHMENT_BYTES ||
    seq > snapshotSeq ||
    typeof sharedAt !== 'number' ||
    !Number.isFinite(sharedAt) ||
    sharedAt <= 0 ||
    sharedAt > 8_640_000_000_000
  ) {
    throw new Error('Invalid shared-files item')
  }

  return {
    attachment: { attachmentId, kind: kind as Attachment['kind'], mime, name, size },
    eventId,
    producer: parseProducer(item?.producer),
    seq,
    sharedAt
  }
}

export function parseGroupFilesPage(
  value: unknown,
  expected: { authorityEpoch?: null | number; authorityId?: null | string; limit?: number } = {}
): GroupFilesPage {
  const response = record(value)
  const authority = record(response?.authority)
  const snapshotSeq = integer(response?.snapshot_seq, 'snapshot')
  const gatewayId = requiredText(authority?.gateway_id, 'authority gateway', 256)
  const epoch = integer(authority?.epoch, 'authority epoch', 1)
  const hasMore = response?.has_more
  const rawCursor = response?.next_cursor
  const nextCursor = rawCursor === null ? null : requiredText(rawCursor, 'cursor', MAX_CURSOR_LENGTH)
  const rawItems = response?.items
  const limit = Math.min(GROUP_FILES_MAX_PAGE_SIZE, integer(expected.limit ?? GROUP_FILES_PAGE_SIZE, 'page size', 1))

  if (
    typeof hasMore !== 'boolean' ||
    response?.ok === false ||
    response?.error !== undefined ||
    !IDENTIFIER_RE.test(gatewayId) ||
    !Array.isArray(rawItems) ||
    rawItems.length > limit ||
    (hasMore && nextCursor === null) ||
    (!hasMore && nextCursor !== null) ||
    (expected.authorityId && gatewayId !== expected.authorityId) ||
    (expected.authorityEpoch && epoch !== expected.authorityEpoch)
  ) {
    throw new Error('Invalid shared-files page')
  }

  const items = rawItems.map(item => parseItem(item, snapshotSeq))
  const ids = new Set(items.map(item => item.attachment.attachmentId))

  if (ids.size !== items.length) {
    throw new Error('Invalid shared-files duplicate')
  }

  for (let index = 1; index < items.length; index += 1) {
    const previous = items[index - 1]
    const current = items[index]

    if (
      current.seq > previous.seq ||
      (current.seq === previous.seq && current.attachment.attachmentId! <= previous.attachment.attachmentId!)
    ) {
      throw new Error('Invalid shared-files order')
    }
  }

  return { authority: { epoch, gatewayId }, hasMore, items, nextCursor, snapshotSeq }
}

export function validateGroupFilesContinuation(previous: GroupFilesPage, next: GroupFilesPage) {
  const last = previous.items.at(-1)
  const first = next.items[0]

  if (
    next.snapshotSeq !== previous.snapshotSeq ||
    next.authority.gatewayId !== previous.authority.gatewayId ||
    next.authority.epoch !== previous.authority.epoch ||
    next.nextCursor === previous.nextCursor ||
    (last &&
      first &&
      (first.seq > last.seq ||
        (first.seq === last.seq && first.attachment.attachmentId! <= last.attachment.attachmentId!)))
  ) {
    throw new Error('Invalid shared-files continuation')
  }
}

export async function listHostedGroupFiles(group: string, input: GroupFilesListInput = {}): Promise<GroupFilesPage> {
  const room: GroupChat | undefined = $groupChats.get()[group]
  const roomId = String(room?.roomId || '')
  const authorityId = groupChatHostedGateway(room)
  const limitInput = input.limit ?? GROUP_FILES_PAGE_SIZE

  if (!Number.isSafeInteger(limitInput) || limitInput < 1) {
    throw new Error('Invalid shared-files page size')
  }

  if (input.cursor !== undefined) {
    requiredText(input.cursor, 'cursor', MAX_CURSOR_LENGTH)
  }

  if (
    input.query !== undefined &&
    (typeof input.query !== 'string' || [...input.query].length > GROUP_FILES_MAX_QUERY_LENGTH)
  ) {
    throw new Error('Invalid shared-files query')
  }

  const route = room ? await boundedRequest(hostedRouteForRoom(room)) : null
  const limit = Math.min(GROUP_FILES_MAX_PAGE_SIZE, limitInput)
  const currentRoom = $groupChats.get()[group]

  if (
    !room ||
    !roomId ||
    !authorityId ||
    !route ||
    currentRoom?.roomId !== roomId ||
    groupChatHostedGateway(currentRoom) !== authorityId ||
    currentRoom?.hostedEpoch !== room.hostedEpoch
  ) {
    throw new Error('Shared files are unavailable.')
  }

  const response = await boundedRequest(
    requestHostedConnection(route, 'groups.attachment.list', {
      room_id: roomId,
      purpose: 'viewer',
      limit,
      ...(input.cursor ? { cursor: input.cursor } : {}),
      ...(input.query ? { query: input.query } : {})
    })
  )

  return parseGroupFilesPage(response, {
    authorityEpoch: room.hostedEpoch,
    authorityId,
    limit
  })
}
