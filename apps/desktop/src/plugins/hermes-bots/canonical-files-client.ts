/** Client for a gateway room's Files catalog: list exact versions, save their exact bytes. */
import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'

export const FILES_PAGE_SIZE = 8
export const FILES_MAX_QUERY = 255
const FILES_DEADLINE_MS = 10_000

/** One shared version: the message that published it and its attachment id name it exactly. */
export interface CanonicalFile {
  eventId: string
  attachmentId: string
  seq: number
  index: number
  kind: string
  name: string
  mime: string
  size: number
  sharer: { kind: 'member' | 'user'; id: string; label: string }
  sharedAt: number
}

export interface CanonicalFilesPage {
  items: CanonicalFile[]
  nextCursor: null | string
  snapshotSeq: number
  authority: string
}

export type FilesFailure = 'access' | 'cursor' | 'error' | 'timeout' | 'unavailable' | 'verification'

export class CanonicalFilesError extends Error {
  constructor(readonly kind: FilesFailure) {
    super(kind)
  }
}

const REASONS: Record<string, FilesFailure> = {
  attachment_cursor_invalid: 'cursor',
  authority_conflict: 'access',
  permission_denied: 'access',
  profile_mismatch: 'access',
  runtime_coordination_required: 'unavailable'
}

export function filesFailure(error: unknown): FilesFailure {
  if (error instanceof CanonicalFilesError) {return error.kind}
  const failure = error as { code?: unknown; data?: { reason?: unknown } } | null
  const reason = String(failure?.data?.reason ?? '')

  if (Object.hasOwn(REASONS, reason)) {return REASONS[reason]}

  return failure?.code === -32601 ? 'unavailable' : 'error'
}

/** A closed dialog retires its requests; a gateway that never answers fails the request. */
async function withDeadline<T>(task: Promise<T>, signal: AbortSignal): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  let abort: (() => void) | undefined

  try {
    return await Promise.race([task, new Promise<never>((_resolve, reject) => {
      abort = () => reject(new DOMException('Cancelled', 'AbortError'))

      if (signal.aborted) {abort()} else {signal.addEventListener('abort', abort, { once: true })}
      timer = setTimeout(() => reject(new CanonicalFilesError('timeout')), FILES_DEADLINE_MS)
    })])
  } finally {
    clearTimeout(timer)

    if (abort) {signal.removeEventListener('abort', abort)}
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0
const whole = (value: unknown, minimum = 0): value is number => Number.isSafeInteger(value) && (value as number) >= minimum

function parseFile(value: unknown, snapshotSeq: number): CanonicalFile {
  const item = record(value)
  const sharer = record(item.producer)

  if (!text(item.event_id) || !text(item.attachment_id) || !whole(item.seq, 1) || item.seq > snapshotSeq ||
    !whole(item.manifest_index) || !text(item.kind) || !text(item.name) || !text(item.mime) || !whole(item.size, 1) ||
    (sharer.kind !== 'member' && sharer.kind !== 'user') || !text(sharer.id) || !text(sharer.label) ||
    typeof item.shared_at !== 'number' || !Number.isFinite(item.shared_at)) {
    throw new CanonicalFilesError('verification')
  }

  return { eventId: item.event_id, attachmentId: item.attachment_id, seq: item.seq, index: item.manifest_index,
    kind: item.kind, name: item.name, mime: item.mime, size: item.size,
    sharer: { kind: sharer.kind, id: sharer.id, label: sharer.label }, sharedAt: item.shared_at }
}

/** Newest share first, then message order; anything else means the page cannot be trusted. */
export function parseFilesPage(value: unknown, roomId: string): CanonicalFilesPage {
  const page = record(value)
  const authority = record(page.authority)
  const cursor = page.next_cursor === null || text(page.next_cursor) ? page.next_cursor : undefined

  if (page.room_id !== roomId || !text(authority.gateway_id) || !whole(authority.epoch, 1) ||
    !whole(page.snapshot_seq) || !Array.isArray(page.items) || page.items.length > FILES_PAGE_SIZE ||
    cursor === undefined || page.has_more !== (cursor !== null)) {
    throw new CanonicalFilesError('verification')
  }

  const snapshotSeq = page.snapshot_seq
  const items = page.items.map(item => parseFile(item, snapshotSeq))

  if (items.some((item, index) => index > 0 &&
    (item.seq > items[index - 1].seq || (item.seq === items[index - 1].seq && item.index <= items[index - 1].index)))) {
    throw new CanonicalFilesError('verification')
  }

  return { items, nextCursor: cursor, snapshotSeq, authority: JSON.stringify([authority.gateway_id, authority.epoch]) }
}

export async function listCanonicalFiles(
  binding: CanonicalGroupBinding, input: { cursor?: string; query?: string }, signal: AbortSignal
): Promise<CanonicalFilesPage> {
  const response = await withDeadline(canonicalGroupRequest<unknown>(binding, 'groups.attachment.list', {
    room_id: binding.roomId,
    limit: FILES_PAGE_SIZE,
    ...(input.cursor ? { cursor: input.cursor } : {}),
    ...(input.query ? { query: input.query } : {})
  }), signal)

  return parseFilesPage(response, binding.roomId)
}

/** Fetch one listed version with the existing download and save it only if the bytes are exactly it. */
export async function saveCanonicalFile(binding: CanonicalGroupBinding, file: CanonicalFile, signal: AbortSignal) {
  const reply = record(await withDeadline(canonicalGroupRequest<unknown>(binding, 'groups.attachment.download', {
    room_id: binding.roomId, event_id: file.eventId, attachment_id: file.attachmentId
  }), signal))

  if (reply.event_id !== file.eventId || reply.attachment_id !== file.attachmentId || reply.name !== file.name ||
    reply.kind !== file.kind || reply.mime !== file.mime || reply.size !== file.size ||
    typeof reply.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(reply.sha256) ||
    typeof reply.data_base64 !== 'string' || reply.data_base64.length !== 4 * Math.ceil(file.size / 3)) {
    throw new CanonicalFilesError('verification')
  }

  let bytes: Uint8Array<ArrayBuffer>

  try {
    bytes = Uint8Array.from(atob(reply.data_base64), char => char.charCodeAt(0))
  } catch {
    throw new CanonicalFilesError('verification')
  }

  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  const hex = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('')

  if (bytes.length !== file.size || hex !== reply.sha256) {throw new CanonicalFilesError('verification')}

  if (signal.aborted) {throw new DOMException('Cancelled', 'AbortError')}
  downloadCanonicalAttachment(bytes, file.name, file.mime, signal)
}
