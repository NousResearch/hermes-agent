/** Client for a gateway room's Files catalog (`groups.attachment.list`): pages of exact versions,
 *  and Save of one version's exact bytes through the existing download. */
import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'

export const FILES_PAGE_SIZE = 8
export const FILES_MAX_QUERY = 255
const LIST_BUDGET_MS = 10_000

/** One shared version: the message that shared it and its attachment id name it exactly. */
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
  /** False when the log still publishes this version but its bytes are gone from this gateway. */
  available: boolean
}

export interface CanonicalFilesPage {
  items: CanonicalFile[]
  nextCursor: null | string
  snapshotSeq: number
  authority: string
}

export type FilesFailure = 'access' | 'cursor' | 'error' | 'missing' | 'timeout' | 'unavailable' | 'verification'

export class CanonicalFilesError extends Error {
  constructor(readonly kind: FilesFailure) {
    super(kind)
  }
}

const REASONS: Record<string, FilesFailure> = {
  attachment_cursor_invalid: 'cursor',
  attachment_unavailable: 'missing',
  authority_conflict: 'access',
  permission_denied: 'access',
  profile_mismatch: 'access',
  room_history_expired: 'access',
  runtime_coordination_required: 'unavailable',
  storage_unavailable: 'unavailable'
}

/** The gateway names its refusal in `error.data.reason`; a server without the method answers -32601. */
export function filesFailure(error: unknown): FilesFailure {
  if (error instanceof CanonicalFilesError) {return error.kind}
  const failure = error as { code?: unknown; data?: { reason?: unknown } } | null
  const reason = String(failure?.data?.reason ?? '')

  if (Object.hasOwn(REASONS, reason)) {return REASONS[reason]}

  return failure?.code === -32601 ? 'unavailable' : 'error'
}

/** One request on one clock: `budgetMs` is the transport's own deadline for it, so a reply that
 *  arrives in time is always used, and a request the transport stopped waiting for is `timeout`.
 *  A retired read rejects with AbortError. */
async function filesRequest(
  binding: CanonicalGroupBinding, method: string, params: Record<string, unknown>, budgetMs: number, signal: AbortSignal
): Promise<unknown> {
  const started = Date.now()
  let abort: (() => void) | undefined

  try {
    return await Promise.race([canonicalGroupRequest<unknown>(binding, method, params, budgetMs), new Promise<never>((_resolve, reject) => {
      abort = () => reject(new DOMException('Cancelled', 'AbortError'))

      if (signal.aborted) {abort()} else {signal.addEventListener('abort', abort, { once: true })}
    })])
  } catch (error) {
    // The transport ends a request it got no answer to, at the budget, with an untyped error.
    if (!signal.aborted && Date.now() - started >= budgetMs) {throw new CanonicalFilesError('timeout')}

    throw error
  } finally {
    if (abort) {signal.removeEventListener('abort', abort)}
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0
const whole = (value: unknown, minimum = 0): value is number => Number.isSafeInteger(value) && (value as number) >= minimum

/** A finite server timestamp can still exceed the dates the UI can format. */
const timestamp = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(new Date(value * 1000).getTime())

function parseFile(value: unknown, snapshotSeq: number): CanonicalFile {
  const item = record(value)
  const sharer = record(item.producer)

  if (!text(item.event_id) || !text(item.attachment_id) || !whole(item.seq, 1) || item.seq > snapshotSeq ||
    !whole(item.manifest_index) || !text(item.kind) || !text(item.name) || !text(item.mime) || !whole(item.size, 1) ||
    (sharer.kind !== 'member' && sharer.kind !== 'user') || !text(sharer.id) || !text(sharer.label) ||
    !timestamp(item.shared_at) || (item.available !== undefined && typeof item.available !== 'boolean')) {
    throw new CanonicalFilesError('verification')
  }

  return { eventId: item.event_id, attachmentId: item.attachment_id, seq: item.seq, index: item.manifest_index,
    kind: item.kind, name: item.name, mime: item.mime, size: item.size,
    sharer: { kind: sharer.kind, id: sharer.id, label: sharer.label }, sharedAt: item.shared_at,
    available: item.available !== false }
}

/** True when `next` comes strictly after `previous` in newest-first, message-order listing. */
export function filesFollow(previous: CanonicalFile, next: CanonicalFile) {
  return next.seq < previous.seq || (next.seq === previous.seq && next.index > previous.index)
}

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

  if (items.some((item, index) => index > 0 && !filesFollow(items[index - 1], item))) {
    throw new CanonicalFilesError('verification')
  }

  return { items, nextCursor: cursor, snapshotSeq, authority: JSON.stringify([authority.gateway_id, authority.epoch]) }
}

/** A search also names the reader's word for the gateway's `You` (`you_label`), so searching your own
 *  shares by what Files shows finds them; a page without a search sends neither. */
export async function listCanonicalFiles(
  binding: CanonicalGroupBinding, input: { cursor?: string; query?: string; youLabel?: string }, signal: AbortSignal
): Promise<CanonicalFilesPage> {
  const response = await filesRequest(binding, 'groups.attachment.list', {
    room_id: binding.roomId,
    limit: FILES_PAGE_SIZE,
    ...(input.cursor ? { cursor: input.cursor } : {}),
    ...(input.query ? { query: input.query, ...(input.youLabel ? { you_label: input.youLabel } : {}) } : {})
  }, LIST_BUDGET_MS, signal)

  return parseFilesPage(response, binding.roomId)
}

/** Base64 into one preallocated buffer: mapping the decoded string through `Uint8Array.from` costs
 *  about 500 MiB and half a second for a 15 MB file, this about 20 MiB. */
function decodeBase64(encoded: string): Uint8Array<ArrayBuffer> {
  const binary = atob(encoded)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index++) {bytes[index] = binary.charCodeAt(index)}

  return bytes
}

export async function saveCanonicalFile(binding: CanonicalGroupBinding, file: CanonicalFile, signal: AbortSignal) {
  if (!file.available) {throw new CanonicalFilesError('missing')}

  // Gateway's 30 s plus the base64 body at 1 Mbit/s (125 bytes/ms).
  const budgetMs = 30_000 + Math.ceil(4 * Math.ceil(file.size / 3) / 125)
  const reply = record(await filesRequest(binding, 'groups.attachment.download', {
    room_id: binding.roomId, event_id: file.eventId, attachment_id: file.attachmentId
  }, budgetMs, signal))

  // The reply must be this version, and its encoding exactly this size, before anything is decoded.
  if (reply.event_id !== file.eventId || reply.attachment_id !== file.attachmentId || reply.name !== file.name ||
    reply.kind !== file.kind || reply.mime !== file.mime || reply.size !== file.size ||
    typeof reply.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(reply.sha256) ||
    typeof reply.data_base64 !== 'string' || reply.data_base64.length !== 4 * Math.ceil(file.size / 3)) {
    throw new CanonicalFilesError('verification')
  }

  let bytes: Uint8Array<ArrayBuffer>

  try {
    bytes = decodeBase64(reply.data_base64)
  } catch {
    throw new CanonicalFilesError('verification')
  }

  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  const hex = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('')

  if (bytes.length !== file.size || hex !== reply.sha256) {throw new CanonicalFilesError('verification')}

  if (signal.aborted) {throw new DOMException('Cancelled', 'AbortError')}
  downloadCanonicalAttachment(bytes, file.name, file.mime, signal)
}
