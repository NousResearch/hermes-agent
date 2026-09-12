/** Pure catalog parser extracted from #104199 (03606761); no legacy session state. */

const ATTACHMENT_ID_RE = /^att_[0-9a-f]{32}$/
const IDENTIFIER_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/
const MIME_RE = /^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/i
const MAX_ATTACHMENT_BYTES = 15_000_000
const MAX_CURSOR_LENGTH = 4096
export const GROUP_FILES_PAGE_SIZE = 8
export const GROUP_FILES_MAX_PAGE_SIZE = 32
export const GROUP_FILES_MAX_QUERY_LENGTH = 255

export interface GroupFileAttachment {
  attachmentId: string
  kind: 'file' | 'image' | 'pdf'
  name: string
  mime: string
  size: number
}

export interface GroupFileProducer {
  identity: string
  kind: 'member' | 'user'
  label: string
}

export interface GroupFileItem {
  attachment: GroupFileAttachment
  eventId: string
  producer: GroupFileProducer
  seq: number
  sharedAt: number
  manifestIndex?: number
  key?: string
}

export interface GroupFilesPage {
  authority: { epoch: number; gatewayId: string } | null
  hasMore: boolean
  items: GroupFileItem[]
  nextCursor: null | string
  snapshotSeq: number
  latestFileSeq?: number
  localSnapshotKey?: string
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

  return (
    outer?.code === 4143 ||
    inner?.code === 4143 ||
    /attachment list cursor (?:is invalid|does not match this request|must be|is too large)/i.test(message)
  )
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

function cursorText(value: unknown) {
  const cursor = requiredText(value, 'cursor', MAX_CURSOR_LENGTH)

  if (new TextEncoder().encode(cursor).length > MAX_CURSOR_LENGTH) {
    throw new Error('Invalid shared-files cursor')
  }

  return cursor
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
  const manifestIndex = item?.manifest_index === undefined ? undefined : integer(item.manifest_index, 'manifest index')

  if (
    !ATTACHMENT_ID_RE.test(attachmentId) ||
    !IDENTIFIER_RE.test(eventId) ||
    !['file', 'image', 'pdf'].includes(kind) ||
    !MIME_RE.test(mime) ||
    size > MAX_ATTACHMENT_BYTES ||
    seq > snapshotSeq ||
    (manifestIndex !== undefined && manifestIndex > 7) ||
    typeof sharedAt !== 'number' ||
    !Number.isFinite(sharedAt) ||
    sharedAt <= 0 ||
    sharedAt > 8_640_000_000_000
  ) {
    throw new Error('Invalid shared-files item')
  }

  return {
    attachment: { attachmentId, kind: kind as GroupFileAttachment['kind'], mime, name, size },
    eventId,
    producer: parseProducer(item?.producer),
    seq,
    sharedAt,
    ...(manifestIndex === undefined ? {} : { manifestIndex })
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
  const nextCursor = rawCursor === null ? null : cursorText(rawCursor)

  const latestFileSeq =
    response?.latest_seq === undefined ? undefined : integer(response.latest_seq, 'latest file sequence')

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

  if (
    ids.size !== items.length ||
    (latestFileSeq !== undefined && items.some(item => item.seq > latestFileSeq)) ||
    (items.some(item => item.manifestIndex !== undefined) && items.some(item => item.manifestIndex === undefined))
  ) {
    throw new Error('Invalid shared-files duplicate')
  }

  for (let index = 1; index < items.length; index += 1) {
    const previous = items[index - 1]
    const current = items[index]

    if (compareGroupFiles(previous, current) >= 0) {
      throw new Error('Invalid shared-files order')
    }
  }

  return {
    authority: { epoch, gatewayId },
    hasMore,
    items,
    nextCursor,
    snapshotSeq,
    ...(latestFileSeq === undefined ? {} : { latestFileSeq })
  }
}

export function compareGroupFiles(left: GroupFileItem, right: GroupFileItem): number {
  return (
    right.seq - left.seq ||
    (left.manifestIndex !== undefined && right.manifestIndex !== undefined
      ? left.manifestIndex - right.manifestIndex
      : 0) ||
    (left.attachment.attachmentId! < right.attachment.attachmentId!
      ? -1
      : left.attachment.attachmentId === right.attachment.attachmentId
        ? 0
        : 1)
  )
}

export function validateGroupFilesContinuation(previous: GroupFilesPage, next: GroupFilesPage) {
  const last = previous.items.at(-1)
  const first = next.items[0]

  if (
    next.snapshotSeq !== previous.snapshotSeq ||
    next.authority?.gatewayId !== previous.authority?.gatewayId ||
    next.authority?.epoch !== previous.authority?.epoch ||
    next.localSnapshotKey !== previous.localSnapshotKey ||
    next.nextCursor === previous.nextCursor ||
    (last &&
      first &&
      previous.authority !== null &&
      (compareGroupFiles(last, first) >= 0 ||
        (last.manifestIndex === undefined) !== (first.manifestIndex === undefined)))
  ) {
    throw new Error('Invalid shared-files continuation')
  }
}
