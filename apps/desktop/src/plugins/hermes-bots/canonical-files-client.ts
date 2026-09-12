/** Canonical adapter for #104199's metadata-only Files catalog. */
import { downloadCanonicalAttachment } from './canonical-attachment-download'
import { type CanonicalGroupBinding, canonicalGroupRequest } from './canonical-groups'
import {
  GROUP_FILES_MAX_PAGE_SIZE,
  GROUP_FILES_MAX_QUERY_LENGTH,
  GROUP_FILES_PAGE_SIZE,
  type GroupFileItem,
  type GroupFilesListInput,
  type GroupFilesPage,
  parseGroupFilesPage
} from './group-files-parser'

export type FilesAuthority = NonNullable<GroupFilesPage['authority']>
export type FilesFailure =
  'access' | 'scope' | 'cursor' | 'gone' | 'verification' | 'timeout' | 'offline' | 'unavailable' | 'error'

export class CanonicalFilesError extends Error {
  constructor(readonly kind: FilesFailure) {
    super(kind)
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

export function canonicalFilesFailure(error: unknown): FilesFailure {
  if (error instanceof CanonicalFilesError) {
    return error.kind
  }

  const outer = record(error)
  const inner = record(outer.error)
  const message = String(outer.message || inner.message || '')
  const reason = String(record(outer.data).reason || record(inner.data).reason || message)

  const reasons: Record<string, FilesFailure> = {
    permission_denied: 'access',
    profile_mismatch: 'access',
    attachment_scope_changed: 'scope',
    attachment_cursor_invalid: 'cursor',
    attachment_not_found: 'gone',
    attachment_integrity_error: 'verification',
    runtime_coordination_required: 'unavailable',
    runtime_draining: 'offline'
  }

  if (Object.hasOwn(reasons, reason)) {
    return reasons[reason]
  }

  if ((outer.code ?? inner.code) === -32601) {
    return 'unavailable'
  }

  if (/timed? out|timeout/i.test(message)) {
    return 'timeout'
  }

  if (/offline|connection.*(?:closed|lost|unavailable|failed)|socket/i.test(message)) {
    return 'offline'
  }

  return 'error'
}

export function assertFilesIntent(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException('Cancelled', 'AbortError')
  }
}

// Source Files budget: cancellation/timeout retires the intent, not just its spinner.
export async function withFilesDeadline<T>(task: Promise<T>, signal?: AbortSignal): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  let abort: (() => void) | undefined

  try {
    return await Promise.race([
      task,
      new Promise<never>((_resolve, reject) => {
        abort = () => reject(new DOMException('Cancelled', 'AbortError'))

        if (signal?.aborted) {
          abort()

          return
        }

        signal?.addEventListener('abort', abort, { once: true })
        timer = setTimeout(() => reject(new CanonicalFilesError('timeout')), 10_000)
      })
    ])
  } finally {
    clearTimeout(timer)

    if (abort) {
      signal?.removeEventListener('abort', abort)
    }
  }
}

function origin(authority?: FilesAuthority) {
  return authority ? { authority_gateway_id: authority.gatewayId, authority_epoch: authority.epoch } : {}
}

export async function listCanonicalFiles(
  binding: CanonicalGroupBinding,
  input: GroupFilesListInput = {},
  authority?: FilesAuthority,
  signal?: AbortSignal
): Promise<GroupFilesPage> {
  assertFilesIntent(signal)
  const limitInput = input.limit ?? GROUP_FILES_PAGE_SIZE

  if (
    !Number.isSafeInteger(limitInput) ||
    limitInput < 1 ||
    (input.query !== undefined &&
      (typeof input.query !== 'string' || [...input.query].length > GROUP_FILES_MAX_QUERY_LENGTH)) ||
    (input.cursor !== undefined &&
      (typeof input.cursor !== 'string' ||
        !input.cursor.trim() ||
        new TextEncoder().encode(input.cursor).length > 4096))
  ) {
    throw new CanonicalFilesError('verification')
  }

  const limit = Math.min(limitInput, GROUP_FILES_MAX_PAGE_SIZE)

  const response = await withFilesDeadline(
    canonicalGroupRequest<unknown>(binding, 'groups.attachment.list', {
      room_id: binding.roomId,
      limit,
      ...origin(authority),
      ...(input.cursor ? { cursor: input.cursor } : {}),
      ...(input.query ? { query: input.query } : {})
    }),
    signal
  )

  assertFilesIntent(signal)

  if (record(response).room_id !== binding.roomId) {
    throw new CanonicalFilesError('scope')
  }

  let page: GroupFilesPage

  try {
    page = parseGroupFilesPage(response, { limit })
  } catch {
    throw new CanonicalFilesError('verification')
  }

  if (authority && (page.authority?.gatewayId !== authority.gatewayId || page.authority.epoch !== authority.epoch)) {
    throw new CanonicalFilesError('scope')
  }

  return page
}

export async function readCanonicalFile(
  binding: CanonicalGroupBinding,
  authority: FilesAuthority,
  item: GroupFileItem,
  signal?: AbortSignal
): Promise<Uint8Array<ArrayBuffer>> {
  assertFilesIntent(signal)

  const response = await withFilesDeadline(
    canonicalGroupRequest<unknown>(binding, 'groups.attachment.download', {
      room_id: binding.roomId,
      event_id: item.eventId,
      attachment_id: item.attachment.attachmentId,
      ...origin(authority)
    }),
    signal
  )

  assertFilesIntent(signal)
  const result = record(response)
  const receivedAuthority = record(result.authority)

  if (
    result.room_id !== binding.roomId ||
    receivedAuthority.gateway_id !== authority.gatewayId ||
    receivedAuthority.epoch !== authority.epoch
  ) {
    throw new CanonicalFilesError('scope')
  }

  const file = item.attachment

  if (
    result.event_id !== item.eventId ||
    result.attachment_id !== file.attachmentId ||
    result.name !== file.name ||
    result.mime !== file.mime ||
    result.kind !== file.kind ||
    result.size !== file.size ||
    typeof result.sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(result.sha256) ||
    typeof result.data_base64 !== 'string' ||
    result.data_base64.length !== 4 * Math.ceil(file.size / 3)
  ) {
    throw new CanonicalFilesError('verification')
  }

  let bytes: Uint8Array<ArrayBuffer>

  try {
    bytes = Uint8Array.from(atob(result.data_base64), char => char.charCodeAt(0))
  } catch {
    throw new CanonicalFilesError('verification')
  }

  if (bytes.length !== file.size) {
    throw new CanonicalFilesError('verification')
  }

  const digest = await crypto.subtle.digest('SHA-256', bytes)
  assertFilesIntent(signal)
  const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')

  if (hex !== result.sha256) {
    throw new CanonicalFilesError('verification')
  }

  return bytes
}

export async function saveCanonicalFile(
  binding: CanonicalGroupBinding,
  authority: FilesAuthority,
  item: GroupFileItem,
  signal?: AbortSignal
) {
  const bytes = await readCanonicalFile(binding, authority, item, signal)
  assertFilesIntent(signal)
  downloadCanonicalAttachment(bytes, item.attachment.name, item.attachment.mime, signal)
}
