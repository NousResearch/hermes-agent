/** Desktop byte staging and retrieval for capable hosted Group Chats.
 *
 * Ported from David Dudok de Wit's upstream
 * `apps/desktop/src/plugins/hermes-bots/hosted-room-attachments-client.ts`
 * (a38d59062deaaa31961b7f7c54f9f3c5e1b1bb75, PR #98307). Adapted only where this branch lacks
 * a dependency: upstream's `GroupFileError` verification class has no local counterpart, so a
 * plain `Error` carries the same message. The caps below are the gateway's own
 * (`gateway/hosted_room_attachments.py`), not client guesses.
 */

import type { Attachment, ProfileRoute } from './types'

type RequestHostedConnection = <T>(route: ProfileRoute, method: string, params?: Record<string, unknown>) => Promise<T>

const MAX_ATTACHMENT_BYTES = 15_000_000
const MAX_ATTACHMENT_TOTAL_BYTES = 25_000_000
const MAX_ATTACHMENTS = 8
const MAX_BASE64_CHARS = Math.ceil(MAX_ATTACHMENT_BYTES / 3) * 4

function validBase64(content: string): boolean {
  if (content.length % 4 !== 0) {
    return false
  }

  const padding = content.endsWith('==') ? 2 : content.endsWith('=') ? 1 : 0
  const end = content.length - padding

  // A repeated regex group grows V8's regex stack on valid multi-megabyte
  // receipts. This scan uses constant stack space up to the existing 15 MB cap.
  for (let index = 0; index < end; index += 1) {
    const code = content.charCodeAt(index)

    if (!(
      (code >= 65 && code <= 90) ||
      (code >= 97 && code <= 122) ||
      (code >= 48 && code <= 57) ||
      code === 43 ||
      code === 47
    )) {
      return false
    }
  }

  return end >= padding
}

function record(value: unknown): null | Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function decodedSize(contentBase64: string): number {
  return Math.max(
    0,
    Math.floor((contentBase64.length * 3) / 4) -
      (contentBase64.endsWith('==') ? 2 : contentBase64.endsWith('=') ? 1 : 0)
  )
}

function stagedAttachmentInput(attachment: Attachment) {
  // Match Python _name: strip Unicode whitespace, count code points, reject lone surrogates.
  // eslint-disable-next-line no-control-regex -- Python str.strip includes these four separators.
  const name = (attachment.name ?? 'attachment').replace(/^[\p{White_Space}\u001c-\u001f]+|[\p{White_Space}\u001c-\u001f]+$/gu, '')

  if (!name || [...name].length > 255 || name === '.' || name === '..' || name.includes('\0') || /[/\\\n\r\p{Surrogate}]/u.test(name)) {
    throw new Error('Attachment name must be a valid basename of at most 255 characters.')
  }

  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(String(attachment.data || ''))

  if (!match) {
    throw new Error(`${attachment.name || 'Attachment'} is no longer available on this Desktop.`)
  }

  const contentBase64 = match[2].replace(/\s+/g, '')
  const byteSize = decodedSize(contentBase64)

  if (!validBase64(contentBase64) || match[1].length > 127 ||
      !/^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/i.test(match[1])) {
    throw new Error(`${attachment.name || 'Attachment'} has invalid file data.`)
  }

  if (!contentBase64 || contentBase64.length > MAX_BASE64_CHARS || byteSize > MAX_ATTACHMENT_BYTES) {
    throw new Error(`${attachment.name || 'Attachment'} exceeds the 15MB Group Chat limit.`)
  }

  return {
    byteSize,
    content_base64: contentBase64,
    kind: attachment.kind,
    mime: match[1].toLowerCase(),
    name
  }
}

/** Reject locally invalid input before it becomes an immutable pending send. */
export function prepareHostedMessageAttachments(attachments: Attachment[]) {
  if (attachments.length > MAX_ATTACHMENTS) {
    throw new Error(`A Group Chat message can contain at most ${MAX_ATTACHMENTS} attachments.`)
  }

  const prepared = attachments.map(stagedAttachmentInput)

  if (prepared.reduce((total, attachment) => total + attachment.byteSize, 0) > MAX_ATTACHMENT_TOTAL_BYTES) {
    throw new Error('Group Chat attachments exceed the 25MB message limit.')
  }

  return prepared
}

/** Stage bytes and return the canonical manifest `groups.send` commits.
 *
 * Each attachment keeps its `uploadId` so a retry of an uncertain send stages the same upload
 * again instead of minting a second one. */
export async function stageHostedMessageAttachments(
  request: RequestHostedConnection,
  route: ProfileRoute,
  roomId: string,
  attachments: Attachment[]
) {
  const prepared = prepareHostedMessageAttachments(attachments)

  const manifest: Array<Record<string, unknown>> = []

  for (const [index, attachment] of attachments.entries()) {
    const { byteSize: _byteSize, ...input } = prepared[index]

    const uploadId =
      attachment.uploadId || globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`

    attachment.uploadId = uploadId

    const staged = record(
      await request<Record<string, unknown>>(route, 'groups.attachment.put', {
        room_id: roomId,
        upload_id: `desktop:${uploadId}`,
        ...input
      })
    )

    const value = record(staged?.attachment)
    const attachmentId = String(value?.attachment_id || '')
    // Read, never coerced: upstream's `Number(value?.size)` turns `true` into 1 and `"12"`
    // into 12, so a receipt with the wrong type could match the staged byte count.
    const stagedSize = value?.size

    if (
      !/^att_[0-9a-f]{32}$/.test(attachmentId) ||
      value?.kind !== attachment.kind ||
      value?.mime !== input.mime ||
      value?.name !== input.name ||
      typeof stagedSize !== 'number' ||
      !Number.isSafeInteger(stagedSize) ||
      stagedSize !== prepared[index].byteSize
    ) {
      throw new Error('The Group Chat host returned an invalid attachment receipt.')
    }

    attachment.attachmentId = attachmentId
    attachment.mime = input.mime
    attachment.size = stagedSize

    manifest.push({
      attachment_id: attachmentId,
      kind: String(value?.kind || ''),
      mime: String(value?.mime || ''),
      name: String(value?.name || ''),
      size: stagedSize
    })
  }

  return manifest
}

/** Fetch one committed attachment's bytes, verified against the metadata already in the log. */
export async function readHostedMessageAttachment(
  request: RequestHostedConnection,
  route: ProfileRoute,
  roomId: string,
  eventId: string,
  attachment: Attachment
): Promise<Attachment> {
  const attachmentId = String(attachment.attachmentId || '')

  if (!/^att_[0-9a-f]{32}$/.test(attachmentId) || !eventId) {
    throw new Error('This Group Chat attachment is unavailable.')
  }

  const response = record(
    await request<Record<string, unknown>>(route, 'groups.attachment.read', {
      attachment_id: attachmentId,
      event_id: eventId,
      purpose: 'viewer',
      room_id: roomId
    })
  )

  const receipt = record(response?.attachment)
  const contentBase64 = typeof response?.content_base64 === 'string' ? response.content_base64 : ''
  const mime = typeof receipt?.mime === 'string' ? receipt.mime : ''
  const receiptName = typeof receipt?.name === 'string' ? receipt.name : ''
  const receiptSize = receipt?.size

  if (
    String(receipt?.attachment_id || '') !== attachmentId ||
    typeof response?.content_base64 !== 'string' ||
    !receiptName ||
    // Kind decides how these bytes are rendered and staged, so it is verified like the rest
    // of the manifest rather than trusted from the log copy alone.
    receipt?.kind !== attachment.kind ||
    (attachment.name !== undefined && receiptName !== attachment.name) ||
    contentBase64.length > MAX_BASE64_CHARS ||
    !validBase64(contentBase64) ||
    !/^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/i.test(mime) ||
    (attachment.mime !== undefined && mime !== attachment.mime) ||
    typeof receiptSize !== 'number' ||
    !Number.isSafeInteger(receiptSize) ||
    receiptSize < 0 ||
    receiptSize > MAX_ATTACHMENT_BYTES ||
    decodedSize(contentBase64) !== receiptSize ||
    (attachment.size !== undefined && receiptSize !== attachment.size)
  ) {
    // Upstream raises its GroupFileError('verification', …); this branch has no such class.
    throw new Error('The Group Chat host returned an invalid attachment.')
  }

  return {
    ...attachment,
    data: `data:${mime};base64,${contentBase64}`,
    mime,
    name: receiptName
  }
}
