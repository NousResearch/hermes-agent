/**
 * Selection-to-composer bridge module.
 *
 * Shared primitives for getting selections from any surface into the composer.
 * Provides two parallel facilities:
 *
 * 1. **Message quotes** — quoted text from a chat message, stored keyed by
 *    messageId and inserted as `@message:<messageId>` refs. Resolved at submit
 *    time into ```quote blocks via the same pattern as terminal selections.
 *    The `@message:` reference kind is registered in `reference-kinds.ts` and
 *    rendered by `directive-text.tsx`.
 *
 * 2. **Artifact image selection** — a Blob from a rendered artifact (chart,
 *    diagram, image response) that becomes a composer image attachment.
 *
 * The message-quote storage lives in `@/store/composer` (alongside
 * `$composerTerminalSelections`) so the submit pipeline can find it alongside
 * terminal selections and other draft-scoped state. This module is the
 * insertion/adapter layer that surfaces those primitives to UI components.
 */

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import {
  addComposerAttachment,
  clearComposerMessageQuotes,
  createComposerAttachmentOccurrenceId,
  messageQuoteContextBlocks,
  reconcileComposerMessageQuotes,
  setComposerMessageQuote,
  type ComposerAttachment,
  type $composerMessageQuotes
} from '@/store/composer'

// Re-export the message-quote store + helpers so callers only need to import
// from this module. The storage itself lives in composer.ts (the established
// home for draft-scoped selection state), matching $composerTerminalSelections.
export {
  $composerMessageQuotes,
  setComposerMessageQuote,
  reconcileComposerMessageQuotes,
  messageQuoteContextBlocks,
  clearComposerMessageQuotes
}

/**
 * Insert a "quote this message" ref into the composer.
 *
 * Stores the quoted text under the messageId (via {@link setComposerMessageQuote})
 * and inserts the `@message:<messageId>` ref text inline so the user can see
 * and remove it before sending.
 *
 * @param text - The quoted message text.
 * @param messageId - Stable identifier of the source message.
 * @param label - Optional display label for the ref chip. Defaults to `messageId`.
 */
export function addMessageSelectionToChat(text: string, messageId: string, label?: string): void {
  const trimmed = text.trim()
  const normalizedId = messageId.trim()
  const normalizedLabel = (label || normalizedId).trim()

  if (!trimmed || !normalizedId) {
    return
  }

  setComposerMessageQuote(normalizedId, trimmed)

  const refText = `@message:${formatRefValue(normalizedLabel)}`

  requestComposerInsert(refText, { mode: 'inline' })
}

/**
 * Add a Blob from an artifact (chart, diagram, rendered image response) as an
 * image attachment in the composer.
 *
 * Saves the blob to disk and creates a composer attachment with kind `'image'`.
 * The attachment carries the artifact id as metadata so the submit pipeline
 * can attribute the upload — callers should pass the artifact's stable id.
 *
 * @param blob - The image Blob from the rendered artifact.
 * @param artifactId - Stable identifier of the source artifact.
 * @param label - Optional display label for the attachment chip. Defaults to
 *   `artifactId`.
 *
 * @returns `true` when the attachment was successfully added, `false` on error
 *   (blob too small or disk write failure).
 */
export async function addArtifactImageSelectionToChat(
  blob: Blob,
  artifactId: string,
  label?: string
): Promise<boolean> {
  if (blob.size === 0) {
    return false
  }

  const normalizedId = artifactId.trim()
  const normalizedLabel = (label || normalizedId).trim()
  const ext = blobExtension(blob)

  try {
    const buffer = await blob.arrayBuffer()
    const data = new Uint8Array(buffer)
    const savedPath = await window.hermesDesktop?.saveImageBuffer(data, ext)

    if (!savedPath) {
      return false
    }

    const attachment: ComposerAttachment = {
      id: `artifact-image-${normalizedId}`,
      occurrenceId: createComposerAttachmentOccurrenceId(),
      kind: 'image',
      label: normalizedLabel,
      detail: normalizedId,
      path: savedPath
    }

    addComposerAttachment(attachment)

    return true
  } catch {
    return false
  }
}

/** MIME-type → file extension lookup (mirrors use-composer-actions.ts). */
const BLOB_MIME_EXTENSION: Record<string, string> = {
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/tiff': '.tiff',
  'image/webp': '.webp',
  'image/x-icon': '.ico'
}

/**
 * Derive a file extension from a Blob's MIME type.
 *
 * Mirrors the `blobExtension` helper in use-composer-actions.ts. Defaults to
 * `.png` when the MIME type is unrecognised.
 */
function blobExtension(blob: Blob): string {
  const mime = blob.type.split(';')[0].toLowerCase().trim()

  return BLOB_MIME_EXTENSION[mime] || '.png'
}