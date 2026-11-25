import { BROWSER_UPLOAD_MAX_BYTES } from '@/lib/browser-uploads'
import { isFreshDraftScope } from '@/store/composer'

import { type ComposerTarget, getVisibleComposerSurfaceId } from './focus'
import type { ChatBarProps } from './types'

// The opaque handle is renderer-local and can only be minted by captureDraft.
// Neither the caller nor a stale session id can redirect it to another draft.
declare const draftAddressBrand: unique symbol
export type ComposerDraftAddress = { readonly [draftAddressBrand]: true }

interface ImageRecipient {
  target: ComposerTarget
  sessionKey: null | string
  surfaceId: string
  capture: () => (() => boolean) | null
  attach: NonNullable<ChatBarProps['onAttachImageBlob']>
}

const recipients = new Map<string, ImageRecipient>()
const addresses = new WeakMap<ComposerDraftAddress, { recipient: ImageRecipient; isCurrent: () => boolean }>()

export function registerComposerImageRecipient(recipient: ImageRecipient): () => void {
  recipients.set(recipient.surfaceId, recipient)

  return () => {
    if (recipients.get(recipient.surfaceId) === recipient) {
      recipients.delete(recipient.surfaceId)
    }
  }
}

function matchesDraft({ sessionKey }: ImageRecipient, expected: 'new' | string[]): boolean {
  // The mounted draft key follows the route. The runtime ID can still name
  // the chat just left (for example when the gateway is disconnected).
  const fresh = sessionKey === null || isFreshDraftScope(sessionKey)

  return expected === 'new' ? fresh : !expected.length || (!fresh && expected.some(id => id === sessionKey))
}

export function captureComposerDraft(target: ComposerTarget, expected: 'new' | string[]): ComposerDraftAddress | null {
  const surfaceId = getVisibleComposerSurfaceId(target)
  const recipient = surfaceId ? recipients.get(surfaceId) : undefined

  if (!recipient || recipient.target !== target || !matchesDraft(recipient, expected)) {
    return null
  }

  const isCurrent = recipient.capture()

  if (!isCurrent) {
    return null
  }

  const address = Object.freeze({}) as ComposerDraftAddress
  addresses.set(address, { recipient, isCurrent })

  return address
}

const MAX_PNG_SIDE = 8192
const MAX_PNG_PIXELS = 24_000_000

/** Bound the raster before invoking Chromium's decoder. A valid signature or
 * IHDR alone does not establish that the pixels can actually be decoded. */
function pngDimensions(header: DataView): { width: number; height: number } | null {
  if (
    header.byteLength < 24 ||
    header.getUint32(0) !== 0x89504e47 ||
    header.getUint32(4) !== 0x0d0a1a0a ||
    header.getUint32(8) !== 13 ||
    header.getUint32(12) !== 0x49484452 // IHDR
  ) {
    return null
  }

  const width = header.getUint32(16)
  const height = header.getUint32(20)

  return width && height && width <= MAX_PNG_SIDE && height <= MAX_PNG_SIDE && width * height <= MAX_PNG_PIXELS
    ? { width, height }
    : null
}

export async function attachImageToComposerDraft(
  address: ComposerDraftAddress,
  blob: Blob,
  options?: { filename?: string }
): Promise<boolean> {
  const entry = addresses.get(address)

  if (
    !entry ||
    !(blob instanceof Blob) ||
    blob.type !== 'image/png' ||
    !blob.size ||
    blob.size > BROWSER_UPLOAD_MAX_BYTES
  ) {
    return false
  }

  const { recipient } = entry

  const isCurrent = () =>
    recipients.get(recipient.surfaceId) === recipient &&
    getVisibleComposerSurfaceId(recipient.target) === recipient.surfaceId &&
    entry.isCurrent()

  if (!isCurrent()) {
    return false
  }

  try {
    const dimensions = pngDimensions(new DataView(await blob.slice(0, 24).arrayBuffer()))

    if (!dimensions || !isCurrent()) {
      return false
    }

    const bitmap = await createImageBitmap(blob)
    const decodedAsExpected = bitmap.width === dimensions.width && bitmap.height === dimensions.height
    bitmap.close()

    if (!decodedAsExpected || !isCurrent()) {
      return false
    }

    const name = typeof options?.filename === 'string' ? options.filename.split(/[\\/]/).pop() : undefined

    const stem = name
      ?.replace(/\.[^.]+$/, '')
      .replace(/[^\p{L}\p{N}._-]+/gu, '_')
      .replace(/^[._-]+|[._-]+$/g, '')

    const image = new File([blob], `${stem?.slice(0, 80) || 'image'}.png`, { type: 'image/png' })

    // Native staging yields. The existing image ingestion path invokes this
    // guard again immediately before it mutates the composer's attachment set.
    return (await recipient.attach(image, isCurrent)) === true
  } catch {
    return false
  }
}
