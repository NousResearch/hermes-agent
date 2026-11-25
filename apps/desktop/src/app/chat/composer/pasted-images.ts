/**
 * Attach the images a paste carried. A pasted image File is a reference to a
 * file on disk, readable only while that file still matches what the paste
 * captured: macOS Universal Clipboard (an image copied on an iPhone) hands
 * over a file it is still fetching, so the renderer read rejects with
 * NotReadableError (#159184). Main re-reads the clipboard's file URL instead,
 * and that read waits for the fetch.
 */

import type { ChatBarProps } from './types'

/** An in-memory copy of a pasted image, or null when its file cannot be read. */
async function readPastedImage(blob: Blob): Promise<Blob | null> {
  if (!(blob instanceof File)) {
    return blob
  }

  try {
    return new File([await blob.arrayBuffer()], blob.name, { lastModified: blob.lastModified, type: blob.type })
  } catch {
    return null
  }
}

export async function attachPastedImages(
  blobs: Blob[],
  attach: NonNullable<ChatBarProps['onAttachImageBlob']>,
  pasteFromMain: ChatBarProps['onPasteClipboardImage']
): Promise<void> {
  const images = await Promise.all(blobs.map(readPastedImage))

  for (const image of images) {
    if (image) {
      void attach(image)
    }
  }

  if (images.includes(null)) {
    void pasteFromMain?.()
  }
}
