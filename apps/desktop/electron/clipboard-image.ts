export interface ClipboardImageItem {
  readonly types: readonly string[]
  getType(type: string): Promise<unknown>
}

export interface DecodedClipboardImage {
  isEmpty(): boolean
  toPNG(): Buffer
}

export type DecodeClipboardImage = (buffer: Buffer) => DecodedClipboardImage

const SUPPORTED_CLIPBOARD_IMAGE_TYPES = ['image/png', 'image/jpeg'] as const

export async function readClipboardImageAsPng(
  items: readonly ClipboardImageItem[],
  decodeImage: DecodeClipboardImage
): Promise<Buffer | null> {
  for (const item of items) {
    for (const imageType of SUPPORTED_CLIPBOARD_IMAGE_TYPES) {
      if (!item.types.includes(imageType)) {
        continue
      }

      try {
        const value = await item.getType(imageType)

        if (!(value instanceof Blob)) {
          continue
        }

        const image = decodeImage(Buffer.from(await value.arrayBuffer()))

        if (!image.isEmpty()) {
          return image.toPNG()
        }
      } catch {
        // A clipboard can advertise stale or unsupported representations.
        // Try the next supported representation or ClipboardItem instead.
      }
    }
  }

  return null
}