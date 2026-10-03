import { IMAGE_EXTENSION_BY_MIME } from '@/lib/browser-uploads'

interface BrowserClipboardOptions {
  /** Stores clipboard image bytes where the agent can read them. */
  saveBuffer: (data: ArrayBuffer | Uint8Array, ext: string) => Promise<string>
}

export function createBrowserClipboardBridge({
  saveBuffer
}: BrowserClipboardOptions): Pick<Window['hermesDesktop'], 'readClipboard' | 'saveClipboardImage' | 'writeClipboard'> {
  // Bound at install: installClipboardShim later routes navigator.clipboard.writeText
  // through writeClipboard, so a late lookup would call itself.
  const nativeWriteClipboard = navigator.clipboard?.writeText?.bind(navigator.clipboard)

  return {
    readClipboard: async () => navigator.clipboard?.readText?.() || '',
    saveClipboardImage: async () => {
      const clipboard = navigator.clipboard as Clipboard & {
        read?: () => Promise<ClipboardItems>
      }

      if (!clipboard?.read) {return ''}

      const items = await clipboard.read()

      for (const item of items) {
        const mimeType = item.types.find(type => type.startsWith('image/'))

        if (!mimeType) {continue}

        const blob = await item.getType(mimeType)
        const extension = IMAGE_EXTENSION_BY_MIME[mimeType] || '.png'

        return saveBuffer(await blob.arrayBuffer(), extension)
      }

      return ''
    },
    writeClipboard: async (text: string) => {
      if (!nativeWriteClipboard) {return false}

      await nativeWriteClipboard(text)

      return true
    }
  }
}
