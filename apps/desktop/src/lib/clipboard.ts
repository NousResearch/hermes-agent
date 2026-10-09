import { isBrowserHostedDesktop } from '@/lib/platform'

// Primitives and CopyButton enter one policy, never nested fallback retries.
const nativeWrites = new WeakMap<Clipboard, Clipboard['writeText'] | undefined>()

export async function writeClipboardText(text: string) {
  if (!text) {
    return
  }

  const clipboard = navigator.clipboard

  const native =
    clipboard && (nativeWrites.has(clipboard) ? nativeWrites.get(clipboard) : clipboard.writeText?.bind(clipboard))

  // The browser bridge calls the same native API, not an independent fallback.
  const ipc = isBrowserHostedDesktop() ? undefined : window.hermesDesktop?.writeClipboard

  if (native) {
    try {
      await native(text)

      return
    } catch (error) {
      if (!ipc) {
        throw error
      }
    }
  }

  if (!ipc) {
    throw new Error('Clipboard API is unavailable')
  }

  if (!(await ipc(text))) {
    throw new Error('Clipboard write is unavailable')
  }
}

export function installClipboardShim() {
  const clipboard = navigator.clipboard

  if (!window.hermesDesktop?.writeClipboard || !clipboard || nativeWrites.has(clipboard)) {
    return
  }

  nativeWrites.set(clipboard, clipboard.writeText?.bind(clipboard))

  try {
    Object.defineProperty(clipboard, 'writeText', { configurable: true, value: writeClipboardText, writable: true })
  } catch {
    // Browser refused override; primitives keep using the native API.
  }
}
