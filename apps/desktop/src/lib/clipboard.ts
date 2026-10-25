import { isBrowserHostedDesktop } from '@/lib/platform'

// Primitives and CopyButton enter one policy, never nested fallback retries.
const nativeWrites = new WeakMap<Clipboard, Clipboard['writeText'] | undefined>()

// Electron writes through IPC first: the main process writes the clipboard even after the document
// loses focus (a portaled Radix menu), where the renderer's clipboard API refuses with "Write
// permission denied". The native API is the fallback. The browser-hosted Desktop has only the native
// API; its bridge's writeClipboard calls that same API.
export async function writeClipboardText(text: string) {
  if (!text) {
    return
  }

  const clipboard = navigator.clipboard

  const native =
    clipboard && (nativeWrites.has(clipboard) ? nativeWrites.get(clipboard) : clipboard.writeText?.bind(clipboard))

  const ipc = isBrowserHostedDesktop() ? undefined : window.hermesDesktop?.writeClipboard

  if (ipc) {
    try {
      if (await ipc(text)) {
        return
      }
    } catch (error) {
      if (!native) {
        throw error
      }
    }
  }

  if (!native) {
    throw new Error(ipc ? 'Clipboard write is unavailable' : 'Clipboard API is unavailable')
  }

  await native(text)
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
