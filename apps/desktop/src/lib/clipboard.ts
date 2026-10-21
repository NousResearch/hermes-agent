// Routes `navigator.clipboard.writeText` through Electron IPC, since the
// renderer's clipboard API throws "Write permission denied" whenever the
// document loses focus (e.g. clicking a portaled Radix dropdown). The IPC
// path runs in the main process and is unconditional.

export function installClipboardShim() {
  const ipc = window.hermesDesktop?.writeClipboard

  if (!ipc || !navigator.clipboard) {
    return
  }

  const native = navigator.clipboard.writeText?.bind(navigator.clipboard)

  const writeText = async (text: string) => {
    try {
      await ipc(text)
    } catch {
      await native?.(text)
    }
  }

  try {
    Object.defineProperty(navigator.clipboard, 'writeText', { configurable: true, value: writeText, writable: true })
  } catch {
    // Browser refused override; primitives keep using the native API.
  }
}
