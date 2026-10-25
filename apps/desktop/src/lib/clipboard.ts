import { isBrowserHostedDesktop } from '@/lib/platform'

// Primitives and CopyButton enter one policy, never nested fallback retries.
const nativeWrites = new WeakMap<Clipboard, Clipboard['writeText'] | undefined>()

// Natively focusable elements (controls, links, embedded content): the textarea never goes inside one.
const FOCUSABLE_LEAVES = 'a, area, audio, button, embed, iframe, input, object, select, summary, textarea, video'

/** An insecure origin (a Webapp opened over plain HTTP from another machine) has no Clipboard API;
 *  the legacy copy command still works there, inside the click that asked for the copy. The
 *  textarea goes inside a focused element that holds focus through tabindex (a dialog, a menu or
 *  its item, maybe the trap's own root: Safari does not focus a clicked button), else beside the
 *  focused control or editor, so a dialog or menu that traps focus keeps it in scope and lets it
 *  take focus. True when the browser reports the copy done. */
export function copyTextThroughSelection(text: string): boolean {
  if (typeof document.execCommand !== 'function') {
    return false
  }

  const active = document.activeElement
  const focused = active instanceof HTMLElement && active !== document.body ? active : null
  const selection = document.getSelection()
  const ranges = Array.from({ length: selection?.rangeCount ?? 0 }, (_, index) => selection!.getRangeAt(index))
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;padding:0;border:0;opacity:0'
  const holds = focused && !focused.isContentEditable && !focused.matches(FOCUSABLE_LEAVES)
  const scope = (holds ? focused : focused?.parentElement) ?? document.body
  scope.append(area)
  area.focus({ preventScroll: true })
  area.select()
  area.setSelectionRange(0, text.length)

  try {
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    area.remove()
    selection?.removeAllRanges()
    ranges.forEach(range => selection?.addRange(range))
    focused?.focus({ preventScroll: true })
  }
}

// Electron writes through IPC first: the main process writes the clipboard even after the document
// loses focus (a portaled Radix menu), where the renderer's clipboard API refuses with "Write
// permission denied". The native API is the fallback. The browser-hosted Desktop has only the native
// API; its bridge's writeClipboard calls that same API. Without it, both copy through a selection.
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
    if (!ipc && copyTextThroughSelection(text)) {
      return
    }

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
