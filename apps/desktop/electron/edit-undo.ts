/**
 * Claim ⌘Z / ⌘⇧Z (and Ctrl+Y) in the main process so macOS cannot run the
 * native Edit-menu undo accelerator against Chromium's one-letter stack.
 *
 * Same hazard as ⌘W (electron#18295): a menu `{ role: 'undo' }` accelerator is
 * consumed before the page sees the keystroke. The rich composer owns its own
 * coalesced undo stack; native undo bypasses it. Menu items ship with no
 * accelerator and call the same IPC as this hook (#101309).
 *
 * Returns an uninstall fn.
 */

export type EditHistoryAction = 'redo' | 'undo'

export function isEditHistoryChord(
  input: Pick<Electron.Input, 'alt' | 'control' | 'key' | 'meta' | 'shift'>,
  isMac: boolean
): EditHistoryAction | null {
  const key = String(input.key || '').toLowerCase()
  const primary = isMac ? input.meta : input.control

  if (!primary || input.alt) {
    return null
  }

  if (key === 'z' && !input.shift) {
    return 'undo'
  }

  if (key === 'z' && input.shift) {
    return 'redo'
  }

  // Ctrl+Y redo on non-Mac (and when a Mac user holds Ctrl).
  if (key === 'y' && input.control && !input.meta && !input.shift) {
    return 'redo'
  }

  return null
}

const IS_MAC = () => process.platform === 'darwin'

export function installEditUndoShortcut(
  window: Electron.BrowserWindow,
  isMac: () => boolean = IS_MAC
): () => void {
  const { webContents } = window

  if (!webContents || webContents.isDestroyed()) {
    return () => {}
  }

  const handler = (event: Electron.Event, input: Electron.Input) => {
    if (!webContents || webContents.isDestroyed() || input.type !== 'keyDown') {
      return
    }

    const action = isEditHistoryChord(input, isMac())

    if (!action) {
      return
    }

    if (typeof event.preventDefault === 'function') {
      event.preventDefault()
    }

    webContents.send(action === 'undo' ? 'hermes:edit-undo' : 'hermes:edit-redo')
  }

  webContents.on('before-input-event', handler)

  return () => {
    if (!webContents.isDestroyed()) {
      webContents.off('before-input-event', handler)
    }
  }
}
