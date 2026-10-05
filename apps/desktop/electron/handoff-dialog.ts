import type { MessageBoxOptions, MessageBoxReturnValue } from 'electron'

interface HandoffDialogWindow {
  isDestroyed(): boolean
  isVisible(): boolean
  once(event: string, listener: () => void): unknown
  removeListener(event: string, listener: () => void): unknown
}

export async function showHandoffDialog<Window extends HandoffDialogWindow>(
  window: Window | null,
  options: MessageBoxOptions,
  showMessageBox: (parent: Window, options: MessageBoxOptions) => Promise<MessageBoxReturnValue>
): Promise<MessageBoxReturnValue | null> {
  if (!window || window.isDestroyed()) {
    return null
  }

  // On macOS even the promise-based dialog blocks the main event loop without
  // a visible parent. Let boot continue while its window reaches first paint.
  if (!window.isVisible()) {
    const shown = await new Promise<boolean>(resolve => {
      const finish = (visible: boolean) => {
        window.removeListener('show', onShow)
        window.removeListener('closed', onClosed)
        resolve(visible)
      }

      const onShow = () => finish(true)
      const onClosed = () => finish(false)

      window.once('show', onShow)
      window.once('closed', onClosed)
    })

    if (!shown || window.isDestroyed()) {
      return null
    }
  }

  return showMessageBox(window, options)
}
