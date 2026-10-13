import type { BrowserWindow, IpcMain, WebContents, WebFrameMain } from 'electron'

const lookupGeneration = new WeakMap<WebContents, number>()

function isLookupPayload(payload: unknown): payload is { text: string; sessionId: string } {
  if (!payload || typeof payload !== 'object') {return false}
  const { text, sessionId } = payload as { text?: unknown; sessionId?: unknown }

  return (
    typeof text === 'string' &&
    !!text.trim() &&
    text.length <= 16_000 &&
    typeof sessionId === 'string' &&
    !!sessionId &&
    sessionId.length <= 256
  )
}

/** Native dictionary access remains sender-bound; the renderer never supplies executable code. */
export async function lookupChatSelection(
  window: BrowserWindow | null,
  frame: WebFrameMain | null,
  payload: unknown,
  isMac: boolean
): Promise<boolean> {
  if (
    !isMac ||
    !window ||
    window.isDestroyed() ||
    !frame ||
    frame !== window.webContents.mainFrame ||
    frame.isDestroyed()
  ) {
    return false
  }

  if (!isLookupPayload(payload)) {return false}
  const { text, sessionId } = payload
  const generation = (lookupGeneration.get(window.webContents) ?? 0) + 1
  lookupGeneration.set(window.webContents, generation)

  try {
    const authorized = await frame.executeJavaScript(`(() => {
      const selection = window.getSelection()
      const selector = '[data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]'
      const editable = 'input, textarea, [contenteditable]:not([contenteditable="false"])'
      const elementFor = node => node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement
      const anchor = elementFor(selection?.anchorNode)
      const focus = elementFor(selection?.focusNode)
      const message = anchor?.closest(selector)
      return !!selection && selection.rangeCount === 1 && !selection.isCollapsed &&
        selection.toString().trim() === ${JSON.stringify(text.trim())} &&
        !anchor?.closest(editable) && !focus?.closest(editable) &&
        !!message && message === focus?.closest(selector) &&
        message.closest('[data-selection-session-id]')?.getAttribute('data-selection-session-id') === ${JSON.stringify(sessionId)}
    })()`)

    if (
      authorized !== true ||
      lookupGeneration.get(window.webContents) !== generation ||
      window.isDestroyed() ||
      frame.isDestroyed() ||
      frame !== window.webContents.mainFrame
    ) {
      return false
    }

    window.webContents.showDefinitionForSelection()

    return true
  } catch {
    return false
  }
}

export function registerSelectionMenuIpc(
  ipc: Pick<IpcMain, 'handle'>,
  windowFor: (contents: WebContents) => BrowserWindow | null,
  isMac: boolean
): void {
  ipc.handle('hermes:context-menu:edit', (event, command) => {
    const actions = {
      copy: () => event.sender.copy(),
      cut: () => event.sender.cut(),
      paste: () => event.sender.paste(),
      selectAll: () => event.sender.selectAll()
    }

    if (typeof command === 'string' && Object.hasOwn(actions, command)) {
      actions[command as keyof typeof actions]()
    }
  })
  ipc.handle('hermes:context-menu:lookup', (event, payload) =>
    lookupChatSelection(windowFor(event.sender), event.senderFrame, payload, isMac)
  )
}
