import type { PreviewUblockController, PreviewUblockState } from './preview-ublock'

const POPUP_WIDTH = 380
const POPUP_HEIGHT = 600
const POPUP_MIN_WIDTH = 320
const POPUP_MIN_HEIGHT = 220
const POPUP_CONTENT_ALLOWANCE = 16
const POPUP_MARGIN = 16

const POPUP_MEASUREMENT_SCRIPT = `(() => {
  const values = [document.documentElement, document.body].filter(Boolean)
  const dimensions = values.flatMap(node => [
    node.scrollWidth,
    node.scrollHeight,
    node.offsetWidth,
    node.offsetHeight,
    node.getBoundingClientRect().width,
    node.getBoundingClientRect().height
  ])
  return {
    height: Math.max(...values.map(node => Math.max(node.scrollHeight, node.offsetHeight, node.getBoundingClientRect().height)), 0),
    width: Math.max(...values.map(node => Math.max(node.scrollWidth, node.offsetWidth, node.getBoundingClientRect().width)), 0),
    valid: dimensions.every(value => Number.isFinite(value) && value >= 0)
  }
})()`

export interface PreviewUblockPopupBounds {
  height: number
  width: number
  x: number
  y: number
}

export interface PreviewUblockPopupParent {
  getBounds(): PreviewUblockPopupBounds
  isDestroyed(): boolean
  once(event: 'closed' | 'destroyed', listener: () => void): unknown
  webContents?: {
    once(event: 'destroyed', listener: () => void): unknown
  }
}

export interface PreviewUblockPopupNavigationEvent {
  preventDefault(): void
}

export interface PreviewUblockPopupWindow {
  close(): void
  focus(): void
  isDestroyed(): boolean
  loadURL(url: string): Promise<void>
  on(event: 'blur' | 'closed' | 'destroyed', listener: () => void): unknown
  show(): void
  setContentSize(width: number, height: number): void
  setPosition(x: number, y: number): void
  webContents: {
    executeJavaScript(code: string): Promise<unknown>
    isDestroyed(): boolean
    on(event: 'will-navigate', listener: (event: PreviewUblockPopupNavigationEvent, url: string) => void): unknown
    on(event: 'destroyed', listener: () => void): unknown
    setWindowOpenHandler(handler: (details: { url: string }) => { action: 'allow' | 'deny' }): void
  }
}

export interface PreviewUblockPopupWindowOptions {
  height: number
  parent: PreviewUblockPopupParent
  resizable: false
  session: unknown
  show: false
  width: number
  x: number
  y: number
}

export interface PreviewUblockPopupManager {
  dispose(): void
  open(parent: PreviewUblockPopupParent): Promise<void>
}

interface PreviewUblockPopupRecord {
  parent: PreviewUblockPopupParent
  ready: Promise<void>
  window: PreviewUblockPopupWindow
}

interface PreviewUblockPopupManagerOptions {
  controller: Pick<PreviewUblockController, 'getState' | 'subscribe'>
  createWindow: (options: PreviewUblockPopupWindowOptions) => PreviewUblockPopupWindow
  log: (message: string) => void
  openExternal: (url: string) => boolean
  session: unknown
  getWorkArea?: (bounds: PreviewUblockPopupBounds) => PreviewUblockPopupBounds
}

function failedPopupError(): Error {
  return new Error('Preview uBlock popup could not be opened')
}

function isExtensionUrl(url: string, extensionId: string): boolean {
  try {
    const parsed = new URL(url)

    return parsed.protocol === 'chrome-extension:' && parsed.hostname === extensionId
  } catch {
    return false
  }
}

function isSafeExternalUrl(url: string): boolean {
  try {
    return ['http:', 'https:', 'mailto:'].includes(new URL(url).protocol)
  } catch {
    return false
  }
}

function isUsablePopupUrl(state: PreviewUblockState): state is PreviewUblockState & { popupUrl: string } {
  if (!state.enabled || !state.available || !state.rulesetsReady || !state.extensionId || !state.popupUrl) {
    return false
  }

  return isExtensionUrl(state.popupUrl, state.extensionId)
}

function popupPosition(
  bounds: PreviewUblockPopupBounds,
  size: { height: number; width: number },
  workArea: PreviewUblockPopupBounds
): { x: number; y: number } {
  const right = Math.max(workArea.x + POPUP_MARGIN, workArea.x + workArea.width - size.width - POPUP_MARGIN)
  const bottom = Math.max(workArea.y + POPUP_MARGIN, workArea.y + workArea.height - size.height - POPUP_MARGIN)
  const preferredY = Math.max(workArea.y + POPUP_MARGIN, bounds.y + POPUP_MARGIN + 32)

  return {
    x: right,
    y: Math.min(preferredY, bottom)
  }
}

function finiteDimension(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

function popupSize(measurement: unknown, workArea: PreviewUblockPopupBounds): { height: number; width: number } {
  const candidate = measurement as { height?: unknown; valid?: unknown; width?: unknown } | null
  const width = finiteDimension(candidate?.width)
  const height = finiteDimension(candidate?.height)

  if (candidate?.valid !== true || width === null || height === null) {
    return { height: POPUP_HEIGHT, width: POPUP_WIDTH }
  }

  return {
    height: Math.min(
      Math.max(POPUP_MIN_HEIGHT, Math.ceil(height + POPUP_CONTENT_ALLOWANCE)),
      Math.max(POPUP_MIN_HEIGHT, workArea.height - POPUP_MARGIN * 2)
    ),
    width: Math.min(
      Math.max(POPUP_MIN_WIDTH, Math.ceil(width + POPUP_CONTENT_ALLOWANCE)),
      Math.max(POPUP_MIN_WIDTH, workArea.width - POPUP_MARGIN * 2)
    )
  }
}

export function createPreviewUblockPopupManager({
  controller,
  createWindow,
  log,
  openExternal,
  session,
  getWorkArea = bounds => bounds
}: PreviewUblockPopupManagerOptions): PreviewUblockPopupManager {
  const records = new Map<PreviewUblockPopupParent, PreviewUblockPopupRecord>()
  let disposed = false

  const forget = (record: PreviewUblockPopupRecord, close: boolean): void => {
    if (records.get(record.parent) === record) {
      records.delete(record.parent)
    }

    if (close && !record.window.isDestroyed()) {
      try {
        record.window.close()
      } catch {
        // Destruction races are expected during app/window teardown.
      }
    }
  }

  const closeAll = (): void => {
    for (const record of records.values()) {
      forget(record, true)
    }
    records.clear()
  }

  const unsubscribe = controller.subscribe(state => {
    if (!isUsablePopupUrl(state)) {
      closeAll()
    }
  })

  const open = async (parent: PreviewUblockPopupParent): Promise<void> => {
    if (disposed || parent.isDestroyed()) {
      throw failedPopupError()
    }

    const existing = records.get(parent)
    if (existing && !existing.window.isDestroyed()) {
      try {
        await existing.ready
        if (!existing.window.isDestroyed() && !existing.window.webContents.isDestroyed()) {
          existing.window.focus()
          return
        }
      } catch {
        // The failed record is removed below and a retry may create a fresh one.
      }
      forget(existing, true)
    }

    const state = controller.getState()
    if (!isUsablePopupUrl(state)) {
      throw failedPopupError()
    }

    let popup: PreviewUblockPopupWindow
    try {
      const parentBounds = parent.getBounds()
      const workArea = getWorkArea(parentBounds)
      const position = popupPosition(parentBounds, { height: POPUP_HEIGHT, width: POPUP_WIDTH }, workArea)
      popup = createWindow({
        height: POPUP_HEIGHT,
        parent,
        resizable: false,
        session,
        show: false,
        width: POPUP_WIDTH,
        x: position.x,
        y: position.y
      })
    } catch {
      log('[preview] uBlock popup creation failed')
      throw failedPopupError()
    }

    const record: PreviewUblockPopupRecord = {
      parent,
      ready: Promise.resolve(),
      window: popup
    }
    records.set(parent, record)

    const cleanup = () => forget(record, false)
    parent.once('closed', () => forget(record, true))
    parent.once('destroyed', () => forget(record, true))
    parent.webContents?.once('destroyed', () => forget(record, true))
    popup.on('closed', cleanup)
    popup.on('destroyed', cleanup)
    popup.webContents.on('destroyed', cleanup)
    // The native window is intentionally temporary, like a browser action
    // popup. Register this only after the initial load/show/focus sequence so
    // a creation-time focus transition cannot reject a successful open.
    popup.webContents.setWindowOpenHandler(details => {
      // Extension JavaScript never owns native-window lifecycle here. Even
      // same-extension URLs are denied; the manager's single child window is
      // the only popup surface, while same-window extension navigation below
      // remains allowed.
      if (isSafeExternalUrl(details.url)) {
        try {
          openExternal(details.url)
        } catch {
          log('[preview] uBlock popup external link failed')
        }
      }

      return { action: 'deny' }
    })
    popup.webContents.on('will-navigate', (event, url) => {
      if (isExtensionUrl(url, state.extensionId!)) {
        return
      }

      event.preventDefault()
      if (isSafeExternalUrl(url)) {
        try {
          openExternal(url)
        } catch {
          log('[preview] uBlock popup external link failed')
        }
      }
    })

    record.ready = (async () => {
      try {
        await popup.loadURL(state.popupUrl)

        if (disposed || parent.isDestroyed() || popup.isDestroyed() || popup.webContents.isDestroyed()) {
          throw failedPopupError()
        }

        let size = { height: POPUP_HEIGHT, width: POPUP_WIDTH }
        try {
          const measured = await popup.webContents.executeJavaScript(POPUP_MEASUREMENT_SCRIPT)
          size = popupSize(measured, getWorkArea(parent.getBounds()))
        } catch {
          // Keep the provisional native size if page measurement races teardown.
        }

        if (!popup.isDestroyed() && !popup.webContents.isDestroyed() && !parent.isDestroyed()) {
          const workArea = getWorkArea(parent.getBounds())
          const position = popupPosition(parent.getBounds(), size, workArea)
          popup.setContentSize(size.width, size.height)
          popup.setPosition(position.x, position.y)
        }

        popup.show()
        popup.focus()
        popup.on('blur', () => forget(record, true))
      } catch {
        log('[preview] uBlock popup resource failed to load')
        forget(record, true)
        throw failedPopupError()
      }
    })()

    await record.ready
  }

  return {
    dispose() {
      disposed = true
      unsubscribe()
      closeAll()
    },
    open
  }
}

export const PREVIEW_UBLOCK_POPUP_SIZE = { height: POPUP_HEIGHT, width: POPUP_WIDTH } as const
