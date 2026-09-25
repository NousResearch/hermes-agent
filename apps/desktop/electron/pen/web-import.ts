// Import from the web: a preview-pane page (or one element picked in it)
// lands on the pen canvas as editable layers.
//
// `@pen.dev/sdk/electron` is pen.dev's own capturer — the Chrome extension's
// crawler and picker, driven from the host process against a WebContents. We
// point it at the preview pane's <webview> guest (the renderer names the guest
// by webContents id, the same door `hermes:capturePreview` uses) and hand the
// payload to the editor over the embed bridge's `browser-import` request.
//
// One capturer per guest, created on first use and dropped when the guest
// goes away. Picker state and Enter/Cmd+Enter flow back to the window that
// hosts the guest so the preview strip can render the pick and run the import.

import { PenCapturer, type PenCapturerPickerState } from '@pen.dev/sdk/electron'
import { webContents as electronWebContents, ipcMain, type WebContents } from 'electron'

import { log } from './state'
import { importPenBrowserCapture } from './web-bridge'
import { ancestorSelector, type PenImportMode, type PenImportOptions, resolveImportMode } from './web-import-select'

interface GuestCapturer {
  capturer: PenCapturer
  /** The window renderer that owns the <webview>; picker events go there. */
  host: WebContents
  warnings: string[]
}

const capturers = new Map<number, GuestCapturer>()

export type { PenImportMode, PenImportOptions } from './web-import-select'

export interface PenImportResult {
  success: boolean
  imported?: PenImportMode
  /** DevTools-style label of the element that was imported, when one was. */
  element?: string
  /** Assets or screenshots the capture could not get; pen.dev shows them as import warnings. */
  warnings?: string[]
  error?: string
}

function sendToHost(entry: GuestCapturer, channel: string, payload: unknown): void {
  if (!entry.host.isDestroyed()) {
    entry.host.send(channel, payload)
  }
}

function guestFor(guestId: number): WebContents {
  const guest = electronWebContents.fromId(guestId)

  if (!guest || guest.isDestroyed()) {
    throw new Error('the preview page is gone')
  }

  return guest
}

function capturerFor(guestId: number, host: WebContents): GuestCapturer {
  const existing = capturers.get(guestId)

  if (existing) {
    existing.host = host

    return existing
  }

  const guest = guestFor(guestId)

  const entry: GuestCapturer = {
    capturer: new PenCapturer(guest, {
      onWarning: message => {
        entry.warnings.push(message)
        log.warn('pen import:', message)
      },
      screenshots: true
    }),
    host,
    warnings: []
  }

  entry.capturer.on('picker', (state: PenCapturerPickerState | undefined) => {
    sendToHost(entry, 'hermes:pen:import:picker', { guestId, state: state ?? null })
  })
  entry.capturer.on('action', action => {
    sendToHost(entry, 'hermes:pen:import:action', { action, guestId })
  })

  guest.once('destroyed', () => {
    entry.capturer.dispose()
    capturers.delete(guestId)
  })

  capturers.set(guestId, entry)

  return entry
}

function pickLabel(entry: GuestCapturer): string | undefined {
  const element = entry.capturer.picker?.pick?.element

  return element?.label ?? element?.selector ?? element?.tag
}

/** Capture per `options`, then hand the payload to the live canvas. */
export async function importIntoPenCanvas(
  guestId: number,
  host: WebContents,
  options: PenImportOptions = {}
): Promise<PenImportResult> {
  const entry = capturerFor(guestId, host)
  const { capturer } = entry
  const mode: PenImportMode = resolveImportMode(options)

  entry.warnings = []

  try {
    if (options.selector) {
      const pick = await capturer.select(options.selector)

      if (!pick) {
        return { error: `nothing on the page matches '${options.selector}'`, success: false }
      }
    } else if (mode === 'selection' && !capturer.picker?.pick) {
      return { error: 'nothing is picked — pick an element on the page first, or import the whole page', success: false }
    } else if (mode === 'page' && capturer.picker) {
      // The SDK captures the pick when one is live; a page import means none.
      await capturer.endPicking()
    }

    const element = mode === 'selection' ? pickLabel(entry) : undefined

    const payload = await capturer.capture({
      onProgress: fraction => sendToHost(entry, 'hermes:pen:import:progress', { fraction, guestId })
    })

    const { success } = await importPenBrowserCapture(payload)

    if (mode === 'selection') {
      await capturer.endPicking()
    }

    if (!success) {
      return { error: 'the canvas could not import the capture', success: false }
    }

    return {
      element,
      imported: mode,
      success: true,
      ...(entry.warnings.length ? { warnings: [...entry.warnings] } : {})
    }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), success: false }
  }
}

export function wirePenImportIpc(): void {
  ipcMain.handle('hermes:pen:import:pick', (event, guestId, active) => {
    const entry = capturerFor(Number(guestId), event.sender)

    if (active) {
      entry.capturer.startPicking({ pageZoom: guestFor(Number(guestId)).getZoomFactor() })

      return
    }

    return entry.capturer.endPicking()
  })

  // Breadcrumb hover: the path only carries labels, so climb from the picked
  // element's selector — `pathIndex - index` ancestors up.
  ipcMain.handle('hermes:pen:import:hover-path', (event, guestId, index) => {
    const { capturer } = capturerFor(Number(guestId), event.sender)
    const pick = capturer.picker?.pick

    const selector =
      pick?.element.selector && index !== null
        ? ancestorSelector(pick.element.selector, pick.pathIndex - Number(index))
        : undefined

    return capturer.hover(selector)
  })

  ipcMain.handle('hermes:pen:import:path', (event, guestId, index) =>
    capturerFor(Number(guestId), event.sender).capturer.selectPathEntry(Number(index))
  )

  ipcMain.handle('hermes:pen:import:run', (event, guestId, options) =>
    importIntoPenCanvas(Number(guestId), event.sender, options && typeof options === 'object' ? options : {})
  )
}
