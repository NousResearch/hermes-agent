/**
 * Drive the guest overlay and crop capture from the preview pane.
 * Overlay I/O only — stack/pack/flush live in lib/preview-annotate.
 */

import {
  ANNOTATE_CROP_PAD,
  annotateInPageSource,
  type AnnotatePageEvent,
  type AnnotatePinChrome
} from '@/lib/preview-annotate'

import {
  type PreviewDriveGuest,
  runPreviewGuestScript,
  withPreviewDeadline,
  withPreviewGuestActivity
} from './preview-drive-channels'

export interface PreviewAnnotateGuest {
  capture?: (rect: { height: number; width: number; x: number; y: number }, signal?: AbortSignal) => Promise<string>
  executeJavaScript: (code: string) => Promise<unknown>
  /** Only bounded capture work is awake; waiting for the next user pick is not. */
  withActivity: <T>(
    operation: (execute: PreviewAnnotateGuest['executeJavaScript']) => Promise<T>,
    signal?: AbortSignal
  ) => Promise<T>
}

/**
 * Electron's `<webview>.executeJavaScript` reads `this.getWebContentsId()`.
 * Pulling the method off the element and calling it unbound throws
 * "Cannot read properties of undefined (reading 'getWebContentsId')".
 */
export function bindPreviewExecuteJavaScript(webview: {
  executeJavaScript?: (code: string) => Promise<unknown>
}): (code: string) => Promise<unknown> {
  return code => {
    if (typeof webview.executeJavaScript !== 'function') {
      return Promise.reject(new Error('preview webview is not ready'))
    }

    return webview.executeJavaScript(code)
  }
}

export function createPreviewAnnotateGuest(webview: PreviewDriveGuest | null): null | PreviewAnnotateGuest {
  if (!webview?.executeJavaScript) {
    return null
  }

  return {
    capture: async (rect, signal) => {
      const webContentsId = webview.getWebContentsId?.()

      if (typeof webContentsId !== 'number') {
        throw new Error('preview guest has no webContents')
      }

      const viewport = (await runPreviewGuestScript(
        webview,
        '({ width: window.innerWidth, height: window.innerHeight })',
        signal
      )) as { height: number; width: number }
      signal?.throwIfAborted()
      const dataUrl = await window.hermesDesktop.capturePreview?.({ rect, viewport, webContentsId })

      if (!dataUrl) {
        throw new Error('preview capture is unavailable')
      }

      return dataUrl
    },
    executeJavaScript: bindPreviewExecuteJavaScript(webview),
    withActivity: (operation, signal) =>
      withPreviewGuestActivity(webview, () => operation(code => runPreviewGuestScript(webview, code, signal)), signal)
  }
}

const padRect = (rect: { height: number; width: number; x: number; y: number }) => ({
  height: rect.height + ANNOTATE_CROP_PAD * 2,
  width: rect.width + ANNOTATE_CROP_PAD * 2,
  x: Math.max(0, rect.x - ANNOTATE_CROP_PAD),
  y: Math.max(0, rect.y - ANNOTATE_CROP_PAD)
})

/**
 * Build the guest install script by concatenation, never a template literal.
 * `annotateInPage.toString()` contains `${...}` (Vite leaves those in the
 * renderer). Wrapping that source in `` `...${source}...` `` evaluates those
 * interpolations in the host — `rect is not defined` — and Annotate looks
 * like a dead button.
 */
export function overlayInstallScript(source: string): string {
  return '(function(){var api=' + source + ';window.__hermesAnnotate=api;api.install();})()'
}

export async function installAnnotateOverlay(guest: PreviewAnnotateGuest): Promise<void> {
  await guest.executeJavaScript(overlayInstallScript(annotateInPageSource()))
}

export async function teardownAnnotateOverlay(guest: PreviewAnnotateGuest): Promise<void> {
  await guest.executeJavaScript(`
    (function () {
      if (window.__hermesAnnotate && window.__hermesAnnotate.teardown) {
        window.__hermesAnnotate.teardown();
      }
      window.__hermesAnnotate = null;
    })()
  `)
}

export async function waitAnnotateEvent(guest: PreviewAnnotateGuest): Promise<AnnotatePageEvent> {
  const event = await guest.executeJavaScript(`
    window.__hermesAnnotate ? window.__hermesAnnotate.wait() : Promise.resolve({ type: 'end' })
  `)

  return event as AnnotatePageEvent
}

export async function syncAnnotatePins(guest: PreviewAnnotateGuest, pins: AnnotatePinChrome[]): Promise<void> {
  await guest.executeJavaScript(`
    window.__hermesAnnotate && window.__hermesAnnotate.showPins(${JSON.stringify(pins)})
  `)
}

export async function showAnnotateDraft(
  guest: PreviewAnnotateGuest,
  rect: AnnotatePinChrome['rect'],
  number: number
): Promise<void> {
  await guest.executeJavaScript(`
    window.__hermesAnnotate && window.__hermesAnnotate.showDraft(${JSON.stringify(rect)}, ${number})
  `)
}

export async function hideAnnotateDraft(guest: PreviewAnnotateGuest): Promise<void> {
  await guest.executeJavaScript(`
    window.__hermesAnnotate && window.__hermesAnnotate.hideDraft()
  `)
}

/** Guest calls are best-effort dressing: never let one fail the capture. */
async function tryGuest(execute: PreviewAnnotateGuest['executeJavaScript'], code: string): Promise<void> {
  try {
    await execute(code)
  } catch {
    // The overlay may be mid-teardown or the guest gone. Shoot anyway.
  }
}

export async function captureAnnotateCrop(
  guest: PreviewAnnotateGuest,
  rect: AnnotatePinChrome['rect'],
  lifetime?: AbortSignal
): Promise<string> {
  if (!guest.capture) {
    throw new Error('preview capture is unavailable')
  }

  // Keep both paint frames, capturePage and cleanup inside one bounded lease.
  // Waiting for the next user pick must not hold this lease.
  return withPreviewDeadline(async signal => {
    const cleanup = '(function () { if (window.__hermesAnnotate) window.__hermesAnnotate.endCapture(); })()'

    try {
      return await guest.withActivity(async execute => {
        try {
          await tryGuest(execute, 'window.__hermesAnnotate ? window.__hermesAnnotate.beginCapture() : null')
          signal.throwIfAborted()

          return await guest.capture!(padRect(rect), signal)
        } finally {
          if (!signal.aborted) {
            await tryGuest(execute, cleanup)
          }
        }
      }, signal)
    } finally {
      // A deadline can interrupt native capture before its promise settles.
      // Restore chrome without acquiring another lease. Stop/navigation already
      // tears down the overlay, so never dress a replacement annotation session.
      if (signal.aborted && !lifetime?.aborted) {
        void guest.executeJavaScript(cleanup).catch(() => undefined)
      }
    }
  }, lifetime)
}
