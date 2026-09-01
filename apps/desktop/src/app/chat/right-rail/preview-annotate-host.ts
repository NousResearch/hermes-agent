/**
 * Drive the guest overlay and crop capture from the preview pane.
 * Overlay I/O only — stack/pack/flush live in lib/preview-annotate.
 */

import {
  annotateInPageSource,
  ANNOTATE_CROP_PAD,
  type AnnotatePageEvent,
  type AnnotatePinChrome
} from '@/lib/preview-annotate'

export interface PreviewAnnotateGuest {
  capture?: (rect: { height: number; width: number; x: number; y: number }) => Promise<string>
  executeJavaScript: (code: string) => Promise<unknown>
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

export async function captureAnnotateCrop(
  guest: PreviewAnnotateGuest,
  rect: AnnotatePinChrome['rect']
): Promise<string> {
  if (!guest.capture) {
    throw new Error('preview capture is unavailable')
  }

  return guest.capture(padRect(rect))
}
