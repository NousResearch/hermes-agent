/**
 * Plugin door into the preview Browser's guest page.
 *
 * drive_preview and the preview tour already reach the guest through
 * `activePreviewScriptRunner` (the pane registers webview.executeJavaScript).
 * Plugins had no supported equivalent, so authors scraped `<webview>` out of
 * the app DOM — which the catalog `desktop surface` rule exists to stop.
 * This module is that equivalent: the focused session's live Browser tab, in
 * this window. A file peek, an empty rail, or a Browser popped out into
 * another window have no runner here.
 */
import { activePreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'

export const PREVIEW_NOTHING_OPEN = 'No preview page is open.'

/** True when this window has a live guest page for the focused session. */
export function previewIsAvailable(): boolean {
  return Boolean(activePreviewScriptRunner())
}

/**
 * Run `code` in the focused session's active preview Browser tab.
 * Always a Promise: a missing page and a guest throw both reject, so plugin
 * `.catch()` sees them the same way (host doors are async-safe).
 */
export function executePreviewJavaScript(code: string): Promise<unknown> {
  return (async () => {
    if (typeof code !== 'string') {
      throw new Error('host.preview.executeJavaScript requires a string')
    }

    const run = activePreviewScriptRunner()

    if (!run) {
      throw new Error(PREVIEW_NOTHING_OPEN)
    }

    return run(code)
  })()
}
