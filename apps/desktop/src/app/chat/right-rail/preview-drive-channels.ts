/**
 * The channels the agent acts on a live preview page through, published per
 * tab while the pane's surface is an Electron `<webview>`: the script runner
 * and the trusted-input channel (preview-act.ts drives the page with both).
 * The browser-hosted iframe is opaque to this renderer, so it publishes none.
 */

import { type RefObject, useEffect } from 'react'

import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'

import { type PreviewInputEvent, registerPreviewInput, toWebviewInputSpace } from './preview-input'
import { registerPreviewScriptRunner } from './preview-script-runner'

export type PreviewDriveGuest = HTMLElement & {
  executeJavaScript?: (code: string) => Promise<unknown>
  getZoomFactor?: () => number
  sendInputEvent?: (event: PreviewInputEvent) => void
}

export function usePreviewDriveChannels(
  webviewRef: RefObject<null | PreviewDriveGuest>,
  tabId: string | undefined,
  usesWebview: boolean
) {
  // Publish the SCRIPT runner for this tab: the one channel into the guest
  // page, shared by the tour tool (injected driver.js walkthroughs) and the
  // drive_preview tool (clicking, typing, scrolling the page the user sees).
  useEffect(() => {
    if (!usesWebview || !tabId) {
      return
    }

    return registerPreviewScriptRunner(tabId, async code => {
      const webview = webviewRef.current

      if (!webview?.executeJavaScript) {
        throw new Error('preview webview is not ready')
      }

      return webview.executeJavaScript(code)
    })
  }, [tabId, usesWebview, webviewRef])

  // Publish the INPUT channel for this tab. Same idea as the script runner, but
  // it carries real Chromium input rather than script — the agent's clicks and
  // keystrokes arrive as trusted events, so the page hovers, focuses and reacts
  // exactly as it would under a human hand.
  useEffect(() => {
    if (!usesWebview || !tabId) {
      return
    }

    return registerPreviewInput(tabId, {
      focus: () => {
        const webview = webviewRef.current

        // Trusted input still reaches the guest while hidden. Focusing the
        // webview element would steal the host's composer focus even when inert.
        if (webview && !isElementInHiddenPane(webview)) {
          webview.focus?.()
        }
      },
      send: event => {
        const webview = webviewRef.current

        // Never optional-chain this call away: a missing method would make every
        // agent click a silent no-op that still reports success, because the
        // overlay and the read-back both run on the separate script channel.
        if (typeof webview?.sendInputEvent !== 'function') {
          throw new Error('preview webview cannot take input events')
        }

        // The guest keeps its own (per-host) zoom, which the act engine's CSS
        // measurements do not include — ask the webview, not the window.
        webview.sendInputEvent(toWebviewInputSpace(event, webview.getZoomFactor?.()))
      }
    })
  }, [tabId, usesWebview, webviewRef])
}
