/**
 * The pen canvas pane body: a <webview> on hermes-pen://, chromeless.
 *
 * The webview fills the pane completely — pen's own toolbar/canvas/panels ARE
 * the pane content, with hermes contributing only the tree tab above it (which
 * carries the title, the close ✕, drag/split/stack). Host chrome inside the
 * editor page (hide pen's agent, blend background, UI scale) is injected by
 * the hermes-pen:// protocol handler in main, not here.
 */

import { useEffect, useRef } from 'react'

import { $canvasTabs } from './canvas-tile'

export function PenTilePane({ docId }: { docId: string }) {
  const hostRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const host = hostRef.current
    const tab = $canvasTabs.get().find(t => t.provider === 'pen' && t.docId === docId)

    if (!host || !tab) {
      return
    }

    // Imperative, not JSX: <webview> is an Electron-only element and creating
    // it imperatively (the preview pane's pattern) keeps React's types and
    // reconciler out of its lifecycle.
    const webview = document.createElement('webview')

    webview.setAttribute('src', tab.url)
    // The pen host preload is assigned by main's will-attach-webview hook
    // (keyed off the hermes-pen:// src), so no preload attribute here — the
    // renderer never learns filesystem paths.
    webview.style.cssText = 'width:100%;height:100%;border:0;background:transparent'

    host.append(webview)

    return () => {
      webview.remove()
    }
  }, [docId])

  return <div className="size-full min-h-0 min-w-0 bg-(--ui-editor-surface-background)" ref={hostRef} />
}
