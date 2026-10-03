/**
 * Browser-hosted Desktop has no Electron `<webview>`, so a live web preview
 * mounts a sandboxed iframe instead. The frame is opaque to this renderer
 * (no console, history, page reader, or input channel); the pane only
 * navigates it.
 */

import type { RefObject } from 'react'

/** The navigation calls the pane makes on its live preview surface — an
 *  Electron webview or the browser-hosted iframe. */
export interface PreviewNavigationSurface {
  getTitle?: () => string
  getURL?: () => string
  loadURL?: (url: string) => Promise<void>
  reload?: () => void
}

export interface BrowserPreviewFrameOptions {
  host: HTMLElement
  navigationRef: RefObject<PreviewNavigationSurface | null>
  /** The frame reported a load failure for `url`. */
  onError: (url: string) => void
  setCurrentUrl: (url: string) => void
  setLoading: (loading: boolean) => void
  url: string
}

/** Mount the frame into `host` and publish it as the pane's navigation
 *  surface. Returns the mount effect's cleanup. */
export function mountBrowserPreviewFrame({
  host,
  navigationRef,
  onError,
  setCurrentUrl,
  setLoading,
  url
}: BrowserPreviewFrameOptions): () => void {
  const frame = document.createElement('iframe')
  frame.className = 'flex h-full w-full flex-1 border-0 bg-transparent'
  frame.referrerPolicy = 'no-referrer'
  frame.setAttribute('sandbox', 'allow-forms allow-scripts')
  // Browser-hosted previews deliberately grant no camera, microphone, or
  // clipboard capability. Fullscreen is the only delegated permission.
  frame.setAttribute('allow', 'fullscreen')
  frame.src = url

  const navigation: PreviewNavigationSurface = {
    getTitle: () => frame.title,
    getURL: () => frame.src,
    loadURL: async next => {
      frame.src = next
      setCurrentUrl(next)
      setLoading(true)
    },
    reload: () => {
      const current = frame.src
      frame.src = 'about:blank'
      frame.src = current
      setLoading(true)
    }
  }

  const onLoad = () => {
    setCurrentUrl(frame.src)
    setLoading(false)
  }

  const onFrameError = () => {
    onError(frame.src || url)
    setLoading(false)
  }

  frame.addEventListener('load', onLoad)
  frame.addEventListener('error', onFrameError)
  host.appendChild(frame)
  navigationRef.current = navigation

  return () => {
    if (navigationRef.current === navigation) {
      navigationRef.current = null
    }

    frame.removeEventListener('load', onLoad)
    frame.removeEventListener('error', onFrameError)
    frame.remove()
  }
}
