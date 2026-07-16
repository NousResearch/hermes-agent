import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from 'react'

import {
  $browserOpen,
  $browserState,
  normalizeBrowserRuntimeUrl,
  setBrowserCapture,
  updateBrowserTab
} from '@/app/browser/store'

const $slot = atom<HTMLElement | null>(null)

const SLOT_CLASS = 'relative flex min-h-0 min-w-0 flex-1 flex-col'
const BROWSER_PARTITION = 'persist:hermes-browser'

interface BrowserWebview extends HTMLElement {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  getWebContentsId?: () => number
  goBack?: () => void
  goForward?: () => void
  reload?: () => void
}

interface Rect {
  top: number
  left: number
  width: number
  height: number
}

const sameRect = (a: Rect | null, b: Rect) =>
  !!a && a.top === b.top && a.left === b.left && a.width === b.width && a.height === b.height

const webviews = new Map<string, BrowserWebview>()
interface BrowserGuestProps {
  active: boolean
  tabId: string
  url: string
}

function BrowserGuest({ active, tabId, url }: BrowserGuestProps) {
  const ref = useRef<BrowserWebview | null>(null)

  useEffect(() => {
    const webview = ref.current

    if (!webview) {
      return
    }

    const updateUrl = (event: Event) => updateBrowserTab(tabId, { url: (event as Event & { url?: string }).url ?? '' })

    const updateTitle = (event: Event) =>
      updateBrowserTab(tabId, { title: (event as Event & { title?: string }).title ?? '' })

    webview.addEventListener('did-navigate', updateUrl)
    webview.addEventListener('did-navigate-in-page', updateUrl)
    webview.addEventListener('page-title-updated', updateTitle)

    return () => {
      webview.removeEventListener('did-navigate', updateUrl)
      webview.removeEventListener('did-navigate-in-page', updateUrl)
      webview.removeEventListener('page-title-updated', updateTitle)
    }
  }, [tabId])

  return (
    <webview
      className="size-full"
      partition={BROWSER_PARTITION}
      ref={element => {
        const webview = element as BrowserWebview | null
        ref.current = webview

        if (webview) {
          webviews.set(tabId, webview)
        } else {
          webviews.delete(tabId)
        }
      }}
      src={url}
      style={{ display: active ? 'flex' : 'none' }}
      webpreferences="contextIsolation=yes,nodeIntegration=no,sandbox=yes"
    />
  )
}

export function BrowserSlot({ className = SLOT_CLASS }: { className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const element = ref.current

    if (!element) {
      return
    }

    $slot.set(element)

    return () => {
      if ($slot.get() === element) {
        $slot.set(null)
      }
    }
  }, [])

  return <div className={className} ref={ref} />
}

export function browserBack(tabId: string) {
  const webview = webviews.get(tabId)

  if (webview?.canGoBack?.()) {
    webview.goBack?.()
  }
}

export function browserForward(tabId: string) {
  const webview = webviews.get(tabId)

  if (webview?.canGoForward?.()) {
    webview.goForward?.()
  }
}

export function browserReload(tabId: string) {
  webviews.get(tabId)?.reload?.()
}

export function browserNavigate(tabId: string, url: string) {
  updateBrowserTab(tabId, { url: normalizeBrowserRuntimeUrl(url) })
}

export async function captureBrowserTab(tabId: string) {
  const guestWebContentsId = webviews.get(tabId)?.getWebContentsId?.()

  if (typeof guestWebContentsId !== 'number') {
    throw new Error('Browser tab is unavailable for capture: guest id is missing.')
  }

  const bridge = window.hermesDesktop?.browser

  if (!bridge?.capture) {
    throw new Error('Browser capture is unavailable: desktop bridge is missing.')
  }

  const capture = await bridge.capture(guestWebContentsId)

  if (!capture) {
    throw new Error('Browser capture failed: capture result is missing.')
  }

  setBrowserCapture({ ...capture, tabId })

  return capture
}

export async function saveBrowserCapture(captureId: string, suggestedName?: string) {
  const bridge = window.hermesDesktop?.browser

  if (!bridge?.saveCapture) {
    throw new Error('Browser capture saving is unavailable: desktop bridge is missing.')
  }

  return bridge.saveCapture(captureId, suggestedName)
}

/**
 * Guest webviews remain in this root-level fixed overlay. Reparenting a webview
 * destroys guest state, so the overlay follows BrowserSlot geometry instead.
 */
export function PersistentBrowser() {
  const slot = useStore($slot)
  const browserOpen = useStore($browserOpen)
  const browserState = useStore($browserState)
  const [rect, setRect] = useState<Rect | null>(null)

  useLayoutEffect(() => {
    if (!slot) {
      setRect(null)

      return
    }

    let previous: Rect | null = null
    let frame = 0

    const tick = () => {
      const bounds = slot.getBoundingClientRect()
      const top = Math.floor(bounds.top)
      const left = Math.floor(bounds.left)
      const next = { top, left, width: Math.ceil(bounds.right) - left, height: Math.ceil(bounds.bottom) - top }

      if (!sameRect(previous, next)) {
        previous = next
        setRect(next)
      }

      frame = requestAnimationFrame(tick)
    }

    tick()

    return () => cancelAnimationFrame(frame)
  }, [slot])

  const visible = Boolean(browserOpen && rect && rect.width > 0 && rect.height > 0)

  const style: CSSProperties = {
    position: 'fixed',
    top: rect?.top ?? 0,
    left: rect?.left ?? 0,
    width: rect?.width ?? 0,
    height: rect?.height ?? 0,
    visibility: visible ? 'visible' : 'hidden',
    pointerEvents: visible ? 'auto' : 'none',
    zIndex: 4,
    contain: 'layout size paint'
  }

  return (
    <div aria-hidden={!visible} style={style}>
      {browserState.tabs
        .filter(tab => tab.url && tab.url !== 'about:blank')
        .map(tab => (
          <BrowserGuest active={tab.id === browserState.activeTabId} key={tab.id} tabId={tab.id} url={tab.url} />
        ))}
    </div>
  )
}
