'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Maximize, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'

// Draw.io Renderer using the POSTMESSAGE pattern.
// Instead of using srcdoc and trying to boot the viewer via a script tag
// (which causes the XMLSerializer Node error), we load a real same-origin
// render.html and send the XML as a message.
let RENDER_URL = '/drawio/render.html'
let RENDER_VERSION = '6' // Bumped to force cache clear
const RENDER_TIMEOUT_MS = 15_000

export function setDrawioRenderUrl(url: string): void {
  RENDER_URL = url
}

function SourcePreview({ code, muted }: { code: string; muted?: boolean }) {
  return (
    <pre
      className={cn(
        'overflow-auto p-3 font-mono text-[0.7rem] leading-relaxed whitespace-pre-wrap wrap-anywhere',
        muted ? 'text-muted-foreground/70' : 'text-foreground/90'
      )}
    >
      {code}
    </pre>
  )
}

// Lazy chunk. Renders ```drawio fences by loading the draw.io viewer in a
// sandboxed iframe and sending the XML via postMessage. Shows the source while
// the message streams (partial XML throws) and falls back to source on failure.
//
// Two iframe modes (controlled by the postMessage mode field):
//   inline   — static diagram, no toolbar/nav/pan/zoom (in-chat display)
//   expanded — toolbar, minimap, drag-pan, wheel-zoom (full-view dialog)
//
// Clicking the expand button opens a dialog with the interactive viewer,
// matching the Excalidraw renderer's UX pattern.
export default function DrawioRenderer({ code, streaming }: RichFenceProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [failed, setFailed] = useState(false)
  const [ready, setReady] = useState(false)
  const [assetAvailable, setAssetAvailable] = useState<boolean | null>(null)
  const [expanded, setExpanded] = useState(false)

  // Check whether the render asset exists before mounting the iframe.
  // A missing asset (clean checkout, failed build, or stale deployment)
  // surfaces the source preview instead of a blank iframe.
  useEffect(() => {
    if (streaming) {return}
    let cancelled = false
    setAssetAvailable(null)
    setFailed(false)
    setReady(false)

    void (async () => {
      try {
        const res = await fetch(`${RENDER_URL}?v=${RENDER_VERSION}`, { method: 'HEAD' })

        if (cancelled) {return}
        setAssetAvailable(res.ok)

        if (!res.ok) {
          setFailed(true)
        }
      } catch {
        if (cancelled) {return}
        setAssetAvailable(false)
        setFailed(true)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [streaming])

  // Create the inline iframe (static mode — no toolbar/nav/pan/zoom)
  useEffect(() => {
    if (streaming || assetAvailable === false) {return}

    if (assetAvailable === null) {return}

    const host = hostRef.current

    if (!host) {return}

    setFailed(false)
    setReady(false)
    setExpanded(false)

    const iframe = document.createElement('iframe')
    iframe.className = 'h-full w-full border-0'
    iframe.setAttribute('sandbox', 'allow-scripts')
    iframe.src = `${RENDER_URL}?v=${RENDER_VERSION}`

    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const clear = () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
    }

    const handleMessage = (ev: MessageEvent) => {
      if (ev.source !== iframe.contentWindow) {return}

      if (!ev.data || typeof ev.data !== 'object') {return}

      if (ev.data.event === 'ready' || ev.data.event === 'init') {
        clear()
        setReady(true)
        iframe.contentWindow?.postMessage(
          { action: 'load', xml: code, mode: 'inline' },
          '*'
        )
      } else if (ev.data.event === 'error' || ev.data.error) {
        clear()
        setFailed(true)
      }
    }

    timeoutId = setTimeout(() => {
      setFailed(true)
    }, RENDER_TIMEOUT_MS)

    window.addEventListener('message', handleMessage)
    iframe.addEventListener('error', () => setFailed(true))

    host.replaceChildren(iframe)

    return () => {
      clear()
      window.removeEventListener('message', handleMessage)
      host.replaceChildren()
    }
  }, [code, streaming, assetAvailable])

  const openExpanded = useCallback(() => setExpanded(true), [])
  const closeExpanded = useCallback(() => setExpanded(false), [])

  if (streaming) {return <SourcePreview code={code} muted />}

  if (failed || assetAvailable === false) {return <SourcePreview code={code} />}

  return (
    <>
      <div className="group/zoomable relative">
        <div
          className="h-[33dvh] w-full overflow-hidden rounded-lg border border-border bg-muted/30"
          ref={hostRef}
        />
        {(!ready || assetAvailable === null) && <SourcePreview code={code} muted />}
        {ready && (
          <button
            className="pointer-events-auto absolute right-2 top-2 grid size-8 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground opacity-0 shadow-sm backdrop-blur transition-opacity group-hover/zoomable:opacity-100 hover:opacity-100"
            onClick={openExpanded}
            title="Open diagram"
            type="button"
          >
            <Maximize className="size-4" />
          </button>
        )}
      </div>
      {expanded && (
        <Dialog onOpenChange={setExpanded} open={expanded}>
          <DialogContent
            className="flex h-[85vh] w-[90vw] max-w-[90vw] flex-col gap-0 overflow-hidden p-0"
            showCloseButton={false}
          >
            <div className="relative flex-1 overflow-hidden">
              <ExpandedIframe code={code} onFailure={() => setFailed(true)} />
              <button
                className="absolute right-3 top-3 z-10 grid size-8 place-items-center rounded-full border border-border/70 bg-background/80 text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-accent hover:text-foreground"
                onClick={closeExpanded}
                title="Close"
                type="button"
              >
                <X className="size-4" />
              </button>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </>
  )
}

function ExpandedIframe({
  code,
  onFailure,
}: {
  code: string
  onFailure: () => void
}) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null)
  const [iframeFailed, setIframeFailed] = useState(false)

  useEffect(() => {
    const iframe = iframeRef.current

    if (!iframe) {return}

    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const clear = () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
    }

    const handleMessage = (ev: MessageEvent) => {
      if (ev.source !== iframe.contentWindow) {return}

      if (!ev.data || typeof ev.data !== 'object') {return}

      if (ev.data.event === 'ready' || ev.data.event === 'init') {
        clear()
        iframe.contentWindow?.postMessage(
          { action: 'load', xml: code, mode: 'expanded' },
          '*'
        )
      } else if (ev.data.event === 'error' || ev.data.error) {
        clear()
        setIframeFailed(true)
        onFailure()
      }
    }

    timeoutId = setTimeout(() => {
      setIframeFailed(true)
      onFailure()
    }, RENDER_TIMEOUT_MS)

    window.addEventListener('message', handleMessage)

    return () => {
      clear()
      window.removeEventListener('message', handleMessage)
    }
  }, [code, onFailure])

  return (
    <>
      <iframe
        className="h-full w-full border-0"
        onError={() => {
          setIframeFailed(true)
          onFailure()
        }}
        ref={iframeRef}
        sandbox="allow-scripts"
        src={`${RENDER_URL}?v=${RENDER_VERSION}`}
      />
      {iframeFailed && (
        <div className="absolute inset-0 overflow-auto bg-background p-4">
          <SourcePreview code={code} />
        </div>
      )}
    </>
  )
}
