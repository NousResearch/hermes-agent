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
  const [expanded, setExpanded] = useState(false)

  // Create the inline iframe (static mode — no toolbar/nav/pan/zoom)
  useEffect(() => {
    if (streaming) return

    const host = hostRef.current
    if (!host) return

    setFailed(false)
    setReady(false)
    setExpanded(false)

    const iframe = document.createElement('iframe')
    iframe.className = 'h-full w-full border-0'
    iframe.setAttribute('sandbox', 'allow-scripts')

    // Use a versioned URL to bypass browser cache for the render page
    iframe.src = `${RENDER_URL}?v=${RENDER_VERSION}`

    const handleLoad = () => {
      setReady(true)
      // Send the XML to the receiver page in inline mode (static)
      iframe.contentWindow?.postMessage({ action: 'load', xml: code, mode: 'inline' }, '*')
    }

    iframe.addEventListener('load', handleLoad)
    iframe.addEventListener('error', () => setFailed(true))

    host.replaceChildren(iframe)
    return () => {
      host.replaceChildren()
    }
  }, [code, streaming])

  const openExpanded = useCallback(() => setExpanded(true), [])
  const closeExpanded = useCallback(() => setExpanded(false), [])

  if (streaming) return <SourcePreview code={code} muted />
  if (failed) return <SourcePreview code={code} />

  return (
    <>
      <div className="group/zoomable relative">
        <div
          ref={hostRef}
          className="h-[33dvh] w-full overflow-hidden rounded-lg border border-border bg-muted/30"
        />
        {!ready && <SourcePreview code={code} muted />}
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
              <iframe
                className="h-full w-full border-0"
                onLoad={e => {
                  e.currentTarget.contentWindow?.postMessage(
                    { action: 'load', xml: code, mode: 'expanded' },
                    '*'
                  )
                }}
                sandbox="allow-scripts"
                src={`${RENDER_URL}?v=${RENDER_VERSION}`}
              />
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
