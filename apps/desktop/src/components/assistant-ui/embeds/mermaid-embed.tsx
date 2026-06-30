'use client'

import mermaid from 'mermaid'
import { useEffect, useState } from 'react'

import { CopyButton } from '@/components/ui/copy-button'
import { Zoomable } from '@/components/ui/zoomable'
import { copySvgAsPng } from '@/lib/svg-image'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'
import { useIsDark } from './use-is-dark'

let lastTheme: 'dark' | 'default' | null = null

// Re-initialise only on first use / theme flip. `securityLevel: 'strict'` makes
// mermaid sanitise label HTML and drop click handlers, so the rendered SVG is
// safe to inject.
function ensureInit(dark: boolean) {
  const theme = dark ? 'dark' : 'default'

  if (theme === lastTheme) {
    return
  }

  mermaid.initialize({ fontFamily: 'inherit', securityLevel: 'strict', startOnLoad: false, theme })
  lastTheme = theme
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

// Lazy chunk (pulls in mermaid). Renders ```mermaid fences as diagrams; shows
// the source while the message streams (partial syntax throws) and falls back
// to source on parse failure.
export default function MermaidRenderer({ code, streaming }: RichFenceProps) {
  const isDark = useIsDark()
  const [svg, setSvg] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (streaming) {
      return
    }

    let cancelled = false

    setFailed(false)

    void (async () => {
      try {
        ensureInit(isDark)
        const id = `mmd-${Math.random().toString(36).slice(2)}`
        const result = await mermaid.render(id, code)

        if (!cancelled) {
          setSvg(result.svg)
        }
      } catch {
        if (!cancelled) {
          setFailed(true)
          setSvg('')
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [code, isDark, streaming])

  if (streaming) {
    return <SourcePreview code={code} muted />
  }

  if (failed) {
    return <SourcePreview code={code} />
  }

  if (!svg) {
    return <SourcePreview code={code} muted />
  }

  // Click to open the diagram full-screen with pan/zoom + copy-as-PNG. The
  // overlay keeps the diagram's natural width (capped to the viewport) so it
  // renders before any zoom; the inline version stays capped at 33dvh.
  return (
    <div className="group/mermaid relative">
      <Zoomable
        label="Open diagram"
        onCopy={() => copySvgAsPng(svg)}
        overlay={
          <div
            className="[&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-h-[80vh] [&_svg]:max-w-[85vw]"
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        }
      >
        <div
          className="overflow-hidden p-3 [&_svg]:mx-auto [&_svg]:h-auto [&_svg]:max-h-[33dvh] [&_svg]:max-w-full"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </Zoomable>
      <CopyButton
        appearance="icon"
        buttonSize="icon-sm"
        className="absolute left-2 top-2 opacity-0 shadow-sm backdrop-blur transition-opacity group-hover/mermaid:opacity-100 focus-visible:opacity-100"
        label="Copy Mermaid source"
        stopPropagation
        text={code}
      />
    </div>
  )
}
