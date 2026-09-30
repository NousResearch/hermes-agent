'use client'

import { useEffect, useState } from 'react'

import { Zoomable } from '@/components/ui/zoomable'
import { copySvgAsPng, normalizeSvgSize } from '@/lib/svg-image'
import { cn } from '@/lib/utils'

import { nextPaint, renderMermaidSvg } from './mermaid-render-cache'
import type { RichFenceProps } from './types'
import { useIsDark } from './use-is-dark'

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

// Lazy chunk (the heavy mermaid runtime is imported inside the render cache's
// deferred callback). Renders ```mermaid fences as diagrams; shows the source
// while the message streams (partial syntax throws) and falls back to source on
// parse failure. Completed diagrams are cached per (source, theme) so remounts
// reuse the SVG instead of re-running parse/layout.
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
        // Let the source fallback paint first; the mermaid runtime import and
        // parse/layout happen only after that frame.
        await nextPaint()

        const { svg: rendered } = await renderMermaidSvg(
          code,
          isDark ? 'dark' : 'default'
        )

        if (!cancelled) {
          setSvg(normalizeSvgSize(rendered))
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
  )
}
