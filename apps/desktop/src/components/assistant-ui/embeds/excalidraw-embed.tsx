'use client'

import { exportToSvg } from '@excalidraw/excalidraw'
import DOMPurify from 'dompurify'
import { useEffect, useState } from 'react'

import { Zoomable } from '@/components/ui/zoomable'
import { copySvgAsPng } from '@/lib/svg-image'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'
import { useIsDark } from './use-is-dark'

// Re-parses a fenced code block that should hold Excalidraw JSON. The block
// may arrive as a bare JSON object or wrapped in ```excalidraw fences; the
// preprocessor strips the fences, so we deal with the inner JSON here.
function parseExcalidraw(code: string): {
  appState?: Record<string, unknown>
  elements?: unknown[]
  files?: Record<string, unknown>
} | null {
  const trimmed = code.trim()
  if (!trimmed) {
    return null
  }
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed === 'object' && Array.isArray(parsed.elements)) {
      return parsed
    }
    return null
  } catch {
    return null
  }
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

// Lazy chunk (pulls in @excalidraw/excalidraw). Renders ```excalidraw
// fences as a static SVG (exportToSvg) — read-only preview, reuse the same
// pan/zoom + copy-as-PNG affordance as mermaid. Shows the source while the
// message streams (partial JSON throws) and falls back to source on parse/export
// failure.
export default function ExcalidrawRenderer({ code, streaming }: RichFenceProps) {
  const isDark = useIsDark()
  const [svg, setSvg] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (streaming) {
      return
    }

    const scene = parseExcalidraw(code)
    if (!scene) {
      setFailed(true)
      setSvg('')
      return
    }

    let cancelled = false
    setFailed(false)

    void (async () => {
      try {
        const node = await exportToSvg({
          // exportToSvg wants the elements + a (partial) appState. We keep the
          // author's appState when present so view defaults (background, etc.)
          // survive; the export path only reads a safe subset.
          elements: scene.elements as never[],
          appState: {
            ...(scene.appState as Record<string, unknown>),
            exportBackground: true,
            viewBackgroundColor: isDark ? '#1e1e1e' : '#ffffff',
          } as never,
          files: scene.files as Record<string, never> | undefined,
        })

        const serialized = new XMLSerializer().serializeToString(node)
        const clean = DOMPurify.sanitize(serialized, {
          USE_PROFILES: { svg: true, svgFilters: true },
        })

        if (!cancelled) {
          setSvg(clean)
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
