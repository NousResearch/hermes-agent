'use client'

import type { Mermaid as MermaidApi } from 'mermaid'
import { useEffect, useState } from 'react'

import { Zoomable } from '@/components/ui/zoomable'
import { copySvgAsPng, normalizeSvgSize } from '@/lib/svg-image'
import { cn } from '@/lib/utils'

import { createMermaidRenderCache, createRetryableLoader } from './mermaid-render-cache'
import type { RichFenceProps } from './types'
import { useIsDark } from './use-is-dark'

let lastTheme: 'dark' | 'default' | null = null
const loadMermaid = createRetryableLoader<MermaidApi>(() => import('mermaid').then(module => module.default))

// Re-initialise only on first use / theme flip. `securityLevel: 'strict'` makes
// mermaid sanitise label HTML and drop click handlers, so the rendered SVG is
// safe to inject.
function ensureInit(mermaid: MermaidApi, dark: boolean) {
  const theme = dark ? 'dark' : 'default'

  if (theme === lastTheme) {
    return
  }

  mermaid.initialize({ fontFamily: 'inherit', securityLevel: 'strict', startOnLoad: false, theme })
  lastTheme = theme
}

const renderCache = createMermaidRenderCache({
  maxEntries: 32,
  render: async (code, theme) => {
    const mermaid = await loadMermaid()
    ensureInit(mermaid, theme === 'dark')
    const id = `mmd-${Math.random().toString(36).slice(2)}`
    const result = await mermaid.render(id, code)

    return normalizeSvgSize(result.svg)
  }
})

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

function svgAccessibleText(svg: string): string {
  const document = new DOMParser().parseFromString(svg, 'image/svg+xml')

  const text = [document.querySelector('title')?.textContent, document.querySelector('desc')?.textContent]
    .map(value => value?.trim())
    .filter((value): value is string => Boolean(value))

  return text.join(' — ') || 'Mermaid diagram'
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
    const controller = new AbortController()

    setFailed(false)
    setSvg('')

    void (async () => {
      try {
        const rendered = await renderCache.render(code, isDark ? 'dark' : 'default', controller.signal)

        if (!cancelled) {
          setSvg(rendered)
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
      controller.abort()
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

  const imageSrc = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
  const imageAlt = svgAccessibleText(svg)

  // Click to open the diagram full-screen with pan/zoom + copy-as-PNG. The
  // overlay keeps the diagram's natural width (capped to the viewport) so it
  // renders before any zoom; the inline version stays capped at 33dvh.
  return (
    <Zoomable
      label="Open diagram"
      onCopy={() => copySvgAsPng(svg)}
      overlay={
        <img alt={imageAlt} className="mx-auto h-auto max-h-[80vh] max-w-[85vw]" draggable={false} src={imageSrc} />
      }
    >
      <div className="overflow-hidden p-3">
        <img alt={imageAlt} className="mx-auto h-auto max-h-[33dvh] max-w-full" draggable={false} src={imageSrc} />
      </div>
    </Zoomable>
  )
}
