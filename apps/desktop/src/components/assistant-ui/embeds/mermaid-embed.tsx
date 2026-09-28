'use client'

import mermaid from 'mermaid'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { Zoomable } from '@/components/ui/zoomable'
import { useI18n } from '@/i18n'
import { Code, Eye } from '@/lib/icons'
import { copySvgAsPng, normalizeSvgSize } from '@/lib/svg-image'
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
  const { t } = useI18n()
  const isDark = useIsDark()
  const [svg, setSvg] = useState('')
  const [failed, setFailed] = useState(false)
  // Which source is currently flipped open, not a boolean: a fence whose `code`
  // changes (regenerated message, recycled list row) must come back in diagram
  // view rather than inherit the previous block's choice.
  const [sourceCode, setSourceCode] = useState<string | null>(null)
  const showSource = sourceCode === code

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
          setSvg(normalizeSvgSize(result.svg))
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
  //
  // A rendered diagram hides the text the agent actually wrote, so the reader
  // gets a way back to it: the toggle swaps the SVG for the source and back
  // under the same 33dvh ceiling as the inline diagram, so the source can never
  // grow a settled message past the diagram's own cap. The control is a sibling
  // of the zoom trigger, never a child — anything nested inside that <button>
  // would open the full-screen viewer instead of toggling. Its label names the
  // view the press switches to (as the preview pane's rendered/source switch
  // does), which keeps it a plain action button instead of a toggle whose name
  // contradicts its own pressed state.
  //
  // The reveal lives on a chip cluster rather than on this button, so a second
  // control can sit beside it with one hover target, and so nothing in the
  // corner answers clicks while it is invisible.
  const toggleLabel = showSource ? t.preview.renderedPreview : t.preview.source

  return (
    <div className="group/mermaid relative">
      {showSource ? (
        <div className="max-h-[33dvh] overflow-auto">
          <SourcePreview code={code} />
        </div>
      ) : (
        <Zoomable
          label={t.desktop.openDiagram}
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
      )}
      <div className="pointer-events-none absolute top-2 left-2 flex items-center gap-1 opacity-0 transition-opacity group-hover/mermaid:pointer-events-auto group-hover/mermaid:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100 pointer-coarse:pointer-events-auto pointer-coarse:opacity-100">
        <Tip label={toggleLabel}>
          <Button
            aria-label={toggleLabel}
            className="bg-background/80 backdrop-blur hover:bg-background"
            onClick={() => setSourceCode(current => (current === code ? null : code))}
            size="icon-sm"
            type="button"
            variant="ghost"
          >
            {showSource ? <Eye className="size-3.5" /> : <Code className="size-3.5" />}
          </Button>
        </Tip>
      </div>
    </div>
  )
}
