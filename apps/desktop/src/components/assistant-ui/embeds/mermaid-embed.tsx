'use client'

import { useEffect, useState, useSyncExternalStore } from 'react'

import { Zoomable } from '@/components/ui/zoomable'
import { onThemeRepaint } from '@/hooks/use-theme-epoch'
import { copySvgAsPng } from '@/lib/svg-image'
import { cn } from '@/lib/utils'

import { createMermaidRenderCache, createRetryableLoader, nextPaint } from './mermaid-render-cache'
import type { RichFenceProps } from './types'

// Pele (obsidianmd/pele) draws Mermaid syntax as plain SVG. Its lazy entry
// puts each diagram type in its own chunk, so a flowchart pulls in only the
// flowchart parser and layout.
const loadPele = createRetryableLoader(() => import('pele/lazy'))

// Pele colors a diagram through `--pele-*` custom properties. The SVG is shown
// as an <img>: its own document, with no scripts and no access to the app's
// CSS. So the app's tokens are resolved to concrete colors here and written
// onto the SVG root. These are the `--dt-*` tokens: Tailwind's `--color-*`
// names are `@theme inline`, so they never exist as runtime properties.
const PALETTE: ReadonlyArray<readonly [pele: string, token: string]> = [
  ['--pele-bg', '--dt-background'],
  ['--pele-fg', '--dt-foreground'],
  ['--pele-muted', '--dt-muted-foreground'],
  ['--pele-line', '--dt-muted-foreground'],
  ['--pele-surface', '--dt-muted'],
  ['--pele-surface-alt', '--dt-card'],
  ['--pele-border', '--dt-muted-foreground'],
  ['--pele-accent', '--dt-primary'],
  ['--pele-critical', '--dt-destructive']
]

interface DiagramTheme {
  font: string
  fontMono: string
  vars: string
}

// Resolved through probes, so color-mix() and oklch() tokens arrive as values
// the isolated SVG document can use. All probes are styled before any is read,
// so the pass costs one style recalc. A token that does not resolve (unset, or
// a broken var() chain in a skin) is left out, so Pele's own default applies
// instead of the probe's inherited color. Serialised: it doubles as the cache
// key.
const UNRESOLVED = 'rgb(1, 2, 3)'

function resolveTheme(): string {
  const root = getComputedStyle(document.documentElement)
  const host = document.createElement('div')
  host.style.display = 'none'

  const probes = PALETTE.map(([, token]) => {
    const probe = document.createElement('span')
    probe.style.color = `var(${token}, ${UNRESOLVED})`

    return probe
  })

  host.append(...probes)
  document.body.append(host)

  const vars = PALETTE.flatMap(([pele], i) => {
    const color = getComputedStyle(probes[i]).color

    return !color || color === UNRESOLVED ? [] : [`${pele}:${color}`]
  }).join(';')

  host.remove()

  const theme: DiagramTheme = {
    font: root.getPropertyValue('--dt-font-sans').trim() || 'sans-serif',
    fontMono: root.getPropertyValue('--dt-font-mono').trim() || 'monospace',
    vars
  }

  return JSON.stringify(theme)
}

// One resolved theme shared by every mounted diagram, re-resolved once per
// theme repaint (mode flip or skin change) rather than once per diagram.
let sharedTheme: null | string = null
let stopRepaints: (() => void) | null = null
const themeListeners = new Set<() => void>()

function subscribeTheme(listener: () => void): () => void {
  if (!stopRepaints) {
    // Nothing tracked repaints while no diagram was mounted.
    sharedTheme = null
    stopRepaints = onThemeRepaint(() => {
      sharedTheme = null
      themeListeners.forEach(notify => notify())
    })
  }

  themeListeners.add(listener)

  return () => {
    themeListeners.delete(listener)

    if (themeListeners.size === 0) {
      stopRepaints?.()
      stopRepaints = null
      sharedTheme = null
    }
  }
}

const currentTheme = () => (sharedTheme ??= resolveTheme())

// Pele's root always carries a style attribute; the palette goes in front of
// it, so Pele's own `var(--pele-*, default)` references pick it up. Colors are
// computed values (rgb()/color()/oklch()) and carry no quote or ampersand.
function withPalette(svg: string, vars: string): string {
  return vars ? svg.replace(/^<svg\b[^>]*?\sstyle="/, root => `${root}${vars};`) : svg
}

const renderCache = createMermaidRenderCache({
  maxEntries: 32,
  render: async (code, themeKey) => {
    const pele = await loadPele()
    const theme = JSON.parse(themeKey) as DiagramTheme

    // Links and images off: the diagram is display-only, as under mermaid's
    // strict mode (no click handlers, no remote fetches).
    const { svg } = await pele.renderAsync(code, {
      fontFamily: theme.font,
      fontFamilyMono: theme.fontMono,
      images: false,
      links: false
    })

    return withPalette(svg, theme.vars)
  },
  // Defer until the source fallback has had a frame to paint, so the chunk
  // import cannot contend with first paint of the message body (#98967's seam;
  // requestIdleCallback is inert in jsdom).
  defer: nextPaint
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
  // Pele writes <title>/<desc> only for accTitle/accDescr or a diagram title;
  // skip the parse for the common diagram that has neither.
  if (!svg.includes('<title') && !svg.includes('<desc')) {
    return 'Mermaid diagram'
  }

  const document = new DOMParser().parseFromString(svg, 'image/svg+xml')

  const text = [document.querySelector('title')?.textContent, document.querySelector('desc')?.textContent]
    .map(value => value?.trim())
    .filter((value): value is string => Boolean(value))

  return text.join(' — ') || 'Mermaid diagram'
}

// Lazy chunk (pulls in Pele). Renders ```mermaid fences as diagrams; shows the
// source while the message streams (partial syntax throws) and falls back to
// source on parse failure.
export default function MermaidRenderer({ code, streaming }: RichFenceProps) {
  const theme = useSyncExternalStore(subscribeTheme, currentTheme)
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
        const rendered = await renderCache.render(code, theme, controller.signal)

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
  }, [code, theme, streaming])

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
