'use client'

import type { Mermaid } from 'mermaid'
import { type ReactNode, useEffect, useState, useSyncExternalStore } from 'react'

import { cn } from '@/lib/utils'

// Lazy singleton: the desktop build inlines all chunks (vite.config.ts sets
// `codeSplitting: false`), so the win here isn't a separate chunk — it's
// deferring evaluation of mermaid's multi-megabyte module graph until the
// first diagram actually appears.
let mermaidPromise: Promise<Mermaid> | null = null

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(mod => mod.default)
  }

  return mermaidPromise
}

// Follow the *rendered* theme exactly as applyTheme() publishes it on <html>
// (`data-hermes-mode`, themes/context.tsx): some skins render light even when
// mode === 'dark', so the dataset is the source of truth — and watching it
// keeps already-rendered diagrams re-theming on toggle.
function subscribeRenderedMode(onChange: () => void) {
  const observer = new MutationObserver(onChange)

  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-hermes-mode'] })

  return () => observer.disconnect()
}

const isDarkSnapshot = () => document.documentElement.dataset.hermesMode === 'dark'

function useRenderedDark(): boolean {
  return useSyncExternalStore(subscribeRenderedMode, isDarkSnapshot, () => false)
}

// mermaid.render() wants a document-unique element id; a module counter keeps
// concurrent blocks from colliding without reaching for randomness.
let renderSeq = 0

interface MermaidBlockProps {
  chart: string
  className?: string
  fallback?: ReactNode
}

/**
 * Renders a ```mermaid fence as an inline SVG diagram. Shows `fallback` (the
 * regular code card) while mermaid loads and whenever the source does not
 * parse, so invalid or unsupported diagrams degrade to highlighted source
 * instead of an error card.
 */
export function MermaidBlock({ chart, className, fallback }: MermaidBlockProps) {
  const isDark = useRenderedDark()
  const [svg, setSvg] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const id = `aui-mermaid-${++renderSeq}`

    setSvg(null)
    loadMermaid()
      .then(async mermaid => {
        // initialize() before each render keeps the singleton's theme in sync
        // with the app; strict securityLevel sanitizes the emitted SVG.
        mermaid.initialize({
          securityLevel: 'strict',
          startOnLoad: false,
          theme: isDark ? 'dark' : 'default'
        })

        if (!(await mermaid.parse(chart, { suppressErrors: true }))) {
          return null
        }

        return mermaid.render(id, chart)
      })
      .then(result => {
        if (!cancelled && result) {
          setSvg(result.svg)
        }
      })
      .catch(() => {
        // render() can still throw past a successful parse (e.g. unsupported
        // directives) and leaves its scratch element behind when it does.
        document.getElementById(id)?.remove()
        document.getElementById(`d${id}`)?.remove()
      })

    return () => {
      cancelled = true
    }
  }, [chart, isDark])

  if (!svg) {
    return <>{fallback}</>
  }

  return (
    <div
      className={cn(
        'my-2 flex max-w-full justify-center overflow-x-auto [&_svg]:h-auto [&_svg]:max-w-full',
        className
      )}
      dangerouslySetInnerHTML={{ __html: svg }}
      data-slot="mermaid-block"
    />
  )
}
